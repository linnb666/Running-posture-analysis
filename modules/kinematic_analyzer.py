# modules/kinematic_analyzer.py
"""
重构版运动学特征解析模块 (3D增强版)

核心改进：
1. 垂直振幅使用躯干长度归一化（解决单位不匹配问题）
2. 膝关节角度分阶段分析（触地期、摆动期、蹬离期）
3. 支持不同视角的分析策略
4. 更精确的步态周期检测
5. 【新增】MotionBERT 3D姿态支持，提供真正的3D关节角度
6. 【新增】数据可靠性标注（可靠/不可靠）

当检测到3D数据时，优先使用3D计算关节角度，提高准确性
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from typing import List, Dict, Tuple, Optional
from config.config import (
    BODY_LEAN_THRESHOLDS,
    KINEMATIC_CONFIG,
    VERTICAL_AMPLITUDE_THRESHOLDS,
)

# 导入工具函数
from modules.kinematic_utils import (
    calculate_3d_joint_angle,
    project_point_to_plane,
    calculate_projected_joint_angle,
    get_3d_point_from_pose,
)


class KinematicAnalyzer:
    """重构版运动学分析器"""

    # 跑步阶段定义
    PHASE_GROUND_CONTACT = 0  # 触地期
    PHASE_FLIGHT = 1          # 腾空期
    PHASE_TRANSITION = 2      # 过渡期

    def __init__(self):
        """初始化分析器"""
        self.smooth_window = KINEMATIC_CONFIG['smooth_window']

        # 3D数据相关属性
        self.has_3d_data = False
        self.poses_3d = None
        self.keypoints_3d = None
        self.view_angle = None  # 存储视角用于投影计算
        self.valid_frame_indices = []

    def analyze_sequence(self, keypoints_sequence: List[Dict], fps: float,
                         view_angle: str = 'side',
                         poses_3d: np.ndarray = None,
                         keypoints_3d: List[Dict] = None) -> Dict:
        """
        分析完整关键点序列

        Args:
            keypoints_sequence: 关键点时间序列（2D）
            fps: 帧率
            view_angle: 视频视角 ('side', 'front', 'back')
            poses_3d: 可选，(T, 17, 3) 3D姿态数组（来自MotionBERT）
            keypoints_3d: 可选，3D关键点序列（带pose_3d字段）

        Returns:
            分析结果字典，包含数据可靠性标注
        """
        # 存储视角（用于3D角度投影计算）
        self.view_angle = view_angle

        # 检测是否有3D数据
        self.has_3d_data = poses_3d is not None or (
            keypoints_3d is not None and len(keypoints_3d) > 0 and
            keypoints_3d[0].get('has_3d', False)
        )

        # 存储3D数据引用
        self.poses_3d = poses_3d
        self.keypoints_3d = keypoints_3d

        if self.has_3d_data:
            print("✅ 检测到3D姿态数据，将使用3D坐标计算关节角度")
            print(f"   📐 视角投影: {view_angle}视角 → {'XY平面(矢状面)' if view_angle == 'side' else 'YZ平面(冠状面)'}")
        else:
            print("ℹ️ 未检测到3D数据，使用2D投影计算（可能不够准确）")

        # 提取有效帧
        self.valid_frame_indices = [i for i, kp in enumerate(keypoints_sequence) if kp['detected']]
        valid_frames = [kp for kp in keypoints_sequence if kp['detected']]

        if len(valid_frames) < 10:
            return self._get_empty_analysis()

        print(f"有效帧数: {len(valid_frames)}/{len(keypoints_sequence)}")

        # 计算躯干参考长度（用于归一化）
        trunk_length = self._calculate_trunk_reference(valid_frames)
        print(f"躯干参考长度: {trunk_length:.4f} (归一化坐标)")

        # 基础运动学指标（所有视角通用）
        results = {
            'fps': fps,
            'total_frames': len(keypoints_sequence),
            'valid_frames': len(valid_frames),
            'view_angle': view_angle,
            'trunk_reference': trunk_length,
        }

        # 根据视角选择分析策略
        if view_angle == 'side':
            results.update(self._analyze_side_view(valid_frames, fps, trunk_length))
        elif view_angle in ['front', 'back']:
            results.update(self._analyze_frontal_view(valid_frames, fps, trunk_length))
        else:
            # 混合分析
            results.update(self._analyze_side_view(valid_frames, fps, trunk_length))

        return results

    def _calculate_trunk_reference(self, keypoints_sequence: List[Dict]) -> float:
        """
        计算躯干参考长度（肩到髋的平均距离）
        用于将垂直振幅归一化为相对身体尺度的比例
        """
        trunk_lengths = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # 左侧躯干长度（只用Y轴差，避免水平摇晃影响）
            left_shoulder = landmarks[11]
            left_hip = landmarks[23]
            if left_shoulder['visibility'] > 0.5 and left_hip['visibility'] > 0.5:
                left_trunk = abs(left_shoulder['y_norm'] - left_hip['y_norm'])
                trunk_lengths.append(left_trunk)

            # 右侧躯干长度（只用Y轴差）
            right_shoulder = landmarks[12]
            right_hip = landmarks[24]
            if right_shoulder['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
                right_trunk = abs(right_shoulder['y_norm'] - right_hip['y_norm'])
                trunk_lengths.append(right_trunk)

        if trunk_lengths:
            return np.median(trunk_lengths)  # 使用中位数更稳健
        return 0.25  # 默认值（约占画面1/4）

    def _calculate_vertical_reference_length_2d(self, keypoints_sequence: List[Dict]) -> float:
        """
        计算垂直振幅专用的2D参考长度

        与通用trunk_reference不同，这里使用肩到髋的欧氏距离而非纯Y差，
        以减少侧面前倾时分母被压小、振幅被高估的问题。
        """
        trunk_lengths = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']
            for shoulder_idx, hip_idx in ((11, 23), (12, 24)):
                shoulder = landmarks[shoulder_idx]
                hip = landmarks[hip_idx]
                if shoulder['visibility'] > 0.5 and hip['visibility'] > 0.5:
                    dx = shoulder['x_norm'] - hip['x_norm']
                    dy = shoulder['y_norm'] - hip['y_norm']
                    trunk_lengths.append(float(np.hypot(dx, dy)))

        if trunk_lengths:
            filtered = self._filter_outliers_iqr(np.array(trunk_lengths), factor=1.0)
            return float(np.median(filtered))

        return float(self._calculate_trunk_reference(keypoints_sequence))

    def _estimate_oscillation_period_frames(self, signal: np.ndarray, fps: float,
                                            min_hz: float = 2.0,
                                            max_hz: float = 4.2) -> Optional[int]:
        """
        估计周期信号的主周期（帧）

        对垂直振荡而言，主频通常与步频一致。这里结合自相关和FFT做稳健估计，
        仅用于约束峰值间隔和周期分段，不改变返回字段结构。
        """
        if fps <= 0 or len(signal) < max(12, int(fps * 1.2)):
            return None

        centered = np.asarray(signal, dtype=float) - float(np.mean(signal))
        if np.std(centered) < 1e-6:
            return None

        min_lag = max(3, int(fps / max_hz))
        max_lag = min(len(centered) // 2, max(min_lag + 1, int(fps / min_hz)))
        if max_lag <= min_lag:
            return None

        autocorr = np.correlate(centered, centered, mode='full')[len(centered) - 1:]
        autocorr_window = autocorr[min_lag:max_lag + 1]
        if len(autocorr_window) == 0:
            return None

        best_lag = min_lag + int(np.argmax(autocorr_window))
        best_value = float(np.max(autocorr_window))
        if best_value <= 0:
            return None

        fft_period = None
        freqs = fftfreq(len(centered), d=1.0 / fps)
        spectrum = np.abs(fft(centered))
        valid = (freqs >= min_hz) & (freqs <= max_hz)
        if np.any(valid):
            valid_freqs = freqs[valid]
            valid_power = spectrum[valid]
            if len(valid_power) > 0 and np.max(valid_power) > 0:
                dominant_freq = float(valid_freqs[int(np.argmax(valid_power))])
                if dominant_freq > 0:
                    fft_period = fps / dominant_freq

        if fft_period is not None:
            if abs(best_lag - fft_period) / max(best_lag, fft_period) < 0.25:
                return int(round((best_lag + fft_period) / 2.0))

        return int(best_lag)

    def _calculate_cycle_amplitudes(self, signal: np.ndarray, fps: float,
                                    estimated_period_frames: Optional[int]) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        基于周期分段计算垂直振幅

        使用相邻峰值/谷值之间的分段来计算每个周期的峰谷差，
        比直接在整段序列上做全局峰谷配对更稳健。
        """
        if len(signal) < 10:
            return np.array([], dtype=int), np.array([], dtype=int), []

        signal_std = float(np.std(signal))
        signal_iqr = float(np.subtract(*np.percentile(signal, [75, 25])))
        min_distance = max(5, int((estimated_period_frames or max(fps * 0.3, 6)) * 0.8))
        prominence = max(signal_std * 0.3, signal_iqr * 0.18, 1e-4)

        peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence)
        troughs, _ = find_peaks(-signal, distance=min_distance, prominence=prominence)

        cycle_amplitudes = []

        def collect_amplitudes(anchors: np.ndarray):
            if len(anchors) < 2:
                return
            for start_idx, end_idx in zip(anchors[:-1], anchors[1:]):
                span = int(end_idx - start_idx)
                if span < 3:
                    continue
                if estimated_period_frames:
                    lower = max(4, int(estimated_period_frames * 0.75))
                    upper = int(estimated_period_frames * 1.25)
                    if span < lower or span > upper:
                        continue
                segment = signal[start_idx:end_idx + 1]
                if len(segment) < 3:
                    continue
                cycle_amplitudes.append(float(np.max(segment) - np.min(segment)))

        collect_amplitudes(peaks)
        collect_amplitudes(troughs)

        if cycle_amplitudes:
            filtered = self._filter_outliers_iqr(np.array(cycle_amplitudes), factor=1.0)
            cycle_amplitudes = [float(x) for x in filtered]

        return peaks, troughs, cycle_amplitudes

    def _robust_cycle_amplitude(self, cycle_amplitudes: List[float]) -> float:
        """
        稳健汇总各周期振幅

        使用截尾均值替代纯中位数，减少被极小周期振幅压低的问题，
        同时仍保持对异常大峰值的鲁棒性。
        """
        if not cycle_amplitudes:
            return 0.0

        arr = np.sort(np.asarray(cycle_amplitudes, dtype=float))
        if len(arr) < 5:
            return float(np.mean(arr))

        trim = max(1, int(len(arr) * 0.15))
        if len(arr) - trim * 2 >= 3:
            arr = arr[trim:-trim]

        return float(np.mean(arr))

    def _extract_vertical_oscillation_component(self, positions: list, fps: float) -> np.ndarray:
        """
        提取垂直振荡分量

        与横摆分量不同，垂直振荡的主频通常更接近 2-4Hz。
        这里保留趋势去除，但放宽低通截止频率，避免把真实步态振幅压小。
        """
        arr = np.array(positions, dtype=float)
        n = len(arr)

        if n < 8:
            return arr - np.mean(arr)

        trend_window = max(7, int(n * 0.4))
        if trend_window % 2 == 0:
            trend_window += 1
        if trend_window > n:
            trend_window = n if n % 2 == 1 else n - 1

        trajectory = savgol_filter(arr, trend_window, polyorder=2)
        oscillation = arr - trajectory

        if fps > 0 and n >= 15:
            nyquist = fps / 2.0
            cutoff = min(4.5 / nyquist, 0.95)
            if cutoff > 0.05:
                b, a = butter(2, cutoff, btype='low')
                oscillation = filtfilt(b, a, oscillation)

        return oscillation

    # ==================== 侧面视角分析 ====================

    def _analyze_side_view(self, valid_frames: List[Dict], fps: float,
                           trunk_length: float) -> Dict:
        """侧面视角分析 - 主要分析策略"""
        results = {
            # 核心指标
            'angles': self._calculate_angles_by_phase(valid_frames, fps),
            'vertical_motion': self._calculate_vertical_motion_normalized(
                valid_frames, fps, trunk_length
            ),
            'cadence': self._calculate_cadence_improved(valid_frames, fps),
            'stride_info': self._calculate_stride_info(valid_frames, fps),

            # 稳定性指标（侧面不评估左右对称性）
            'stability': self._calculate_stability_side_view(valid_frames),
            'body_lean': self._calculate_body_lean(valid_frames, fps),
            'arm_swing': self._calculate_arm_swing(valid_frames),

            # 步态周期分析（包含毫秒时间）
            'gait_cycle': self._analyze_gait_cycle(valid_frames, fps),
        }
        return results

    def _calculate_angles_by_phase(self, keypoints_sequence: List[Dict],
                                    fps: float) -> Dict:
        """
        分阶段计算关节角度

        核心改进：
        1. 区分触地期、摆动期、蹬离期的角度
        2. 【3D增强】如有3D数据，使用3D坐标计算真实角度
        """
        # 收集原始角度数据
        knee_angles_left = []
        knee_angles_right = []
        hip_angles_left = []
        hip_angles_right = []
        ankle_angles_left = []
        ankle_angles_right = []

        # 判断是否使用3D计算（只有3D可用时才计算膝关节角度）
        use_3d = self.has_3d_data and self.keypoints_3d is not None

        for i, kp in enumerate(keypoints_sequence):
            landmarks = kp['landmarks']

            # 【仅3D模式】膝关节角度只在3D数据可用时计算
            if use_3d and i < len(self.keypoints_3d) and self.keypoints_3d[i].get('has_3d', False):
                pose_3d = self.keypoints_3d[i].get('pose_3d', {})

                # 3D膝关节角度（髋-膝-踝）
                knee_angles_left.append(self._calculate_joint_angle_3d(
                    pose_3d, 'l_hip', 'l_knee', 'l_ankle'
                ))
                knee_angles_right.append(self._calculate_joint_angle_3d(
                    pose_3d, 'r_hip', 'r_knee', 'r_ankle'
                ))

                # 3D髋关节角度（肩-髋-膝）
                hip_angles_left.append(self._calculate_joint_angle_3d(
                    pose_3d, 'l_shoulder', 'l_hip', 'l_knee'
                ))
                hip_angles_right.append(self._calculate_joint_angle_3d(
                    pose_3d, 'r_shoulder', 'r_hip', 'r_knee'
                ))

            # 踝关节角度始终使用2D（3D缺少足部关键点）
            ankle_angles_left.append(self._calculate_joint_angle_safe(
                landmarks[25], landmarks[27], landmarks[31]
            ))
            ankle_angles_right.append(self._calculate_joint_angle_safe(
                landmarks[26], landmarks[28], landmarks[32]
            ))

        # 平滑处理
        knee_left_smooth = self._smooth_and_filter_angles(knee_angles_left)
        knee_right_smooth = self._smooth_and_filter_angles(knee_angles_right)
        hip_left_smooth = self._smooth_and_filter_angles(hip_angles_left)
        hip_right_smooth = self._smooth_and_filter_angles(hip_angles_right)
        ankle_left_smooth = self._smooth_and_filter_angles(ankle_angles_left)
        ankle_right_smooth = self._smooth_and_filter_angles(ankle_angles_right)

        # 检测步态阶段
        phases = self._detect_gait_phases(keypoints_sequence, fps)

        # 分阶段统计膝关节角度
        phase_angles = self._analyze_angles_by_phase(
            knee_left_smooth, knee_right_smooth, phases
        )

        # ⭐ 使用新方法计算落地膝角（方案A：取峰值前的帧）
        landing_angles_result = self._calculate_landing_knee_angles(
            keypoints_sequence, fps, knee_left_smooth, knee_right_smooth
        )

        # 将落地膝角结果合并到触地期统计中（包含每步统计）
        if landing_angles_result['landing_count'] > 0:
            landing_angles = landing_angles_result['landing_angles']
            phase_angles['ground_contact']['landing_angle_mean'] = landing_angles_result['landing_angle_mean']
            phase_angles['ground_contact']['landing_angle_std'] = landing_angles_result.get('landing_angle_std', 0)
            phase_angles['ground_contact']['landing_count'] = landing_angles_result['landing_count']
            phase_angles['ground_contact']['landing_angles'] = landing_angles
            phase_angles['ground_contact']['per_step_stats'] = landing_angles_result.get('per_step_stats', [])
            # 使用落地瞬间的角度作为主要指标（包括min/max，确保数据一致性）
            phase_angles['ground_contact']['mean'] = landing_angles_result['landing_angle_mean']
            phase_angles['ground_contact']['std'] = landing_angles_result.get('landing_angle_std', 0)
            phase_angles['ground_contact']['min'] = float(np.min(landing_angles)) if landing_angles else 0
            phase_angles['ground_contact']['max'] = float(np.max(landing_angles)) if landing_angles else 0
            phase_angles['ground_contact']['count'] = len(landing_angles)

            # 重新计算 rating，确保与新的 mean 一致
            new_mean = landing_angles_result['landing_angle_mean']
            if 150 <= new_mean <= 165:
                new_rating = {'level': 'optimal', 'score': 100, 'description': '落地膝角理想，缓冲良好'}
            elif 140 <= new_mean <= 170:
                new_rating = {'level': 'acceptable', 'score': 75, 'description': '落地膝角可接受'}
            elif new_mean > 0:
                new_rating = {'level': 'poor', 'score': 60, 'description': '落地膝角偏离标准'}
            else:
                new_rating = {'level': 'unknown', 'score': 0, 'description': '数据不足'}
            phase_angles['ground_contact']['rating'] = new_rating

        # 检查是否有有效的膝角数据
        has_knee_data = len(knee_left_smooth) > 0 and not np.all(np.isnan(knee_left_smooth))

        # 构建返回结果
        result = {
            # 原始时间序列（可能为空）
            'knee_left': knee_left_smooth if has_knee_data else [],
            'knee_right': knee_right_smooth if has_knee_data else [],
            'hip_left': hip_left_smooth if has_knee_data else [],
            'hip_right': hip_right_smooth if has_knee_data else [],
            'ankle_left': ankle_left_smooth,
            'ankle_right': ankle_right_smooth,

            # 数据可靠性标注
            'data_reliability': {
                'is_3d': use_3d,
                'has_knee_data': has_knee_data,
                'reliability': 'high' if use_3d and has_knee_data else 'unavailable',
                'description': '使用MotionBERT 3D姿态计算，数据可靠' if use_3d and has_knee_data else '3D数据不可用，无法提供可靠的膝关节角度',
                'recommendation': None if use_3d and has_knee_data else '需要3D姿态数据才能计算膝关节角度'
            }
        }

        # 只有在有膝角数据时才添加统计量
        if has_knee_data:
            result.update({
                'knee_left_mean': float(np.nanmean(knee_left_smooth)),
                'knee_right_mean': float(np.nanmean(knee_right_smooth)),
                'knee_left_std': float(np.nanstd(knee_left_smooth)),
                'knee_right_std': float(np.nanstd(knee_right_smooth)),
                'knee_rom': float(np.nanmax(knee_left_smooth) - np.nanmin(knee_left_smooth)),
                'phase_analysis': phase_angles,
            })
        else:
            result.update({
                'knee_left_mean': 0,
                'knee_right_mean': 0,
                'knee_left_std': 0,
                'knee_right_std': 0,
                'knee_rom': 0,
                'phase_analysis': {
                    'ground_contact': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0},
                    'flight': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0},
                    'transition': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0},
                    'max_flexion': 0,
                    'max_extension': 0,
                    'range_of_motion': 0,
                },
            })
            print("   ⚠️ 无3D数据，膝关节角度分析不可用")

        return result

    def _detect_gait_phases(self, keypoints_sequence: List[Dict],
                            fps: float) -> List[int]:
        """
        检测每一帧的步态阶段（重写版：基于速度的状态机）

        核心改进：
        1. 使用脚踝Y坐标速度判断状态
        2. 基于脚踝高度（Y值）确定触地/腾空
        3. 自适应阈值（按帧率缩放）
        """
        n_frames = len(keypoints_sequence)
        if n_frames < 3:
            return [self.PHASE_TRANSITION] * n_frames

        # 提取脚踝Y坐标
        left_ankle_y = []
        right_ankle_y = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][27]
            right = kp['landmarks'][28]

            left_y = left['y_norm'] if left['visibility'] > 0.5 else np.nan
            right_y = right['y_norm'] if right['visibility'] > 0.5 else np.nan

            left_ankle_y.append(left_y)
            right_ankle_y.append(right_y)

        # 插值和平滑
        left_ankle_y = self._interpolate_nans(np.array(left_ankle_y))
        right_ankle_y = self._interpolate_nans(np.array(right_ankle_y))

        if len(left_ankle_y) > 5:
            left_ankle_y = self._smooth_signal_advanced(left_ankle_y)
            right_ankle_y = self._smooth_signal_advanced(right_ankle_y)

        # 计算每帧取较低脚踝的Y值（触地的脚）
        lower_ankle_y = np.maximum(left_ankle_y, right_ankle_y)  # Y轴向下，值大=位置低

        # 计算速度（Y坐标变化率）
        velocity = np.gradient(lower_ankle_y) * fps

        # 自适应阈值计算
        y_median = np.median(lower_ankle_y)
        y_std = np.std(lower_ankle_y)
        y_max = np.max(lower_ankle_y)  # 最低点（触地）
        y_min = np.min(lower_ankle_y)  # 最高点（腾空）
        y_range = y_max - y_min

        # 阈值设置（基于数据分布）
        ground_threshold = y_max - y_range * 0.25  # 上方25%范围为触地
        flight_threshold = y_min + y_range * 0.35  # 下方35%范围为腾空
        velocity_threshold = y_std * fps * 0.3  # 速度阈值

        # 状态机检测相位
        phases = []
        for i in range(n_frames):
            y = lower_ankle_y[i]
            v = abs(velocity[i]) if i < len(velocity) else 0

            if y >= ground_threshold and v < velocity_threshold * 2:
                # 脚踝位置低且速度小 = 触地
                phases.append(self.PHASE_GROUND_CONTACT)
            elif y <= flight_threshold:
                # 脚踝位置高 = 腾空
                phases.append(self.PHASE_FLIGHT)
            else:
                # 中间区域 = 过渡
                phases.append(self.PHASE_TRANSITION)

        # 后处理：消除孤立状态（至少连续2帧才算有效状态）
        phases = self._smooth_phases(phases, min_duration=2)

        return phases

    def _smooth_phases(self, phases: List[int], min_duration: int = 2) -> List[int]:
        """平滑相位序列，消除孤立的错误检测"""
        if len(phases) < min_duration * 2:
            return phases

        smoothed = phases.copy()

        # 检测并修复孤立状态
        i = 0
        while i < len(smoothed):
            # 找到当前状态的连续区间
            start = i
            current_phase = smoothed[i]
            while i < len(smoothed) and smoothed[i] == current_phase:
                i += 1
            duration = i - start

            # 如果持续时间太短，用前后状态替换
            if duration < min_duration and start > 0 and i < len(smoothed):
                # 使用前一个状态填充
                prev_phase = smoothed[start - 1]
                for j in range(start, i):
                    smoothed[j] = prev_phase

        return smoothed

    def _analyze_angles_by_phase(self, knee_left: List[float],
                                  knee_right: List[float],
                                  phases: List[int]) -> Dict:
        """
        按阶段统计膝关节角度（增强版：结合膝角变化率）

        关键改进：
        1. 使用膝角变化率辅助判断真正的摆动期
        2. 摆动期定义：膝角 < 阈值 且 处于屈曲趋势
        3. 触地期定义：膝角 > 阈值 且 处于伸展状态
        """
        n_frames = min(len(knee_left), len(knee_right), len(phases))
        if n_frames < 5:
            return self._get_empty_phase_stats()

        # 转换为numpy数组便于计算
        knee_left_arr = np.array(knee_left[:n_frames])
        knee_right_arr = np.array(knee_right[:n_frames])

        # 计算每帧的最小膝角（摆动腿）和最大膝角（支撑腿）
        min_knee = np.array([
            min(l, r) if not (np.isnan(l) and np.isnan(r)) else np.nan
            for l, r in zip(knee_left_arr, knee_right_arr)
        ])
        max_knee = np.array([
            max(l, r) if not (np.isnan(l) and np.isnan(r)) else np.nan
            for l, r in zip(knee_left_arr, knee_right_arr)
        ])

        # 处理NaN：插值
        min_knee = self._interpolate_nans(min_knee)
        max_knee = self._interpolate_nans(max_knee)

        # 计算膝角变化率（用于判断屈曲/伸展趋势）
        min_knee_rate = np.gradient(min_knee) if len(min_knee) > 1 else np.zeros_like(min_knee)

        # 动态阈值：基于数据分布
        valid_min = min_knee[~np.isnan(min_knee)]
        if len(valid_min) < 5:
            return self._get_empty_phase_stats()

        # 摆动期阈值：角度小于中位数-0.5*标准差 视为明显屈曲
        swing_threshold = np.median(valid_min) - 0.3 * np.std(valid_min)
        # 确保阈值在合理范围内
        swing_threshold = max(100, min(150, swing_threshold))

        # 分组收集各阶段的角度
        ground_contact_angles = []
        flight_angles = []
        transition_angles = []

        for i in range(n_frames):
            if np.isnan(min_knee[i]) or np.isnan(max_knee[i]):
                continue

            phase = phases[i] if i < len(phases) else self.PHASE_TRANSITION
            rate = min_knee_rate[i]

            # 增强判断：结合原始相位和膝角状态
            is_swing_by_angle = min_knee[i] < swing_threshold
            is_flexing = rate < -0.5  # 膝角快速减小 = 屈曲中

            if phase == self.PHASE_GROUND_CONTACT:
                # 触地期：取支撑腿角度（较大的）
                ground_contact_angles.append(max_knee[i])
            elif phase == self.PHASE_FLIGHT or is_swing_by_angle:
                # 腾空/摆动期：膝角小于阈值，或原始检测为腾空
                # 取摆动腿角度（较小的）
                flight_angles.append(min_knee[i])
            else:
                # 过渡期
                transition_angles.append((min_knee[i] + max_knee[i]) / 2)

        # 如果腾空期样本太少，尝试从角度最小的帧中补充
        if len(flight_angles) < 5:
            # 找出膝角最小的N个帧（这些很可能是真正的摆动期）
            sorted_indices = np.argsort(min_knee)
            top_n = min(20, len(sorted_indices) // 5)  # 取最小的20%
            for idx in sorted_indices[:top_n]:
                if not np.isnan(min_knee[idx]) and min_knee[idx] not in flight_angles:
                    flight_angles.append(min_knee[idx])

        # 计算各阶段统计量
        def safe_stats(angles):
            if len(angles) > 0:
                return {
                    'mean': float(np.mean(angles)),
                    'std': float(np.std(angles)),
                    'min': float(np.min(angles)),
                    'max': float(np.max(angles)),
                    'count': len(angles)
                }
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}

        # 计算关键时刻角度（分别考虑左右腿）
        knee_left_valid = [x for x in knee_left if not np.isnan(x)]
        knee_right_valid = [x for x in knee_right if not np.isnan(x)]

        # 最大屈曲：使用摆动期(flight)的平均角度，而非全序列最小值
        # 这样更能反映实际跑步时的典型屈曲角度
        if len(flight_angles) >= 3:
            max_flexion = float(np.mean(flight_angles))
        else:
            # 数据不足时回退到原始方法，但标记为不可靠
            all_min_angles = []
            if knee_left_valid:
                all_min_angles.append(np.min(knee_left_valid))
            if knee_right_valid:
                all_min_angles.append(np.min(knee_right_valid))
            max_flexion = float(min(all_min_angles)) if all_min_angles else 0

        # 最大伸展：使用触地期(ground_contact)的平均角度
        if len(ground_contact_angles) >= 3:
            max_extension = float(np.mean(ground_contact_angles))
        else:
            # 数据不足时回退到原始方法
            all_max_angles = []
            if knee_left_valid:
                all_max_angles.append(np.max(knee_left_valid))
            if knee_right_valid:
                all_max_angles.append(np.max(knee_right_valid))
            max_extension = float(max(all_max_angles)) if all_max_angles else 0

        # 关节活动范围：触地期平均 - 摆动期平均
        # 这更能反映单个步态周期内的实际活动范围
        if len(ground_contact_angles) >= 3 and len(flight_angles) >= 3:
            range_of_motion = max_extension - max_flexion
        else:
            range_of_motion = 0  # 数据不足时不计算

        # 触地期统计
        gc_stats = safe_stats(ground_contact_angles)

        # 腾空期统计
        fl_stats = safe_stats(flight_angles)

        # 过渡期统计
        tr_stats = safe_stats(transition_angles)

        # 为各阶段添加rating评级
        def calc_phase_rating(mean_angle: float, phase_type: str) -> Dict:
            """计算膝角各阶段的评级"""
            if mean_angle <= 0:
                return {'level': 'unknown', 'score': 0, 'description': '数据不足'}

            if phase_type == 'ground_contact':
                # 触地期理想范围：150-165°（约15-30°屈曲）
                if 150 <= mean_angle <= 165:
                    return {'level': 'optimal', 'score': 100, 'description': '落地膝角理想，缓冲良好'}
                elif 140 <= mean_angle <= 170:
                    return {'level': 'acceptable', 'score': 75, 'description': '落地膝角可接受'}
                else:
                    return {'level': 'poor', 'score': 50, 'description': '落地膝角偏离标准'}
            elif phase_type == 'flight':
                # 腾空期理想范围：90-130°（明显屈曲）
                if 90 <= mean_angle <= 130:
                    return {'level': 'optimal', 'score': 100, 'description': '摆动期膝角理想'}
                elif 80 <= mean_angle <= 140:
                    return {'level': 'acceptable', 'score': 75, 'description': '摆动期膝角可接受'}
                else:
                    return {'level': 'poor', 'score': 50, 'description': '摆动期膝角偏离标准'}
            else:
                return {'level': 'unknown', 'score': 60, 'description': '过渡阶段'}

        # 添加rating到各阶段统计
        gc_stats['rating'] = calc_phase_rating(gc_stats['mean'], 'ground_contact')
        fl_stats['rating'] = calc_phase_rating(fl_stats['mean'], 'flight')
        tr_stats['rating'] = calc_phase_rating(tr_stats['mean'], 'transition')

        return {
            # 触地期角度（理想范围：150-165°）
            'ground_contact': gc_stats,
            # 腾空期/摆动期角度（理想范围：90-130°，弯曲较大）
            'flight': fl_stats,
            # 过渡期角度
            'transition': tr_stats,
            # 关键指标
            'max_flexion': max_flexion,
            'max_extension': max_extension,
            'range_of_motion': range_of_motion,
        }

    def _detect_landing_windows(self, keypoints_sequence: List[Dict], fps: float) -> List[Dict]:
        """
        检测落地区间（重构版：基于周期性 + 相对阈值）

        核心改进：
        1. 检测落地"区间"而非单一帧
        2. 使用IQR/百分位数自适应阈值
        3. 周期性验证（相邻落地间隔应相似）
        4. 分别检测左右脚落地

        返回：落地区间列表，每个包含 {start, peak, end, foot, duration_ms}
        """
        n_frames = len(keypoints_sequence)
        if n_frames < 15:
            return []

        # 提取左右脚踝Y坐标
        left_ankle_y = []
        right_ankle_y = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][27]
            right = kp['landmarks'][28]
            left_y = left['y_norm'] if left['visibility'] > 0.5 else np.nan
            right_y = right['y_norm'] if right['visibility'] > 0.5 else np.nan
            left_ankle_y.append(left_y)
            right_ankle_y.append(right_y)

        left_ankle_y = self._interpolate_nans(np.array(left_ankle_y))
        right_ankle_y = self._interpolate_nans(np.array(right_ankle_y))

        # 轻度平滑（Savitzky-Golay，保留峰值特征）
        window_size = min(7, len(left_ankle_y) // 3)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size >= 3:
            left_ankle_y = savgol_filter(left_ankle_y, window_size, 2)
            right_ankle_y = savgol_filter(right_ankle_y, window_size, 2)

        # 分别检测左右脚落地区间
        left_windows = self._detect_foot_landing_windows(left_ankle_y, fps, 'left')
        right_windows = self._detect_foot_landing_windows(right_ankle_y, fps, 'right')

        # 合并并按时间排序
        all_windows = left_windows + right_windows
        all_windows.sort(key=lambda x: x['peak'])

        # 周期性验证：过滤异常间隔的落地
        validated_windows = self._validate_landing_periodicity(all_windows, fps)

        print(f"  落地区间检测：左脚{len(left_windows)}次，右脚{len(right_windows)}次")
        print(f"  周期性验证后：{len(validated_windows)}次有效落地")

        return validated_windows

    def _detect_foot_landing_windows(self, ankle_y: np.ndarray, fps: float, foot: str) -> List[Dict]:
        """
        检测单脚的落地区间

        使用相对阈值（基于IQR）而非硬编码常数
        """
        n = len(ankle_y)
        if n < 10:
            return []

        # 计算自适应阈值（基于IQR）
        q25 = np.percentile(ankle_y, 25)
        q75 = np.percentile(ankle_y, 75)
        iqr = q75 - q25
        median = np.median(ankle_y)

        # 峰值检测参数（基于数据分布）
        # prominence：至少要突出IQR的30%
        min_prominence = iqr * 0.3
        # distance：基于典型步频（150-200步/分），最小间隔约0.3秒
        min_distance = max(3, int(fps * 0.25))

        # 检测峰值（Y值最大=位置最低=触地）
        peaks, properties = find_peaks(
            ankle_y,
            prominence=max(min_prominence, 0.01),  # 至少0.01防止太小
            distance=min_distance
        )

        if len(peaks) == 0:
            return []

        # 为每个峰值确定落地区间
        landing_windows = []
        prominences = properties['prominences']

        # 区间边界阈值：峰值高度减去prominence的50%
        for i, peak in enumerate(peaks):
            peak_value = ankle_y[peak]
            prominence = prominences[i]
            boundary_threshold = peak_value - prominence * 0.5

            # 向前找区间开始
            start = peak
            for j in range(peak - 1, max(0, peak - int(fps * 0.2)) - 1, -1):
                if ankle_y[j] < boundary_threshold:
                    start = j + 1
                    break

            # 向后找区间结束
            end = peak
            for j in range(peak + 1, min(n, peak + int(fps * 0.2))):
                if ankle_y[j] < boundary_threshold:
                    end = j
                    break

            duration_frames = int(end - start + 1)
            duration_ms = float(duration_frames * 1000.0 / fps)

            # 合理性检查：触地区间应在50-300ms之间
            if 50 <= duration_ms <= 300:
                landing_windows.append({
                    'start': int(start),
                    'peak': int(peak),
                    'end': int(end),
                    'foot': str(foot),
                    'duration_frames': duration_frames,
                    'duration_ms': duration_ms,
                    'peak_value': float(peak_value),
                    'prominence': float(prominence)
                })

        return landing_windows

    def _validate_landing_periodicity(self, windows: List[Dict], fps: float) -> List[Dict]:
        """
        基于周期性验证落地检测的有效性

        核心逻辑：相邻落地间隔应相似（跑步是周期性运动）
        使用IQR方法剔除异常间隔
        """
        if len(windows) < 3:
            return windows

        # 计算相邻落地间隔
        intervals = []
        for i in range(1, len(windows)):
            interval = windows[i]['peak'] - windows[i-1]['peak']
            intervals.append(interval)

        if len(intervals) < 2:
            return windows

        intervals = np.array(intervals)

        # 使用IQR方法确定合理间隔范围
        q25 = np.percentile(intervals, 25)
        q75 = np.percentile(intervals, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        # 同时考虑物理约束：步频通常在120-220步/分
        min_interval = fps * 60 / 220  # 约0.27秒
        max_interval = fps * 60 / 120  # 约0.5秒

        lower_bound = max(lower_bound, min_interval)
        upper_bound = min(upper_bound, max_interval)

        # 修复：如果边界颠倒（IQR范围与物理约束不重叠），使用物理约束范围
        if lower_bound > upper_bound:
            lower_bound = min_interval
            upper_bound = max_interval

        # 标记有效的落地
        valid_windows = [windows[0]]  # 第一个默认保留
        for i in range(1, len(windows)):
            interval = windows[i]['peak'] - windows[i-1]['peak']
            if lower_bound <= interval <= upper_bound:
                valid_windows.append(windows[i])

        return valid_windows

    def _calculate_landing_knee_angles(self, keypoints_sequence: List[Dict],
                                        fps: float,
                                        knee_left: List[float],
                                        knee_right: List[float]) -> Dict:
        """
        计算落地时刻的膝关节角度（生物力学约束版）

        核心原则：
        1. 宁可少输出，也不输出错误的落地膝角
        2. 使用生物力学硬约束过滤不可信的检测
        3. 每个落地都给出通过/拒绝原因

        生物力学约束（基于马拉松运动员研究数据）：
        - 落地膝角范围：140° ≤ angle ≤ 170°（中长跑标准）
        - 理想范围：150-165°（约15-30°屈曲，符合精英跑者特征）
        - 伸展趋势：落地前膝关节应处于伸展状态（d(knee)/dt ≥ 0）

        参考文献：
        - SimpliFaster: 初始接触时约20°屈曲（~160°膝角）
        - Sports Medicine 2024: 较大膝屈曲与较高能耗相关
        """
        # 生物力学参数（硬约束）- 基于研究数据调整
        MIN_LANDING_ANGLE = 140.0  # 最小落地膝角（允许较大屈曲以缓冲）
        MAX_LANDING_ANGLE = 175.0  # 最大落地膝角（允许接近伸直的状态）
        PRE_LANDING_WINDOW_MS = 100.0  # 峰值前的搜索窗口（毫秒）
        POST_LANDING_WINDOW_MS = 30.0  # 峰值后的搜索窗口（毫秒）- 膝关节最大伸展可能在触地后

        # 检测落地区间
        landing_windows = self._detect_landing_windows(keypoints_sequence, fps)

        if not landing_windows:
            return {
                'landing_angle_mean': 0,
                'landing_angle_std': 0,
                'landing_count': 0,
                'valid_count': 0,
                'rejected_count': 0,
                'landing_angles': [],
                'per_step_stats': [],
                'rejected_steps': [],
                'method': 'biomechanical_constrained_v3'
            }

        # 计算膝角变化率（用于伸展趋势判断）
        # 注意：使用有符号的梯度，正值=伸展，负值=屈曲
        knee_left_arr = np.array(knee_left)
        knee_right_arr = np.array(knee_right)

        # 检查数组是否足够长以计算梯度（至少需要2个元素）
        if len(knee_left_arr) < 2 or len(knee_right_arr) < 2:
            return {
                'landing_angle_mean': 0,
                'landing_angle_std': 0,
                'landing_count': len(landing_windows),
                'valid_count': 0,
                'rejected_count': len(landing_windows),
                'landing_angles': [],
                'per_step_stats': [],
                'rejected_steps': [{'reason': 'Insufficient knee angle data'}],
                'method': 'biomechanical_constrained_v3'
            }

        knee_left_rate = np.gradient(knee_left_arr)  # 有符号
        knee_right_rate = np.gradient(knee_right_arr)

        # 计算峰值前后搜索窗口的帧数
        pre_landing_frames = int(PRE_LANDING_WINDOW_MS * fps / 1000.0)
        pre_landing_frames = max(2, min(pre_landing_frames, 6))  # 限制在2-6帧
        post_landing_frames = max(1, int(POST_LANDING_WINDOW_MS * fps / 1000.0))  # 至少1帧

        # 对每个落地候选应用生物力学约束
        valid_landings = []
        rejected_landings = []

        for window in landing_windows:
            result = self._validate_landing_biomechanics(
                knee_left_arr, knee_right_arr,
                knee_left_rate, knee_right_rate,
                window['peak'], window['foot'],
                pre_landing_frames, post_landing_frames,
                MIN_LANDING_ANGLE, MAX_LANDING_ANGLE
            )

            step_info = {
                'peak_frame': int(window['peak']),
                'foot': str(window['foot']),
                'duration_ms': float(window['duration_ms'])
            }

            if result['valid']:
                step_info.update({
                    'landing_angle': float(result['landing_angle']),
                    'extension_rate': float(result['extension_rate']),
                    'confidence': result['confidence']
                })
                valid_landings.append(step_info)
            else:
                step_info.update({
                    'rejection_reason': result['rejection_reason'],
                    'actual_angle': float(result.get('actual_angle', 0)),
                    'actual_rate': float(result.get('actual_rate', 0))
                })
                rejected_landings.append(step_info)

        # 整体汇总统计
        landing_angles = [s['landing_angle'] for s in valid_landings]

        print(f"  落地膝角分析（生物力学约束版）：")
        print(f"    候选落地: {len(landing_windows)} 次")
        print(f"    通过约束: {len(valid_landings)} 次")
        print(f"    被拒绝: {len(rejected_landings)} 次")

        if rejected_landings:
            reasons = {}
            for r in rejected_landings:
                reason = r['rejection_reason']
                reasons[reason] = reasons.get(reason, 0) + 1
            print(f"    拒绝原因: {reasons}")

        if landing_angles:
            # 提取置信度（从字符串格式"75%"转换为浮点数0.75）
            confidences = []
            for step in valid_landings:
                conf_str = step.get('confidence', '50%')
                try:
                    conf_val = float(conf_str.replace('%', '')) / 100.0
                except:
                    conf_val = 0.5
                confidences.append(conf_val)

            confidences = np.array(confidences)
            landing_angles_arr = np.array(landing_angles)

            # 计算置信度加权平均（如果有多个有效样本）
            if len(landing_angles) > 1 and np.sum(confidences) > 0:
                # 归一化权重
                weights = confidences / np.sum(confidences)
                weighted_mean = float(np.sum(landing_angles_arr * weights))
                # 加权标准差
                weighted_var = float(np.sum(weights * (landing_angles_arr - weighted_mean) ** 2))
                weighted_std = float(np.sqrt(weighted_var))
            else:
                weighted_mean = float(np.mean(landing_angles))
                weighted_std = 0.0

            # 同时保留简单平均值用于对比
            simple_mean = float(np.mean(landing_angles))
            simple_std = float(np.std(landing_angles)) if len(landing_angles) > 1 else 0.0

            print(f"    有效角度: {[f'{a:.1f}°' for a in landing_angles]}")
            print(f"    置信度: {[f'{c:.0%}' for c in confidences]}")
            print(f"    简单平均: {simple_mean:.1f}°, 加权平均: {weighted_mean:.1f}°")

            return {
                'landing_angle_mean': weighted_mean,  # 使用加权平均
                'landing_angle_std': weighted_std,
                'landing_angle_simple_mean': simple_mean,  # 保留简单平均用于对比
                'landing_angle_simple_std': simple_std,
                'landing_count': int(len(landing_angles)),
                'valid_count': int(len(valid_landings)),
                'rejected_count': int(len(rejected_landings)),
                'landing_angles': [float(a) for a in landing_angles],
                'per_step_stats': valid_landings,
                'rejected_steps': rejected_landings,
                'method': 'biomechanical_constrained_weighted'
            }

        return {
            'landing_angle_mean': 0,
            'landing_angle_std': 0,
            'landing_count': 0,
            'valid_count': 0,
            'rejected_count': int(len(rejected_landings)),
            'landing_angles': [],
            'per_step_stats': [],
            'rejected_steps': rejected_landings,
            'method': 'biomechanical_constrained_v3'
        }

    def _validate_landing_biomechanics(self, knee_left: np.ndarray, knee_right: np.ndarray,
                                        knee_left_rate: np.ndarray, knee_right_rate: np.ndarray,
                                        peak: int, foot: str, pre_window_frames: int,
                                        post_window_frames: int,
                                        min_angle: float, max_angle: float) -> Dict:
        """
        使用生物力学约束验证单次落地

        约束条件：
        1. 膝角范围：min_angle ≤ angle ≤ max_angle
        2. 伸展趋势：落地前膝关节应在伸展（rate ≥ 0）

        搜索策略：
        - 在peak前后的时间窗内搜索（包含触地瞬间和触地后短暂时间）
        - 只保留同时满足角度范围+伸展趋势的帧
        - 在合法帧中选取最大角度
        """
        n = len(knee_left)

        # 确定搜索范围：peak前的pre_window_frames帧 + peak本身 + peak后的post_window_frames帧
        search_start = max(0, peak - pre_window_frames)
        search_end = min(n, peak + post_window_frames + 1)  # +1确保包含peak本身

        if search_start >= search_end:
            return {'valid': False, 'rejection_reason': 'window_too_small'}

        # 选择对应脚的膝角数据
        if foot == 'left':
            angles = knee_left[search_start:search_end]
            rates = knee_left_rate[search_start:search_end]
        elif foot == 'right':
            angles = knee_right[search_start:search_end]
            rates = knee_right_rate[search_start:search_end]
        else:
            angles = (knee_left[search_start:search_end] + knee_right[search_start:search_end]) / 2
            rates = (knee_left_rate[search_start:search_end] + knee_right_rate[search_start:search_end]) / 2

        # 过滤NaN
        valid_mask = ~np.isnan(angles)
        if not np.any(valid_mask):
            return {'valid': False, 'rejection_reason': 'no_valid_frames'}

        angles = angles[valid_mask]
        rates = rates[valid_mask]

        # 约束1：膝角范围 [min_angle, max_angle]
        angle_valid = (angles >= min_angle) & (angles <= max_angle)

        # 约束2：伸展趋势（rate ≥ -1，允许轻微波动）
        # 使用-1而非0是为了容忍测量噪声
        extension_valid = rates >= -1.0

        # 同时满足两个约束
        both_valid = angle_valid & extension_valid

        # 统计分析
        max_angle_in_window = float(np.max(angles))
        mean_rate_in_window = float(np.mean(rates))

        if not np.any(both_valid):
            # 判断拒绝原因
            if not np.any(angle_valid):
                if max_angle_in_window < min_angle:
                    return {
                        'valid': False,
                        'rejection_reason': 'angle_too_low',
                        'actual_angle': max_angle_in_window,
                        'actual_rate': mean_rate_in_window
                    }
                else:
                    return {
                        'valid': False,
                        'rejection_reason': 'angle_too_high',
                        'actual_angle': max_angle_in_window,
                        'actual_rate': mean_rate_in_window
                    }
            else:
                return {
                    'valid': False,
                    'rejection_reason': 'flexion_trend',
                    'actual_angle': max_angle_in_window,
                    'actual_rate': mean_rate_in_window
                }

        # 在合法帧中选取最大角度
        valid_angles = angles[both_valid]
        valid_rates = rates[both_valid]

        best_angle = float(np.max(valid_angles))
        best_idx = np.argmax(valid_angles)
        best_rate = float(valid_rates[best_idx])

        # 计算置信度（基于合法帧占比和角度稳定性）
        valid_ratio = np.sum(both_valid) / len(angles)
        angle_stability = 1.0 - min(np.std(valid_angles) / 10.0, 1.0) if len(valid_angles) > 1 else 0.5
        confidence = (valid_ratio * 0.5 + angle_stability * 0.5)

        return {
            'valid': True,
            'landing_angle': best_angle,
            'extension_rate': best_rate,
            'confidence': f"{confidence:.0%}",
            'valid_frame_count': int(np.sum(both_valid)),
            'total_frame_count': int(len(angles))
        }

    def _calculate_vertical_motion_normalized(self, keypoints_sequence: List[Dict],
                                               fps: float, trunk_length_2d: float) -> Dict:
        """
        归一化垂直运动分析（改进版）

        核心改进：
        1. 先做去趋势，抑制镜头抖动和整体漂移
        2. 基于周期分段计算振幅，而非整段全局峰谷配对
        3. 侧面2D归一化改用肩髋欧氏距离，降低前倾导致的高估
        4. 侧面垂直位移默认使用2D信号，避免root-relative 3D压低振幅
        5. 保持原有返回字段，兼容前后端
        """
        # 提取身体中心Y坐标（以髋为主，肩部辅助）
        body_y_positions = []
        data_source = '2D'
        trunk_length = trunk_length_2d  # 默认使用2D躯干长度
        trunk_length_2d_vertical = self._calculate_vertical_reference_length_2d(keypoints_sequence)

        for kp in keypoints_sequence:
            left_hip = kp['landmarks'][23]
            right_hip = kp['landmarks'][24]
            left_shoulder = kp['landmarks'][11]
            right_shoulder = kp['landmarks'][12]

            hip_center = None
            shoulder_center = None

            if left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
                hip_center = (left_hip['y_norm'] + right_hip['y_norm']) / 2
            if left_shoulder['visibility'] > 0.5 and right_shoulder['visibility'] > 0.5:
                shoulder_center = (left_shoulder['y_norm'] + right_shoulder['y_norm']) / 2

            if hip_center is None and shoulder_center is None:
                continue

            if hip_center is not None and shoulder_center is not None:
                center_y = hip_center * 0.9 + shoulder_center * 0.1
            else:
                center_y = hip_center if hip_center is not None else shoulder_center

            body_y_positions.append(center_y)

        data_source = '2D'
        if trunk_length_2d_vertical > 0 and trunk_length_2d > 0:
            trunk_length = trunk_length_2d_vertical * 0.65 + trunk_length_2d * 0.35
        else:
            trunk_length = trunk_length_2d_vertical or trunk_length_2d

        if len(body_y_positions) < 10:
            return self._get_empty_vertical_motion()

        body_y_positions = np.array(body_y_positions, dtype=float)
        vertical_oscillation = self._extract_vertical_oscillation_component(body_y_positions.tolist(), fps)
        if len(vertical_oscillation) >= 7:
            window = min(7, len(vertical_oscillation))
            if window % 2 == 0:
                window -= 1
            vertical_oscillation = savgol_filter(vertical_oscillation, window_length=window, polyorder=2)

        estimated_period = self._estimate_oscillation_period_frames(vertical_oscillation, fps)
        peaks, troughs, cycle_amplitudes = self._calculate_cycle_amplitudes(
            vertical_oscillation, fps, estimated_period
        )

        if cycle_amplitudes:
            raw_amplitude = self._robust_cycle_amplitude(cycle_amplitudes)
        else:
            q90, q10 = np.percentile(vertical_oscillation, [90, 10])
            raw_amplitude = float(q90 - q10)

        # ⭐ 核心：归一化为躯干长度的百分比
        # 现在3D数据使用3D躯干长度，2D数据使用2D躯干长度，单位一致
        if trunk_length > 0:
            normalized_amplitude = (raw_amplitude / trunk_length) * 100
        else:
            normalized_amplitude = 0

        # 合理性检查：跑步垂直振幅通常在2-20%躯干长度范围内
        # 如果超出合理范围，可能是计算问题
        if normalized_amplitude > 50:
            print(f"  ⚠️ 垂直振幅异常大 ({normalized_amplitude:.1f}%)，可能是检测问题")
            # 尝试使用更保守的估计
            normalized_amplitude = min(normalized_amplitude, 20)  # 限制最大值
        elif normalized_amplitude < 0.5 and data_source == '3D':
            print(f"  ⚠️ 垂直振幅异常小 ({normalized_amplitude:.1f}%)，可能是坐标尺度问题")

        # 计算频率（Hz）
        if estimated_period and estimated_period > 0 and fps > 0:
            frequency = fps / estimated_period
        elif len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            frequency = fps / float(np.median(peak_intervals))
        else:
            frequency = 0.0

        return {
            'amplitude': float(raw_amplitude),  # 原始振幅（与躯干同单位）
            'amplitude_normalized': float(normalized_amplitude),  # ⭐ 相对躯干长度的百分比
            'frequency': float(frequency),
            'positions': vertical_oscillation.tolist(),
            'mean_position': float(np.mean(vertical_oscillation)),
            'std_position': float(np.std(vertical_oscillation)),
            'peak_count': len(peaks),
            'trough_count': len(troughs),
            # 评估等级
            'amplitude_rating': self._rate_vertical_amplitude(normalized_amplitude),
            # 数据来源（2D或3D）
            'data_source': data_source,
            'trunk_length_used': float(trunk_length),  # 记录使用的躯干长度
            # 调试信息
            'debug_info': {
                'raw_amplitude': float(raw_amplitude),
                'estimated_period_frames': int(estimated_period) if estimated_period else 0,
                'cycle_count': len(cycle_amplitudes),
                'trunk_length_2d_input': float(trunk_length_2d),
                'trunk_length_2d_vertical': float(trunk_length_2d_vertical),
                'trunk_length_3d_samples': 0,
                'vertical_motion_mode': '2d_preferred',
            }
        }

    def _rate_vertical_amplitude(self, normalized_amplitude: float) -> Dict:
        """
        评估垂直振幅等级（4档稳健标准，参考高驰换算）
        基于躯干百分比的专业标准：
        - 优秀：≤11%
        - 良好：11-17%
        - 一般：17-25%
        - 待改进：>25%
        """
        excellent_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['excellent_max'])
        good_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['good_max'])
        fair_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['fair_max'])

        if normalized_amplitude <= excellent_max:
            return {'level': 'excellent', 'score': 95, 'description': '垂直振幅控制优秀，能量利用效率高'}
        elif normalized_amplitude <= good_max:
            return {'level': 'good', 'score': 80, 'description': '垂直振幅良好，可继续优化'}
        elif normalized_amplitude <= fair_max:
            return {'level': 'fair', 'score': 65, 'description': '垂直振幅偏大，能量损耗较多'}
        else:
            return {'level': 'poor', 'score': 45, 'description': '垂直振幅过大，建议改善跑姿'}

    def _get_empty_phase_stats(self) -> Dict:
        """返回空的相位角度统计结果"""
        empty_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        return {
            'ground_contact': empty_stats.copy(),
            'flight': empty_stats.copy(),
            'transition': empty_stats.copy(),
            'max_flexion': 0,
            'max_extension': 0,
            'range_of_motion': 0,
        }

    def _get_empty_vertical_motion(self) -> Dict:
        """返回空的垂直运动结果"""
        return {
            'amplitude': 0.0,
            'amplitude_normalized': 0.0,
            'frequency': 0.0,
            'positions': [],
            'mean_position': 0.0,
            'std_position': 0.0,
            'peak_count': 0,
            'trough_count': 0,
            'amplitude_rating': {'level': 'unknown', 'score': 0, 'description': '数据不足'},
            'data_source': 'none',
        }

    def _analyze_gait_cycle(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """
        分析完整步态周期（优化版：更精确的触地时间检测）

        帧率自适应：根据fps调整过滤阈值

        重要说明：
        - fps是处理后的帧率（通常是30fps），不是原始视频帧率
        - frame_duration_ms = 1000 / fps，即每帧对应的毫秒数
        - 例如：60fps原始视频 -> 30fps处理 -> 每帧约33.3ms
        """
        # 使用改进的触地检测算法
        ground_contacts, flight_phases = self._detect_ground_contacts_improved(keypoints_sequence, fps)

        frame_duration_ms = 1000.0 / fps

        # 调试输出
        print(f"  步态分析: fps={fps:.1f}, 每帧={frame_duration_ms:.1f}ms")
        print(f"  检测到触地: {len(ground_contacts)}次, 腾空: {len(flight_phases)}次")

        # 帧率自适应的过滤范围
        # 精英跑者触地时间：160-220ms
        # 普通跑者触地时间：220-300ms
        # 较差跑者触地时间：280-400ms
        min_gc_ms = 120  # 最小触地时间（考虑测量误差）
        max_gc_ms = 450  # 最大触地时间

        # 腾空时间范围
        min_flight_ms = 40  # 最小腾空时间
        max_flight_ms = 350  # 最大腾空时间

        # 计算触地时间（毫秒）
        ground_contact_durations_ms = []
        for gc in ground_contacts:
            duration_ms = gc['duration_frames'] * frame_duration_ms
            if min_gc_ms <= duration_ms <= max_gc_ms:
                ground_contact_durations_ms.append(duration_ms)

        # 计算腾空时间（毫秒）
        flight_durations_ms = []
        for fl in flight_phases:
            duration_ms = fl['duration_frames'] * frame_duration_ms
            if min_flight_ms <= duration_ms <= max_flight_ms:
                flight_durations_ms.append(duration_ms)

        # 使用稳健统计（去除异常值后的平均）
        if len(ground_contact_durations_ms) >= 3:
            # 去除最高和最低值后取平均
            sorted_gc = sorted(ground_contact_durations_ms)
            trimmed_gc = sorted_gc[1:-1] if len(sorted_gc) > 2 else sorted_gc
            avg_ground_contact_ms = float(np.mean(trimmed_gc))
        elif ground_contact_durations_ms:
            avg_ground_contact_ms = float(np.median(ground_contact_durations_ms))
        else:
            avg_ground_contact_ms = 0

        if len(flight_durations_ms) >= 3:
            sorted_fl = sorted(flight_durations_ms)
            trimmed_fl = sorted_fl[1:-1] if len(sorted_fl) > 2 else sorted_fl
            avg_flight_ms = float(np.mean(trimmed_fl))
        elif flight_durations_ms:
            avg_flight_ms = float(np.median(flight_durations_ms))
        else:
            avg_flight_ms = 0

        # 计算步态周期时间
        if avg_ground_contact_ms > 0 and avg_flight_ms > 0:
            avg_cycle_duration_ms = avg_ground_contact_ms + avg_flight_ms
        else:
            avg_cycle_duration_ms = 0

        # 计算比例
        total_time = avg_ground_contact_ms + avg_flight_ms
        if total_time > 0:
            ground_contact_ratio = avg_ground_contact_ms / total_time
            flight_ratio = avg_flight_ms / total_time
        else:
            # 使用默认比例（跑步典型值）
            ground_contact_ratio = 0.45
            flight_ratio = 0.35

        # 过渡期比例（从相位检测中计算）
        phases = self._detect_gait_phases(keypoints_sequence, fps)
        if phases:
            transition_count = sum(1 for p in phases if p == self.PHASE_TRANSITION)
            transition_ratio = transition_count / len(phases)
            # 重新归一化
            total = ground_contact_ratio + flight_ratio + transition_ratio
            if total > 0:
                ground_contact_ratio /= total
                flight_ratio /= total
                transition_ratio /= total
        else:
            transition_ratio = 0.20

        return {
            'phase_distribution': {
                'ground_contact': float(round(ground_contact_ratio, 3)),
                'flight': float(round(flight_ratio, 3)),
                'transition': float(round(transition_ratio, 3)),
            },
            # 各阶段时间（毫秒）
            'phase_duration_ms': {
                'ground_contact': float(round(avg_ground_contact_ms, 1)),
                'flight': float(round(avg_flight_ms, 1)),
                'transition': 0.0,
            },
            'avg_cycle_duration': float(avg_cycle_duration_ms / 1000) if avg_cycle_duration_ms > 0 else 0,
            'avg_cycle_duration_ms': float(round(avg_cycle_duration_ms, 1)),
            'cycle_count': len(ground_contact_durations_ms),
            'ground_contact_times': ground_contact_durations_ms,  # 详细数据
            'flight_times': flight_durations_ms,
            # 评估
            'gait_rating': self._rate_gait_timing(avg_ground_contact_ms),
        }

    def _detect_ground_contacts_improved(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[List[Dict], List[Dict]]:
        """
        改进的触地检测算法（重写版）

        核心改进：
        1. 使用自适应阈值（基于数据分布）
        2. 基于速度变化检测触地/离地时刻
        3. 分别检测左右脚，支持精英跑者的快速步频
        4. 帧率自适应

        预期触地时间：160-250ms（精英到普通跑者）
        """
        n_frames = len(keypoints_sequence)

        # 提取左右脚踝Y坐标
        left_ankle_y = []
        right_ankle_y = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][27]
            right = kp['landmarks'][28]

            left_y = left['y_norm'] if left['visibility'] > 0.5 else np.nan
            right_y = right['y_norm'] if right['visibility'] > 0.5 else np.nan

            left_ankle_y.append(left_y)
            right_ankle_y.append(right_y)

        left_ankle_y = self._interpolate_nans(np.array(left_ankle_y))
        right_ankle_y = self._interpolate_nans(np.array(right_ankle_y))

        # 平滑处理（轻度平滑，保留细节）
        if len(left_ankle_y) > 5:
            # 使用较小的平滑窗口
            window = min(5, len(left_ankle_y) // 2)
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                left_ankle_y = savgol_filter(left_ankle_y, window, 2)
                right_ankle_y = savgol_filter(right_ankle_y, window, 2)

        # 帧率自适应参数
        frame_duration_ms = 1000.0 / fps
        min_gc_frames = max(2, int(160 / frame_duration_ms))  # 最小触地帧数（160ms）
        max_gc_frames = int(350 / frame_duration_ms)  # 最大触地帧数（350ms）
        min_peak_distance = max(3, int(fps * 0.15))  # 最小150ms间隔

        # 自适应prominence计算
        left_range = np.max(left_ankle_y) - np.min(left_ankle_y)
        right_range = np.max(right_ankle_y) - np.min(right_ankle_y)
        prominence = min(left_range, right_range) * 0.15  # 范围的15%作为prominence

        # 检测左脚触地峰值
        left_peaks, _ = find_peaks(left_ankle_y, distance=min_peak_distance, prominence=prominence)
        # 检测右脚触地峰值
        right_peaks, _ = find_peaks(right_ankle_y, distance=min_peak_distance, prominence=prominence)

        ground_contacts = []
        flight_phases = []

        # 处理左脚触地
        for peak in left_peaks:
            gc = self._extract_single_ground_contact(
                left_ankle_y, peak, fps, min_gc_frames, max_gc_frames, 'left'
            )
            if gc:
                ground_contacts.append(gc)

        # 处理右脚触地
        for peak in right_peaks:
            gc = self._extract_single_ground_contact(
                right_ankle_y, peak, fps, min_gc_frames, max_gc_frames, 'right'
            )
            if gc:
                ground_contacts.append(gc)

        # 按时间排序
        ground_contacts.sort(key=lambda x: x['start_frame'])

        # 计算腾空时间（相邻触地之间）
        for i in range(len(ground_contacts) - 1):
            current_end = ground_contacts[i]['end_frame']
            next_start = ground_contacts[i + 1]['start_frame']
            flight_duration = next_start - current_end

            if flight_duration >= 2:  # 至少2帧
                flight_phases.append({
                    'start_frame': current_end,
                    'end_frame': next_start,
                    'duration_frames': flight_duration
                })

        return ground_contacts, flight_phases

    def _extract_single_ground_contact(self, ankle_y: np.ndarray, peak_frame: int,
                                        fps: float, min_frames: int, max_frames: int,
                                        foot: str) -> Optional[Dict]:
        """
        提取单次触地的起止帧（改进版）

        核心改进：
        1. 使用相对阈值检测触地边界（相对于峰值的百分比）
        2. 结合速度信息辅助判断
        3. 添加帧率自适应的最小/最大持续时间限制

        注意：在归一化图像坐标系中，Y值越大表示位置越低（接近地面）
        所以ankle_y的峰值（局部最大值）对应脚踝在最低位置（触地时刻）
        """
        n = len(ankle_y)
        if peak_frame < 0 or peak_frame >= n:
            return None

        peak_y = ankle_y[peak_frame]
        min_y = np.min(ankle_y)
        amplitude = peak_y - min_y

        if amplitude < 0.005:  # 振幅太小，无法可靠检测
            return None

        # 触地阈值：当脚踝Y值高于 (峰值 - 振幅*阈值比例) 时认为在触地
        # 使用70%阈值：即Y值在峰值附近30%范围内视为触地
        threshold_ratio = 0.30
        touch_threshold = peak_y - amplitude * threshold_ratio

        # 向前找触地开始
        start_frame = peak_frame
        search_start = max(0, peak_frame - max_frames)
        for j in range(peak_frame - 1, search_start, -1):
            if ankle_y[j] < touch_threshold:
                start_frame = j + 1
                break
        else:
            # 如果没找到，使用搜索起点
            start_frame = search_start

        # 向后找触地结束
        end_frame = peak_frame
        search_end = min(n, peak_frame + max_frames)
        for j in range(peak_frame + 1, search_end):
            if ankle_y[j] < touch_threshold:
                end_frame = j
                break
        else:
            # 如果没找到，使用搜索终点
            end_frame = min(search_end, n - 1)

        duration_frames = end_frame - start_frame

        # 确保持续时间为正
        if duration_frames <= 0:
            duration_frames = 1

        # 验证触地时长合理性
        if min_frames <= duration_frames <= max_frames:
            return {
                'start_frame': start_frame,
                'end_frame': end_frame,
                'peak_frame': peak_frame,
                'duration_frames': duration_frames,
                'foot': foot
            }

        return None

    def _rate_gait_timing(self, ground_contact_ms: float) -> Dict:
        """
        评估触地时间（新标准）
        <210ms精英，210-240ms优秀，240-270ms良好，270-300ms一般，>300ms较差
        """
        if ground_contact_ms <= 0:
            return {'level': 'unknown', 'score': 0, 'description': '数据不足'}
        elif ground_contact_ms < 210:
            return {'level': 'elite', 'score': 100, 'description': '精英水平'}
        elif ground_contact_ms < 240:
            return {'level': 'excellent', 'score': 90, 'description': '优秀'}
        elif ground_contact_ms < 270:
            return {'level': 'good', 'score': 75, 'description': '良好'}
        elif ground_contact_ms < 300:
            return {'level': 'fair', 'score': 60, 'description': '一般'}
        else:
            return {'level': 'poor', 'score': 45, 'description': '较差'}

    def _rate_gait_distribution(self, ground_ratio: float, flight_ratio: float) -> Dict:
        """
        评估步态分布
        理想的跑步步态：触地期约40-50%，腾空期约30-40%
        """
        if 0.35 <= ground_ratio <= 0.55 and 0.25 <= flight_ratio <= 0.45:
            return {'level': 'excellent', 'score': 100, 'description': '步态节奏优秀'}
        elif 0.30 <= ground_ratio <= 0.60 and 0.20 <= flight_ratio <= 0.50:
            return {'level': 'good', 'score': 80, 'description': '步态节奏良好'}
        elif ground_ratio > 0.60:
            return {'level': 'heavy', 'score': 60, 'description': '触地时间偏长，可能步态较重'}
        elif flight_ratio < 0.15:
            return {'level': 'shuffling', 'score': 50, 'description': '腾空不足，可能为拖步跑'}
        else:
            return {'level': 'fair', 'score': 70, 'description': '步态节奏一般'}

    # ==================== 正面/后方视角分析 ====================

    def _analyze_frontal_view(self, valid_frames: List[Dict], fps: float,
                               trunk_length: float) -> Dict:
        """
        正面/后方视角分析（完全重构版 - 3D增强）

        正面视角的特点：
        1. 跑者朝向相机跑来（或远离），深度变化大
        2. 脚踝Y坐标变化不明显（不能用于检测触地）
        3. 优势指标：下肢力线、横向稳定性、肩部晃动

        分析策略：
        1. 核心指标：下肢力线、横向稳定性（正面视角优势）
        2. 辅助指标：使用髋部/肩部Y坐标估算垂直振幅
        3. 步频检测：多方法融合（3D数据 + 2D模式）
        4. 不可靠指标：触地时间（正面视角无法准确检测，不输出）
        """
        # 检测3D数据可用性
        use_3d = self.has_3d_data and self.keypoints_3d is not None
        if use_3d:
            print("✅ 正面视角分析：检测到3D数据，将用于提高准确性")
        else:
            print("ℹ️ 正面视角分析：未检测到3D数据，使用2D分析")

        # 1. 正面视角核心指标（高可信度）
        lower_limb = self._calculate_lower_limb_alignment(valid_frames)
        lateral_stability = self._calculate_lateral_stability(valid_frames, fps)
        stability = self._calculate_stability_front_view(valid_frames)

        # 1.5 【修复】调用步态对称性计算（此函数已存在但之前未被调用）
        gait_symmetry = self._calculate_gait_symmetry_detailed(valid_frames)

        # 将对称性数据注入到现有结构中（修复数据流断裂）
        lateral_stability['symmetry'] = gait_symmetry['overall_score']
        lateral_stability['symmetry_details'] = gait_symmetry
        stability['symmetry'] = gait_symmetry['overall_score']

        # 2. 正面视角改进的垂直振幅（使用髋部Y，非脚踝Y）
        vertical_motion = self._calculate_vertical_motion_frontal(valid_frames, fps, trunk_length)

        # 3. 正面视角步频检测（多方法融合，更准确）
        cadence = self._calculate_cadence_frontal_improved(valid_frames, fps)

        # 4. 正面视角步态周期（仅输出可靠数据）
        gait_cycle = self._analyze_gait_cycle_frontal(valid_frames, fps)

        # 5. 肩部倾斜分析（正面视角特有）
        shoulder_analysis = self._calculate_shoulder_tilt(valid_frames)

        # 6. 如果有3D数据，添加额外的3D分析结果
        angles_3d = None
        if use_3d:
            angles_3d = self._calculate_angles_frontal_3d(valid_frames, fps)

        results = {
            # 正面视角核心指标（高可信度）
            'lower_limb_alignment': lower_limb,
            'lateral_stability': lateral_stability,
            'stability': stability,
            'shoulder_analysis': shoulder_analysis,

            # 【新增】独立的对称性分析字段（方便evaluator直接读取）
            'symmetry_analysis': {
                'overall_score': gait_symmetry['overall_score'],
                'gait_symmetry': gait_symmetry,
                'rating': gait_symmetry['rating'],
                'data_source': 'gait_symmetry_detailed'
            },

            # 正面视角适配的通用指标
            'vertical_motion': vertical_motion,
            'cadence': cadence,
            'gait_cycle': gait_cycle,

            # 3D分析结果（如果可用）
            'angles': angles_3d if angles_3d else self._get_empty_angles_frontal(),

            # 标记视角类型和可信度
            'view_type': 'frontal',
            'has_3d_data': use_3d,
            'reliability_note': '正面视角：下肢力线和稳定性数据可靠' +
                               ('，已使用3D数据增强' if use_3d else '，步频为估算值') +
                               '，触地时间不可用',
        }
        return results

    def _calculate_angles_frontal_3d(self, valid_frames: List[Dict], fps: float) -> Dict:
        """
        正面视角的3D角度分析

        使用3D数据计算关节角度，提供更准确的数据
        """
        if not self.has_3d_data or self.keypoints_3d is None:
            return self._get_empty_angles_frontal()

        knee_angles_left = []
        knee_angles_right = []

        for i, kp in enumerate(valid_frames):
            if i >= len(self.keypoints_3d):
                break

            kp_3d = self.keypoints_3d[i]
            if not kp_3d.get('has_3d', False):
                knee_angles_left.append(np.nan)
                knee_angles_right.append(np.nan)
                continue

            pose_3d = kp_3d.get('pose_3d', {})

            # 3D膝关节角度
            knee_angles_left.append(self._calculate_joint_angle_3d(
                pose_3d, 'l_hip', 'l_knee', 'l_ankle'
            ))
            knee_angles_right.append(self._calculate_joint_angle_3d(
                pose_3d, 'r_hip', 'r_knee', 'r_ankle'
            ))

        # 平滑处理
        knee_left_smooth = self._smooth_and_filter_angles(knee_angles_left)
        knee_right_smooth = self._smooth_and_filter_angles(knee_angles_right)

        has_data = len(knee_left_smooth) > 0 and not np.all(np.isnan(knee_left_smooth))

        if has_data:
            return {
                'knee_left': knee_left_smooth,
                'knee_right': knee_right_smooth,
                'knee_left_mean': float(np.nanmean(knee_left_smooth)),
                'knee_right_mean': float(np.nanmean(knee_right_smooth)),
                'knee_left_std': float(np.nanstd(knee_left_smooth)),
                'knee_right_std': float(np.nanstd(knee_right_smooth)),
                'data_reliability': {
                    'is_3d': True,
                    'has_knee_data': True,
                    'reliability': 'high',
                    'description': '使用MotionBERT 3D姿态计算，数据可靠'
                }
            }
        return self._get_empty_angles_frontal()

    def _get_empty_angles_frontal(self) -> Dict:
        """返回空的正面视角角度结果"""
        return {
            'knee_left': [],
            'knee_right': [],
            'knee_left_mean': 0,
            'knee_right_mean': 0,
            'knee_left_std': 0,
            'knee_right_std': 0,
            'data_reliability': {
                'is_3d': False,
                'has_knee_data': False,
                'reliability': 'low',
                'description': '正面视角未使用3D数据'
            }
        }

    def _calculate_vertical_motion_frontal(self, keypoints_sequence: List[Dict],
                                            fps: float, trunk_length: float) -> Dict:
        """
        正面视角的垂直振幅计算（使用髋部/肩部中心Y坐标）

        正面视角下，脚踝Y坐标不能反映真实的垂直运动（因为深度变化）
        改用躯干中心（髋部+肩部中点）的Y坐标变化
        """
        hip_y_positions = []
        shoulder_y_positions = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # 髋部中心Y坐标
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            if left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
                hip_y = (left_hip['y_norm'] + right_hip['y_norm']) / 2
                hip_y_positions.append(hip_y)

            # 肩部中心Y坐标
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            if left_shoulder['visibility'] > 0.5 and right_shoulder['visibility'] > 0.5:
                shoulder_y = (left_shoulder['y_norm'] + right_shoulder['y_norm']) / 2
                shoulder_y_positions.append(shoulder_y)

        # 优先使用髋部Y（更稳定）
        if len(hip_y_positions) >= 10:
            positions = np.array(hip_y_positions)
            data_source = 'hip_center'
        elif len(shoulder_y_positions) >= 10:
            positions = np.array(shoulder_y_positions)
            data_source = 'shoulder_center'
        else:
            return self._get_empty_vertical_motion()

        # 【关键】去趋势处理：消除人物从远到近跑来的透视变化
        # 使用与髋部横摆相同的自去趋势方法
        positions_detrended = self._extract_sway_component(positions.tolist(), fps)

        # 平滑处理
        positions_smooth = self._smooth_signal_advanced(positions_detrended)

        # 分析振荡（添加 prominence 参数过滤噪声峰值）
        min_distance = max(3, int(fps * 0.15))
        prominence = np.std(positions_smooth) * 0.3
        peaks, _ = find_peaks(positions_smooth, distance=min_distance, prominence=prominence)
        troughs, _ = find_peaks(-positions_smooth, distance=min_distance, prominence=prominence)

        # 计算振幅
        if len(peaks) > 0 and len(troughs) > 0:
            cycle_amplitudes = []
            all_extrema = sorted(
                [(p, 'peak', positions_smooth[p]) for p in peaks] +
                [(t, 'trough', positions_smooth[t]) for t in troughs],
                key=lambda x: x[0]
            )
            for i in range(len(all_extrema) - 1):
                if all_extrema[i][1] != all_extrema[i+1][1]:
                    amp = abs(all_extrema[i][2] - all_extrema[i+1][2])
                    cycle_amplitudes.append(amp)

            if cycle_amplitudes:
                raw_amplitude = np.median(cycle_amplitudes)
            else:
                raw_amplitude = abs(np.mean(positions_smooth[peaks]) - np.mean(positions_smooth[troughs]))
        else:
            q75, q25 = np.percentile(positions_smooth, [75, 25])
            raw_amplitude = q75 - q25

        # 归一化
        normalized_amplitude = (raw_amplitude / trunk_length) * 100 if trunk_length > 0 else 0

        # 异常值检查（与侧面视角保持一致）
        if normalized_amplitude > 50:
            print(f"  Warning: 正面视角垂直振幅异常大 ({normalized_amplitude:.1f}%)，限制最大值")
            normalized_amplitude = min(normalized_amplitude, 20)

        # 正面视角的振幅可能偏小（因为深度变化），标注为估算值
        return {
            'amplitude': float(raw_amplitude),
            'amplitude_normalized': float(normalized_amplitude),
            'frequency': float(len(peaks) / (len(positions) / fps)) if len(peaks) > 1 else 0.0,
            'positions': positions_smooth.tolist(),
            'mean_position': float(np.mean(positions_smooth)),
            'std_position': float(np.std(positions_smooth)),
            'peak_count': len(peaks),
            'trough_count': len(troughs),
            'amplitude_rating': self._rate_vertical_amplitude(normalized_amplitude),
            'data_source': data_source,
            'is_estimate': True,  # 标记为估算值
            'reliability': 'medium',  # 可信度中等
        }

    def _calculate_cadence_frontal_improved(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """
        正面视角步频计算（FFT主频提取法）

        算法原理：
        1. 提取左右膝关节X坐标差值作为周期信号
        2. 带通滤波(1.5-4Hz)去除噪声，保留步频范围(90-240spm)
        3. FFT提取主频，主频即为步频的一半（每个周期包含左右两步）
        4. 自相关验证FFT结果

        优势：
        - 纯2D计算，不依赖3D数据
        - 物理意义明确：直接检测左右腿交替节律
        - 对跑者距离变化（透视缩放）不敏感（使用差值信号）
        """
        n_frames = len(keypoints_sequence)
        duration = n_frames / fps

        if duration < 1.0 or n_frames < int(fps):
            return self._get_empty_cadence()

        # ========== 1. 提取左右膝关节X坐标差值信号 ==========
        x_diff_signal = self._extract_leg_alternation_signal(keypoints_sequence)
        if x_diff_signal is None or len(x_diff_signal) < int(fps):
            return self._get_empty_cadence()

        # ========== 2. 带通滤波 ==========
        # 步频范围：90-240 spm → 1.5-4.0 Hz
        # 每个完整周期(左右各一步) → 频率为步频/2，即0.75-2.0 Hz
        # 但我们检测的是左右交替，每次交替算一步，所以频率等于步频/60
        try:
            filtered_signal = self._bandpass_filter(x_diff_signal, fps,
                                                     low_freq=1.2, high_freq=4.0)
        except Exception:
            # 滤波失败时使用原始信号
            filtered_signal = x_diff_signal - np.mean(x_diff_signal)

        # ========== 3. FFT主频提取 ==========
        cadence_fft, fft_confidence, spectrum_info = self._extract_cadence_fft(
            filtered_signal, fps
        )

        # ========== 4. 自相关验证 ==========
        cadence_autocorr, autocorr_confidence = self._extract_cadence_autocorr(
            filtered_signal, fps
        )

        # ========== 5. 结果融合 ==========
        # FFT和自相关结果加权融合
        if cadence_fft > 0 and cadence_autocorr > 0:
            # 两个方法都有结果，检查一致性
            diff_ratio = abs(cadence_fft - cadence_autocorr) / max(cadence_fft, cadence_autocorr)
            if diff_ratio < 0.15:  # 误差<15%认为一致
                # 一致，加权平均
                final_cadence = (cadence_fft * fft_confidence + cadence_autocorr * autocorr_confidence) / \
                               (fft_confidence + autocorr_confidence)
                total_confidence = min(1.0, (fft_confidence + autocorr_confidence) / 1.5)
            else:
                # 不一致，选择置信度更高的
                if fft_confidence >= autocorr_confidence:
                    final_cadence = cadence_fft
                    total_confidence = fft_confidence * 0.8
                else:
                    final_cadence = cadence_autocorr
                    total_confidence = autocorr_confidence * 0.8
        elif cadence_fft > 0:
            final_cadence = cadence_fft
            total_confidence = fft_confidence * 0.9
        elif cadence_autocorr > 0:
            final_cadence = cadence_autocorr
            total_confidence = autocorr_confidence * 0.9
        else:
            final_cadence = 0.0
            total_confidence = 0.0

        # 步数估算
        step_count = int(final_cadence * duration / 60) if final_cadence > 0 else 0

        return {
            'cadence': float(round(final_cadence, 1)),
            'step_count': step_count,
            'duration': float(duration),
            'confidence': float(total_confidence),
            'method': 'fft_autocorr',
            'method_details': {
                'fft': {'cadence': cadence_fft, 'confidence': fft_confidence},
                'autocorr': {'cadence': cadence_autocorr, 'confidence': autocorr_confidence},
                'spectrum': spectrum_info,
            },
            'rating': self._rate_cadence(final_cadence),
            'is_estimate': False,  # FFT是直接测量，不是估算
            'reliability': 'high' if total_confidence > 0.7 else ('medium' if total_confidence > 0.4 else 'low'),
        }

    def _extract_leg_alternation_signal(self, keypoints_sequence: List[Dict]) -> Optional[np.ndarray]:
        """
        提取左右腿交替信号（X坐标差值）

        使用膝关节而非脚踝，因为膝关节在正面视角下更稳定可见
        """
        left_x = []
        right_x = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']
            # 优先使用膝关节(25, 26)，其次脚踝(27, 28)
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]

            # 选择可见度更高的关节
            if left_knee['visibility'] > 0.5 and right_knee['visibility'] > 0.5:
                left_x.append(left_knee['x_norm'])
                right_x.append(right_knee['x_norm'])
            elif left_ankle['visibility'] > 0.5 and right_ankle['visibility'] > 0.5:
                left_x.append(left_ankle['x_norm'])
                right_x.append(right_ankle['x_norm'])
            else:
                left_x.append(np.nan)
                right_x.append(np.nan)

        # 插值填充缺失值
        left_x = self._interpolate_nans(np.array(left_x))
        right_x = self._interpolate_nans(np.array(right_x))

        if len(left_x) < 10:
            return None

        # 计算差值信号
        x_diff = left_x - right_x

        # 去除线性趋势（跑者接近相机时的透视变化）
        x_diff_detrend = self._detrend_signal(x_diff)

        return x_diff_detrend

    def _detrend_signal(self, signal: np.ndarray) -> np.ndarray:
        """去除信号的线性趋势"""
        n = len(signal)
        x = np.arange(n)
        # 线性回归
        coeffs = np.polyfit(x, signal, 1)
        trend = np.polyval(coeffs, x)
        return signal - trend

    def _bandpass_filter(self, signal: np.ndarray, fs: float,
                         low_freq: float, high_freq: float) -> np.ndarray:
        """
        带通滤波器

        Args:
            signal: 输入信号
            fs: 采样频率
            low_freq: 低截止频率 (Hz)
            high_freq: 高截止频率 (Hz)

        Returns:
            滤波后信号
        """
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        # 确保频率在有效范围内
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))

        # 使用4阶巴特沃斯滤波器
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)

        return filtered

    def _extract_cadence_fft(self, signal: np.ndarray, fps: float) -> Tuple[float, float, Dict]:
        """
        使用FFT提取步频

        Returns:
            (cadence_spm, confidence, spectrum_info)
        """
        n = len(signal)

        # 加窗减少频谱泄漏
        window = np.hanning(n)
        windowed_signal = signal * window

        # FFT
        yf = fft(windowed_signal)
        xf = fftfreq(n, 1/fps)

        # 只取正频率部分
        positive_mask = xf > 0
        xf_pos = xf[positive_mask]
        yf_pos = np.abs(yf[positive_mask])

        # 步频范围: 90-240 spm → 1.5-4.0 Hz
        freq_mask = (xf_pos >= 1.2) & (xf_pos <= 4.5)

        if not np.any(freq_mask):
            return 0.0, 0.0, {}

        xf_range = xf_pos[freq_mask]
        yf_range = yf_pos[freq_mask]

        if len(yf_range) == 0:
            return 0.0, 0.0, {}

        # 找到主频
        peak_idx = np.argmax(yf_range)
        peak_freq = xf_range[peak_idx]
        peak_power = yf_range[peak_idx]

        # 步频 = 频率 × 60 × 2（因为一个周期包含左右各一步）
        cadence = peak_freq * 60 * 2

        # 计算置信度（基于峰值的显著性）
        total_power = np.sum(yf_range)
        if total_power > 0:
            peak_ratio = peak_power / total_power
            # 还需要考虑峰值相对于噪声的高度
            noise_level = np.median(yf_range)
            snr = peak_power / (noise_level + 1e-10)
            confidence = min(1.0, peak_ratio * 2) * min(1.0, snr / 10)
        else:
            confidence = 0.0

        spectrum_info = {
            'peak_freq_hz': float(peak_freq),
            'peak_power': float(peak_power),
            'total_power': float(total_power),
            'snr': float(snr) if total_power > 0 else 0.0,
        }

        return float(cadence), float(confidence), spectrum_info

    def _extract_cadence_autocorr(self, signal: np.ndarray, fps: float) -> Tuple[float, float]:
        """
        使用自相关提取步频

        自相关可以找到信号的周期性，作为FFT的验证方法
        """
        n = len(signal)

        # 归一化
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

        # 计算自相关
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[n-1:]  # 只取正lag部分
        autocorr = autocorr / autocorr[0]  # 归一化

        # 步频范围: 90-240 spm
        # 对应周期: 0.5-1.33秒（每两步一个完整周期）
        # 对应lag: 0.25-0.67秒（每一步）
        min_lag = int(fps * 0.15)  # 最小0.15秒（240spm）
        max_lag = int(fps * 0.5)   # 最大0.5秒（120spm）

        if max_lag >= len(autocorr):
            max_lag = len(autocorr) - 1

        if min_lag >= max_lag:
            return 0.0, 0.0

        # 在有效范围内找峰值
        search_range = autocorr[min_lag:max_lag+1]
        peaks, properties = find_peaks(search_range, prominence=0.1)

        if len(peaks) == 0:
            return 0.0, 0.0

        # 找到最高的峰（最强的周期性）
        peak_heights = search_range[peaks]
        best_peak_idx = peaks[np.argmax(peak_heights)]
        best_lag = min_lag + best_peak_idx

        # 步频计算：每个lag对应半个步态周期
        period_seconds = best_lag / fps
        cadence = 60 / period_seconds  # 每分钟步数

        # 置信度基于自相关峰值高度
        peak_height = search_range[best_peak_idx]
        confidence = min(1.0, peak_height * 1.5)

        return float(cadence), float(confidence)

    def _cadence_from_hip_y(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[float, int, float]:
        """
        使用髋部Y坐标检测步频

        髋部中心的垂直运动在正面视角下相对可靠
        """
        hip_y_positions = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            if left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
                hip_y = (left_hip['y_norm'] + right_hip['y_norm']) / 2
                hip_y_positions.append(hip_y)
            else:
                hip_y_positions.append(np.nan)

        hip_y = self._interpolate_nans(np.array(hip_y_positions))
        if len(hip_y) < 10:
            return 0.0, 0, 0.0

        # 平滑处理
        hip_y_smooth = self._smooth_signal_advanced(hip_y)

        # 检测峰值（每步都有一个垂直运动周期）
        min_distance = max(3, int(fps * 0.2))  # 最小0.2秒间隔
        prominence = np.std(hip_y_smooth) * 0.3

        peaks, properties = find_peaks(hip_y_smooth, distance=min_distance, prominence=prominence)
        troughs, _ = find_peaks(-hip_y_smooth, distance=min_distance, prominence=prominence)

        # 每个峰谷对代表一步
        step_count = max(len(peaks), len(troughs))
        duration = len(keypoints_sequence) / fps

        if step_count >= 2 and duration > 0:
            cadence = (step_count / duration) * 60
            # 置信度基于峰值清晰度
            if len(properties.get('prominences', [])) > 0:
                avg_prominence = np.mean(properties['prominences'])
                signal_range = np.max(hip_y_smooth) - np.min(hip_y_smooth)
                confidence = min(1.0, avg_prominence / (signal_range * 0.5)) if signal_range > 0 else 0.0
            else:
                confidence = 0.5
        else:
            cadence = 0.0
            confidence = 0.0

        return cadence, step_count, confidence

    def _cadence_from_x_alternation(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[float, int, float]:
        """
        使用左右脚X坐标交替检测步频（改进版）

        改进：
        1. 使用膝关节而非脚踝（更稳定）
        2. 添加尺度归一化
        3. 更精确的过零点检测
        """
        left_knee_x = []
        right_knee_x = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']
            left = landmarks[25]  # 左膝
            right = landmarks[26]  # 右膝

            left_x = left['x_norm'] if left['visibility'] > 0.5 else np.nan
            right_x = right['x_norm'] if right['visibility'] > 0.5 else np.nan

            left_knee_x.append(left_x)
            right_knee_x.append(right_x)

        left_knee_x = self._interpolate_nans(np.array(left_knee_x))
        right_knee_x = self._interpolate_nans(np.array(right_knee_x))

        if len(left_knee_x) < 10:
            return 0.0, 0, 0.0

        # 计算左右膝的X坐标差
        x_diff = left_knee_x - right_knee_x

        # 平滑处理
        x_diff_smooth = self._smooth_signal_advanced(x_diff)

        # 归一化到标准范围
        x_range = np.max(np.abs(x_diff_smooth))
        if x_range > 0.001:
            x_diff_norm = x_diff_smooth / x_range
        else:
            return 0.0, 0, 0.0

        # 检测过零点
        zero_crossings = []
        for i in range(1, len(x_diff_norm)):
            if x_diff_norm[i-1] * x_diff_norm[i] < 0:
                # 线性插值找精确过零点
                zero_crossings.append(i - x_diff_norm[i] / (x_diff_norm[i] - x_diff_norm[i-1]))

        # 过滤过近的过零点
        min_interval = fps * 0.15  # 最小0.15秒
        filtered_crossings = [zero_crossings[0]] if zero_crossings else []
        for zc in zero_crossings[1:]:
            if zc - filtered_crossings[-1] >= min_interval:
                filtered_crossings.append(zc)

        # 每两次过零点代表一个完整步态周期
        step_count = len(filtered_crossings) // 2
        duration = len(keypoints_sequence) / fps

        if step_count > 0 and duration > 0:
            cadence = (step_count / duration) * 60 * 2  # 左右脚各算一步
            # 置信度基于X坐标变化幅度
            confidence = min(1.0, x_range * 20)  # 变化幅度5%以上较可信
        else:
            cadence = 0.0
            confidence = 0.0

        return cadence, step_count * 2, confidence

    def _cadence_from_3d_depth(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[float, int, float]:
        """
        使用3D深度数据检测步频

        利用脚踝在Z轴（深度）方向的周期性变化
        """
        if not self.has_3d_data or self.keypoints_3d is None:
            return 0.0, 0, 0.0

        left_ankle_z = []
        right_ankle_z = []

        for i, kp in enumerate(keypoints_sequence):
            if i >= len(self.keypoints_3d):
                break

            kp_3d = self.keypoints_3d[i]
            if not kp_3d.get('has_3d', False):
                left_ankle_z.append(np.nan)
                right_ankle_z.append(np.nan)
                continue

            pose_3d = kp_3d.get('pose_3d', {})
            l_ankle = get_3d_point_from_pose(pose_3d, 'l_ankle')
            r_ankle = get_3d_point_from_pose(pose_3d, 'r_ankle')

            if l_ankle is not None:
                left_ankle_z.append(l_ankle[2])  # Z坐标
            else:
                left_ankle_z.append(np.nan)

            if r_ankle is not None:
                right_ankle_z.append(r_ankle[2])
            else:
                right_ankle_z.append(np.nan)

        left_ankle_z = self._interpolate_nans(np.array(left_ankle_z))
        right_ankle_z = self._interpolate_nans(np.array(right_ankle_z))

        if len(left_ankle_z) < 10:
            return 0.0, 0, 0.0

        # 计算深度差（前后脚交替）
        z_diff = left_ankle_z - right_ankle_z
        z_diff_smooth = self._smooth_signal_advanced(z_diff)

        # 检测过零点
        zero_crossings = np.where(np.diff(np.sign(z_diff_smooth)))[0]

        step_count = len(zero_crossings) // 2
        duration = len(keypoints_sequence) / fps

        if step_count > 0 and duration > 0:
            cadence = (step_count / duration) * 60 * 2
            # 3D数据通常更可靠
            confidence = 0.8 if step_count >= 4 else 0.5
        else:
            cadence = 0.0
            confidence = 0.0

        return cadence, step_count * 2, confidence

    def _calculate_cadence_frontal(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """
        正面视角的步频计算（原版本，保留用于兼容）
        """
        # 调用改进版
        return self._calculate_cadence_frontal_improved(keypoints_sequence, fps)

    def _get_empty_cadence(self) -> Dict:
        """返回空的步频结果"""
        return {
            'cadence': 0.0,
            'step_count': 0,
            'duration': 0.0,
            'confidence': 0.0,
            'rating': {'level': 'unknown', 'score': 0, 'description': '数据不足'},
            'is_estimate': True,
            'reliability': 'low',
        }

    def _calculate_frontal_cadence_confidence(self, cadence_x: float, cadence_knee: float) -> float:
        """计算正面视角步频的置信度"""
        if cadence_x <= 0 and cadence_knee <= 0:
            return 0.0
        if cadence_x <= 0 or cadence_knee <= 0:
            return 0.3  # 只有一种方法有效

        # 两种方法一致性高则置信度高
        diff_ratio = abs(cadence_x - cadence_knee) / max(cadence_x, cadence_knee)
        if diff_ratio < 0.1:
            return 0.9
        elif diff_ratio < 0.2:
            return 0.7
        elif diff_ratio < 0.3:
            return 0.5
        else:
            return 0.3

    def _analyze_gait_cycle_frontal(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """
        正面视角的步态周期分析

        正面视角无法准确检测触地/离地时刻，因此：
        1. 不输出触地时间（不可靠）
        2. 只输出可以从X坐标估算的周期信息
        """
        # 使用X坐标交替检测步态周期
        left_ankle_x = []
        right_ankle_x = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']
            left = landmarks[27]
            right = landmarks[28]

            left_x = left['x_norm'] if left['visibility'] > 0.5 else np.nan
            right_x = right['x_norm'] if right['visibility'] > 0.5 else np.nan

            left_ankle_x.append(left_x)
            right_ankle_x.append(right_x)

        left_ankle_x = self._interpolate_nans(np.array(left_ankle_x))
        right_ankle_x = self._interpolate_nans(np.array(right_ankle_x))

        if len(left_ankle_x) < 10:
            return self._get_empty_gait_cycle_frontal()

        x_diff = left_ankle_x - right_ankle_x
        if len(x_diff) > 5:
            x_diff = self._smooth_signal_advanced(x_diff)

        # 检测过零点
        zero_crossings = np.where(np.diff(np.sign(x_diff)))[0]

        # 计算步态周期（两次过零点间隔）
        if len(zero_crossings) >= 2:
            intervals = np.diff(zero_crossings)
            # 每两次过零是一个完整周期
            cycle_frames = intervals[::2] if len(intervals) > 1 else intervals
            avg_cycle_frames = np.median(cycle_frames) * 2 if len(cycle_frames) > 0 else 0
            avg_cycle_duration_ms = avg_cycle_frames * (1000.0 / fps)
        else:
            avg_cycle_duration_ms = 0

        cycle_count = len(zero_crossings) // 2

        return {
            'phase_distribution': {
                'ground_contact': 0.0,  # 正面视角无法检测
                'flight': 0.0,  # 正面视角无法检测
                'transition': 0.0,
            },
            'phase_duration_ms': {
                'ground_contact': 0.0,  # 不可用，设为0
                'flight': 0.0,  # 不可用，设为0
                'transition': 0.0,
            },
            'avg_cycle_duration': float(avg_cycle_duration_ms / 1000) if avg_cycle_duration_ms > 0 else 0,
            'avg_cycle_duration_ms': float(round(avg_cycle_duration_ms, 1)),
            'cycle_count': cycle_count,
            'ground_contact_times': [],  # 正面视角不输出
            'flight_times': [],  # 正面视角不输出
            'gait_rating': {'level': 'not_available', 'score': 0, 'description': '正面视角无法检测触地时间'},
            'is_frontal_estimate': True,
            'note': '正面视角无法准确检测触地/离地时刻，触地时间数据不可用',
        }

    def _get_empty_gait_cycle_frontal(self) -> Dict:
        """返回空的正面视角步态周期结果"""
        return {
            'phase_distribution': {'ground_contact': 0.0, 'flight': 0.0, 'transition': 0.0},
            'phase_duration_ms': {'ground_contact': 0.0, 'flight': 0.0, 'transition': 0.0},
            'avg_cycle_duration': 0.0,
            'avg_cycle_duration_ms': 0.0,
            'cycle_count': 0,
            'ground_contact_times': [],
            'flight_times': [],
            'gait_rating': {'level': 'unknown', 'score': 0, 'description': '数据不足'},
            'is_frontal_estimate': True,
        }

    def _calculate_shoulder_tilt(self, keypoints_sequence: List[Dict]) -> Dict:
        """
        计算肩部倾斜（正面视角特有指标）

        【优化】输出有物理意义的角度（度），而非图像坐标百分比
        使用肩宽归一化，提高可见性阈值到0.7

        分析跑步时的肩部上下倾斜模式：
        - 理想：左右肩交替上下运动，幅度适中
        - 问题：单侧肩膀持续偏高/偏低，或晃动过大
        """
        shoulder_heights_diff = []
        shoulder_widths = []
        vis_threshold = 0.7  # 【优化】提高可见性阈值

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']
            left = landmarks[11]
            right = landmarks[12]

            if left['visibility'] > vis_threshold and right['visibility'] > vis_threshold:
                # 高度差（正值表示左肩更低）
                height_diff = left['y_norm'] - right['y_norm']
                shoulder_heights_diff.append(height_diff)
                # 肩宽（用于归一化）
                shoulder_width = abs(left['x_norm'] - right['x_norm'])
                shoulder_widths.append(shoulder_width)
            else:
                shoulder_heights_diff.append(np.nan)
                shoulder_widths.append(np.nan)

        # 插值填充
        shoulder_heights_diff = self._interpolate_nans(np.array(shoulder_heights_diff))
        shoulder_widths = self._interpolate_nans(np.array(shoulder_widths))

        if len(shoulder_heights_diff) < 10:
            return {'tilt_mean': 0.0, 'tilt_std': 0.0, 'rating': {'level': 'unknown', 'score': 0}}

        # 【优化】使用肩宽归一化，转换为角度
        valid_widths = shoulder_widths[shoulder_widths > 0.01]
        # 【校准】改用分位数法计算肩宽基准，更鲁棒
        if len(valid_widths) > 0:
            q1 = np.percentile(valid_widths, 25)
            q3 = np.percentile(valid_widths, 75)
            median_shoulder_width = (q1 + q3) / 2  # 四分位中点
        else:
            median_shoulder_width = 0.22  # 更现实的默认值（从0.15改为0.22）

        # 转换为角度（度）
        raw_angles = np.degrees(np.arctan(shoulder_heights_diff / median_shoulder_width))
        # 【校准】添加1.3倍补偿系数，抵消arctan在小角度的压缩特性
        tilt_angles = raw_angles * 1.3

        # 【校准】减轻平滑强度，保留更多幅度信息
        if len(tilt_angles) > 5:
            window = min(5, len(tilt_angles) // 2 * 2 + 1)  # 窗口从7改为5
            if window >= 3:
                tilt_angles = savgol_filter(tilt_angles, window, 1)  # polyorder从2改为1

        # 统计分析（现在是角度）
        tilt_mean = np.mean(tilt_angles)
        tilt_std = np.std(tilt_angles)
        tilt_max = np.max(np.abs(tilt_angles))

        # 【优化】基于角度的评分标准
        if abs(tilt_mean) < 2 and 1 < tilt_std < 5:
            rating = {'level': 'excellent', 'score': 95, 'description': '肩部运动对称且协调'}
        elif abs(tilt_mean) < 4 and tilt_std < 8:
            rating = {'level': 'good', 'score': 80, 'description': '肩部运动良好'}
        elif abs(tilt_mean) < 6 or tilt_std < 12:
            rating = {'level': 'fair', 'score': 65, 'description': '肩部存在轻微不对称'}
        else:
            rating = {'level': 'poor', 'score': 45, 'description': '肩部晃动明显，建议关注'}

        return {
            'tilt_mean': float(tilt_mean),   # 现在是角度（度）
            'tilt_std': float(tilt_std),
            'tilt_max': float(tilt_max),
            'shoulder_width_ref': float(median_shoulder_width),  # 用于调试
            'rating': rating,
        }

    def _calculate_lower_limb_alignment(self, keypoints_sequence: List[Dict]) -> Dict:
        """
        计算下肢力线（膝内扣/外翻趋势）
        通过分析髋-膝-踝的横向对齐情况

        【优化】提高可见性阈值到0.7，过滤低质量帧

        返回：
        - 左右腿统计数据
        - 时序数据（用于图表展示）
        - 整体评级
        """
        vis_threshold = 0.7  # 【优化】提高可见性阈值
        left_valgus_angles = []  # 左腿外翻角
        right_valgus_angles = []  # 右腿外翻角
        left_valgus_timeseries = []  # 左腿时序数据（含nan）
        right_valgus_timeseries = []  # 右腿时序数据（含nan）
        hip_drop_angles = []  # 髋部下沉角度

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # ========== 左腿力线 ==========
            left_hip = landmarks[23]
            left_knee = landmarks[25]
            left_ankle = landmarks[27]

            left_angle = np.nan
            if all(p['visibility'] > vis_threshold for p in [left_hip, left_knee, left_ankle]):
                # 计算膝关节相对于髋-踝连线的横向偏移
                # 正值表示膝外翻（X向外），负值表示膝内扣
                hip_ankle_x = (left_hip['x_norm'] + left_ankle['x_norm']) / 2
                knee_offset = left_knee['x_norm'] - hip_ankle_x
                # 转换为角度（简化计算）
                hip_ankle_dist = abs(left_hip['y_norm'] - left_ankle['y_norm'])
                if hip_ankle_dist > 0.01:
                    left_angle = np.degrees(np.arctan(knee_offset / hip_ankle_dist))
                    left_valgus_angles.append(left_angle)

            left_valgus_timeseries.append(left_angle)

            # ========== 右腿力线 ==========
            right_hip = landmarks[24]
            right_knee = landmarks[26]
            right_ankle = landmarks[28]

            right_angle = np.nan
            if all(p['visibility'] > vis_threshold for p in [right_hip, right_knee, right_ankle]):
                hip_ankle_x = (right_hip['x_norm'] + right_ankle['x_norm']) / 2
                knee_offset = right_knee['x_norm'] - hip_ankle_x
                hip_ankle_dist = abs(right_hip['y_norm'] - right_ankle['y_norm'])
                if hip_ankle_dist > 0.01:
                    right_angle = np.degrees(np.arctan(knee_offset / hip_ankle_dist))
                    right_valgus_angles.append(right_angle)

            right_valgus_timeseries.append(right_angle)

            # ========== 髋部下沉 ==========
            if left_hip['visibility'] > vis_threshold and right_hip['visibility'] > vis_threshold:
                # 髋部Y坐标差（正值表示左侧更低）
                hip_diff_y = left_hip['y_norm'] - right_hip['y_norm']
                # 转换为角度估算
                hip_x_dist = abs(left_hip['x_norm'] - right_hip['x_norm'])
                if hip_x_dist > 0.01:
                    hip_drop_angle = np.degrees(np.arctan(hip_diff_y / hip_x_dist))
                    hip_drop_angles.append(hip_drop_angle)

        # ========== 统计分析 ==========
        def analyze_alignment(angles, side):
            if not angles:
                return {
                    'mean': 0, 'max': 0, 'min': 0, 'std': 0,
                    'issue': 'unknown', 'severity': 'unknown',
                    'count': 0
                }

            mean_angle = np.mean(angles)
            max_angle = np.max(angles)
            min_angle = np.min(angles)
            std_angle = np.std(angles)

            # 判断问题类型
            if mean_angle > 2:
                issue = 'valgus'  # 膝外翻
                if mean_angle < 5:
                    severity = 'mild'
                elif mean_angle < 8:
                    severity = 'moderate'
                else:
                    severity = 'severe'
            elif mean_angle < -2:
                issue = 'varus'  # 膝内扣
                if mean_angle > -5:
                    severity = 'mild'
                elif mean_angle > -8:
                    severity = 'moderate'
                else:
                    severity = 'severe'
            else:
                issue = 'normal'
                severity = 'none'

            # 动态偏移检测
            if std_angle > 3 and issue == 'normal':
                issue = 'unstable'
                severity = 'mild'

            return {
                'mean': float(mean_angle),
                'max': float(max_angle),
                'min': float(min_angle),
                'std': float(std_angle),
                'issue': issue,
                'severity': severity,
                'count': len(angles)
            }

        # ========== 髋部下沉统计（优化：使用median/P95/MAD） ==========
        hip_drop_stats = {}
        if hip_drop_angles:
            hip_arr = np.array(hip_drop_angles)
            median_val = np.median(hip_arr)
            hip_drop_stats = {
                'mean': float(median_val),  # 使用中位数替代均值
                'max': float(np.percentile(np.abs(hip_arr), 95)),  # P95替代max
                'std': float(np.median(np.abs(hip_arr - median_val))),  # MAD替代std
            }

        # ========== 生成简化曲线数据（用于图表） ==========
        # 最多100个点，平滑处理
        chart_data = self._prepare_alignment_chart_data(
            left_valgus_timeseries,
            right_valgus_timeseries,
            max_points=100
        )

        # ========== 左右对比统计（优化：添加IQR异常值过滤） ==========
        left_filtered = self._filter_outliers_iqr(np.array(left_valgus_angles)) if left_valgus_angles else []
        right_filtered = self._filter_outliers_iqr(np.array(right_valgus_angles)) if right_valgus_angles else []
        left_stats = analyze_alignment(left_filtered.tolist() if isinstance(left_filtered, np.ndarray) else left_filtered, 'left')
        right_stats = analyze_alignment(right_filtered.tolist() if isinstance(right_filtered, np.ndarray) else right_filtered, 'right')

        # 左右差异
        asymmetry = abs(left_stats['mean'] - right_stats['mean'])

        return {
            'left_leg': left_stats,
            'right_leg': right_stats,
            'hip_drop': hip_drop_stats,
            'asymmetry': float(asymmetry),
            'chart_data': chart_data,
            'overall_rating': self._rate_lower_limb_alignment(
                left_valgus_angles, right_valgus_angles
            ),
            # 兼容旧版接口
            'knee_valgus': {
                'left_mean': left_stats['mean'],
                'right_mean': right_stats['mean'],
                'left_max': left_stats['max'],
                'right_max': right_stats['max'],
            }
        }

    def _prepare_alignment_chart_data(self, left_series: List, right_series: List,
                                       max_points: int = 100) -> Dict:
        """
        准备下肢力线图表数据

        对时序数据进行降采样和平滑处理
        """
        left_arr = np.array(left_series)
        right_arr = np.array(right_series)

        n = len(left_arr)
        if n < 5:
            return {'left': [], 'right': [], 'time_pct': []}

        # 插值填充nan
        left_filled = self._interpolate_nans(left_arr)
        right_filled = self._interpolate_nans(right_arr)

        # 降采样
        if n > max_points:
            indices = np.linspace(0, n-1, max_points, dtype=int)
            left_sampled = left_filled[indices]
            right_sampled = right_filled[indices]
        else:
            indices = np.arange(n)
            left_sampled = left_filled
            right_sampled = right_filled

        # 轻微平滑
        if len(left_sampled) > 5:
            try:
                left_smooth = savgol_filter(left_sampled, min(5, len(left_sampled)//2*2+1), 2)
                right_smooth = savgol_filter(right_sampled, min(5, len(right_sampled)//2*2+1), 2)
            except Exception:
                left_smooth = left_sampled
                right_smooth = right_sampled
        else:
            left_smooth = left_sampled
            right_smooth = right_sampled

        # 时间百分比
        time_pct = (indices / n * 100).tolist()

        return {
            'left': [float(v) for v in left_smooth],
            'right': [float(v) for v in right_smooth],
            'time_pct': time_pct
        }

    def _rate_lower_limb_alignment(self, left_angles: List, right_angles: List) -> Dict:
        """评估下肢力线（提高灵敏度）"""
        if not left_angles or not right_angles:
            return {'level': 'unknown', 'score': 0, 'description': '数据不足'}

        # 使用平均偏移和最大偏移的加权组合
        left_mean = np.mean(np.abs(left_angles)) if left_angles else 0
        right_mean = np.mean(np.abs(right_angles)) if right_angles else 0
        mean_deviation = (left_mean + right_mean) / 2

        max_deviation = max(
            np.max(np.abs(left_angles)) if left_angles else 0,
            np.max(np.abs(right_angles)) if right_angles else 0
        )

        # 综合得分：70%平均偏移 + 30%最大偏移
        combined_deviation = mean_deviation * 0.7 + max_deviation * 0.3

        # 更严格的阈值
        if combined_deviation < 2:
            return {'level': 'excellent', 'score': 100, 'description': '下肢力线非常标准'}
        elif combined_deviation < 4:
            return {'level': 'good', 'score': 85, 'description': '下肢力线良好'}
        elif combined_deviation < 6:
            return {'level': 'fair', 'score': 70, 'description': '存在轻度膝关节偏移'}
        elif combined_deviation < 10:
            return {'level': 'moderate', 'score': 55, 'description': '膝关节偏移需要注意'}
        else:
            return {'level': 'poor', 'score': 40, 'description': '膝关节偏移明显，建议关注'}

    def _calculate_gait_symmetry_detailed(self, keypoints_sequence: List[Dict]) -> Dict:
        """
        详细的步态对称性分析
        """
        # 收集左右侧关键点运动数据
        left_ankle_y = []
        right_ankle_y = []
        left_knee_angles = []
        right_knee_angles = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # 脚踝Y坐标
            if landmarks[27]['visibility'] > 0.5:
                left_ankle_y.append(landmarks[27]['y_norm'])
            if landmarks[28]['visibility'] > 0.5:
                right_ankle_y.append(landmarks[28]['y_norm'])

            # 膝关节角度
            left_angle = self._calculate_joint_angle_safe(
                landmarks[23], landmarks[25], landmarks[27]
            )
            right_angle = self._calculate_joint_angle_safe(
                landmarks[24], landmarks[26], landmarks[28]
            )

            if not np.isnan(left_angle):
                left_knee_angles.append(left_angle)
            if not np.isnan(right_angle):
                right_knee_angles.append(right_angle)

        # 计算对称性指标（改用运动幅度对比，避免相位差影响）
        symmetry_scores = []

        # 脚踝运动幅度对称性（左右腿是交替运动，不适合用相关系数）
        if len(left_ankle_y) > 10 and len(right_ankle_y) > 10:
            left_range = np.max(left_ankle_y) - np.min(left_ankle_y)
            right_range = np.max(right_ankle_y) - np.min(right_ankle_y)

            # 幅度差异百分比（优化：使用较大值作为分母，更公平）
            max_range = max(left_range, right_range)
            if max_range > 0.01:  # 避免除零
                range_diff = abs(left_range - right_range) / max_range * 100
                # 【校准】放宽阈值，适应2D检测精度
                if range_diff < 25:      # 从15改为25
                    ankle_score = 95
                elif range_diff < 45:    # 从30改为45
                    ankle_score = 80
                elif range_diff < 65:    # 从50改为65
                    ankle_score = 65
                else:
                    ankle_score = 50
                symmetry_scores.append(ankle_score)

        # 膝关节角度幅度对称性
        if len(left_knee_angles) > 10 and len(right_knee_angles) > 10:
            left_range = np.max(left_knee_angles) - np.min(left_knee_angles)
            right_range = np.max(right_knee_angles) - np.min(right_knee_angles)

            # 优化：使用较大值作为分母
            max_range = max(left_range, right_range)
            if max_range > 1:  # 角度阈值
                range_diff = abs(left_range - right_range) / max_range * 100
                # 【校准】放宽阈值，适应2D检测精度
                if range_diff < 20:      # 从15改为20（角度更敏感，放宽幅度稍小）
                    knee_score = 95
                elif range_diff < 40:    # 从30改为40
                    knee_score = 80
                elif range_diff < 60:    # 从50改为60
                    knee_score = 65
                else:
                    knee_score = 50
                symmetry_scores.append(knee_score)

            # 左右平均值差异（保留用于详细分析）
            left_mean = np.mean(left_knee_angles)
            right_mean = np.mean(right_knee_angles)
            angle_diff = abs(left_mean - right_mean)
        else:
            angle_diff = 0

        overall_symmetry = np.mean(symmetry_scores) if symmetry_scores else 75

        return {
            'overall_score': float(overall_symmetry),
            'knee_angle_difference': float(angle_diff),
            'rating': self._rate_symmetry(overall_symmetry, angle_diff)
        }

    def _rate_symmetry(self, symmetry_score: float, angle_diff: float) -> Dict:
        """
        评估对称性

        【校准】改用加权逻辑，避免AND条件过于严格
        - 幅度对称性权重70%
        - 均值差异权重30%（每超过3°扣3分）
        """
        # 计算角度差异惩罚（每超过3°扣3分，最多扣30分）
        angle_penalty = min(30, max(0, (angle_diff - 3) * 3))
        # 综合得分 = 幅度对称性*0.7 + 均值差异部分*0.3
        overall = symmetry_score * 0.7 + (100 - angle_penalty) * 0.3

        if overall >= 88:
            return {'level': 'excellent', 'score': 95, 'description': '步态高度对称'}
        elif overall >= 75:
            return {'level': 'good', 'score': 80, 'description': '步态对称性良好'}
        elif overall >= 60:
            return {'level': 'fair', 'score': 65, 'description': '存在轻度不对称'}
        else:
            return {'level': 'poor', 'score': 50, 'description': '步态不对称明显'}

    def _extract_sway_component(self, positions: list, fps: float) -> np.ndarray:
        """
        自去趋势提取横摆振荡分量

        原理:
        1. 大窗口 Savitzky-Golay 拟合运动轨迹(多项式曲线)
        2. 原始信号 - 轨迹 = 振荡分量 + 高频噪声
        3. Butterworth 低通 (cutoff 3Hz) 去除高频噪声

        优势(相比旧版肩部相对坐标):
        - 不使用肩部参考 → 消除肩部反向振荡放大
        - SavGol 跟踪任意曲线轨迹 → 处理非直线运动
        - 低通滤波 → 去除镜头抖动和检测噪声
        """
        arr = np.array(positions, dtype=float)
        n = len(arr)

        if n < 8:
            return arr - np.mean(arr)

        # Step 1: 大窗口 SavGol 提取轨迹趋势
        # 窗口 = 40% 序列长度, 至少 7 帧, 必须奇数
        trend_window = max(7, int(n * 0.4))
        if trend_window % 2 == 0:
            trend_window += 1
        # 确保窗口不超过序列长度
        if trend_window > n:
            trend_window = n if n % 2 == 1 else n - 1
        trajectory = savgol_filter(arr, trend_window, polyorder=2)

        # Step 2: 减去趋势得到振荡分量
        oscillation = arr - trajectory

        # Step 3: 低通滤波去噪 (cutoff 3Hz)
        if fps > 0 and n >= 15:
            nyquist = fps / 2.0
            cutoff = min(3.0 / nyquist, 0.9)
            if cutoff > 0.05:
                b, a = butter(2, cutoff, btype='low')
                oscillation = filtfilt(b, a, oscillation)

        return oscillation

    def _calculate_lateral_stability(self, keypoints_sequence: List[Dict],
                                     fps: float = 30.0) -> Dict:
        """
        计算横向稳定性（正面视角）

        【优化v2】使用自去趋势(Self-Detrending)替代肩部相对坐标
        解决三个叠加问题:
        1. 肩部反向振荡放大 → 不再使用肩部参考
        2. 人物非直线运动 → SavGol大窗口拟合轨迹后减去
        3. 镜头抖动/检测噪声 → Butterworth低通3Hz去除
        """
        hip_x_abs_positions = []      # 绝对坐标(不再计算相对坐标)
        shoulder_x_abs_positions = []
        hip_widths = []
        shoulder_widths = []
        vis_threshold = 0.7

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # 肩部数据采集
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            if (left_shoulder['visibility'] > vis_threshold and
                    right_shoulder['visibility'] > vis_threshold):
                shoulder_x = (left_shoulder['x_norm'] + right_shoulder['x_norm']) / 2
                shoulder_x_abs_positions.append(shoulder_x)
                shoulder_widths.append(abs(left_shoulder['x_norm'] - right_shoulder['x_norm']))

            # 髋部数据采集(绝对坐标)
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            if (left_hip['visibility'] > vis_threshold and
                    right_hip['visibility'] > vis_threshold):
                hip_x = (left_hip['x_norm'] + right_hip['x_norm']) / 2
                hip_x_abs_positions.append(hip_x)
                hip_widths.append(abs(left_hip['x_norm'] - right_hip['x_norm']))

        # 计算身体尺寸参考值
        valid_hip_widths = [w for w in hip_widths if w > 0.01]
        valid_shoulder_widths = [w for w in shoulder_widths if w > 0.01]

        if valid_hip_widths:
            median_hip_width = np.median(valid_hip_widths)
        elif valid_shoulder_widths:
            median_hip_width = np.median(valid_shoulder_widths) * 0.87
        else:
            median_hip_width = 0.18

        if valid_shoulder_widths:
            median_shoulder_width = np.median(valid_shoulder_widths)
        else:
            median_shoulder_width = 0.22

        # 【核心改进】自去趋势提取横摆分量
        # 髋部横摆
        if len(hip_x_abs_positions) >= 5:
            hip_oscillation = self._extract_sway_component(hip_x_abs_positions, fps)
            q75, q25 = np.percentile(hip_oscillation, [75, 25])
            hip_sway_raw = q75 - q25
            hip_sway = (hip_sway_raw / median_hip_width) * 100 if median_hip_width > 0.01 else 0
        else:
            hip_sway = 0

        # 肩部横摆
        if len(shoulder_x_abs_positions) >= 5:
            shoulder_oscillation = self._extract_sway_component(shoulder_x_abs_positions, fps)
            q75, q25 = np.percentile(shoulder_oscillation, [75, 25])
            shoulder_sway_raw = q75 - q25
            shoulder_sway = (shoulder_sway_raw / median_shoulder_width) * 100 if median_shoulder_width > 0.01 else 0
        else:
            shoulder_sway = 0

        # 综合评分
        total_sway = (hip_sway + shoulder_sway) / 2
        stability_score = max(0, 100 - total_sway * 4)

        return {
            'hip_sway': float(hip_sway),
            'shoulder_sway': float(shoulder_sway),
            'stability_score': float(stability_score),
            'hip_width_ref': float(median_hip_width),
            'shoulder_width_ref': float(median_shoulder_width),
            'rating': self._rate_lateral_stability(stability_score)
        }

    def _rate_lateral_stability(self, score: float) -> Dict:
        """评估横向稳定性"""
        if score >= 90:
            return {'level': 'excellent', 'score': 100, 'description': '横向稳定性优秀'}
        elif score >= 75:
            return {'level': 'good', 'score': 80, 'description': '横向稳定性良好'}
        elif score >= 60:
            return {'level': 'fair', 'score': 60, 'description': '存在横向摆动'}
        else:
            return {'level': 'poor', 'score': 40, 'description': '横向摆动明显，影响效率'}

    # ==================== 通用计算方法 ====================

    def _filter_by_visibility(self, keypoints_sequence: List[Dict],
                              joint_indices: List[int],
                              threshold: float = 0.5) -> Tuple[List[Dict], Dict]:
        """
        根据可见度过滤数据（在分析层统一处理）

        策略：
        - threshold >= 0.7: 高置信度数据
        - threshold >= 0.5: 中等置信度数据（默认）
        - threshold >= 0.3: 低置信度数据

        Args:
            keypoints_sequence: 关键点序列
            joint_indices: 需要检查的关节索引列表（MediaPipe索引）
            threshold: 可见度阈值

        Returns:
            filtered: 过滤后的帧列表
            stats: 统计信息（包含警告）
        """
        filtered = []
        low_conf_count = 0
        total_frames = len(keypoints_sequence)

        for frame_data in keypoints_sequence:
            landmarks = frame_data['landmarks']

            # 检查所有指定关节的可见度
            all_visible = all(
                landmarks[idx]['visibility'] >= threshold
                for idx in joint_indices
                if idx < len(landmarks)
            )

            if all_visible:
                filtered.append(frame_data)
            else:
                # 统计低置信度帧（visibility < 0.3）
                if any(landmarks[idx]['visibility'] < 0.3
                       for idx in joint_indices
                       if idx < len(landmarks)):
                    low_conf_count += 1

        # 生成统计信息
        stats = {
            'total_frames': total_frames,
            'filtered_frames': len(filtered),
            'filter_ratio': len(filtered) / total_frames if total_frames > 0 else 0,
            'low_conf_count': low_conf_count,
            'low_conf_ratio': low_conf_count / total_frames if total_frames > 0 else 0,
            'warning': None
        }

        # 如果低置信度帧过多（>30%），添加警告
        if stats['low_conf_ratio'] > 0.3:
            stats['warning'] = f"数据质量警告：{stats['low_conf_ratio']*100:.1f}%的帧存在低可见度关键点"

        return filtered, stats

    def _calculate_joint_angle_safe(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """安全的关节角度计算（2D版，p2为关节点）"""
        if p1['visibility'] < 0.5 or p2['visibility'] < 0.5 or p3['visibility'] < 0.5:
            return np.nan

        try:
            v1 = np.array([p1['x_norm'] - p2['x_norm'], p1['y_norm'] - p2['y_norm']])
            v2 = np.array([p3['x_norm'] - p2['x_norm'], p3['y_norm'] - p2['y_norm']])

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 < 1e-6 or norm2 < 1e-6:
                return np.nan

            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            return np.degrees(angle)
        except:
            return np.nan

    def _calculate_joint_angle_3d(self, pose_3d: Dict, joint1: str, joint2: str, joint3: str,
                                    use_projection: bool = True) -> float:
        """
        使用3D坐标计算关节角度（3D版，joint2为关节点）

        【优化】根据视角投影到对应平面，消除相机角度偏差：
        - 侧面视角：投影到XY平面（矢状面）
        - 正面/背面：投影到YZ平面（冠状面）

        Args:
            pose_3d: H36M格式的3D姿态字典
            joint1, joint2, joint3: 关节名称
            use_projection: 是否根据视角投影（默认True）

        Returns:
            角度（度），计算失败返回np.nan
        """
        try:
            p1 = get_3d_point_from_pose(pose_3d, joint1)
            p2 = get_3d_point_from_pose(pose_3d, joint2)
            p3 = get_3d_point_from_pose(pose_3d, joint3)

            if p1 is None or p2 is None or p3 is None:
                return np.nan

            # 根据视角使用投影角度计算
            if use_projection and self.view_angle:
                return calculate_projected_joint_angle(p1, p2, p3, self.view_angle)
            else:
                return calculate_3d_joint_angle(p1, p2, p3)
        except:
            return np.nan

    def _smooth_and_filter_angles(self, angles: List[float]) -> List[float]:
        """平滑和滤波角度序列"""
        angles = np.array(angles)

        # 插值处理NaN
        if np.any(np.isnan(angles)):
            valid_idx = ~np.isnan(angles)
            if np.sum(valid_idx) < 2:
                return [0.0] * len(angles)

            interp_func = interp1d(
                np.where(valid_idx)[0],
                angles[valid_idx],
                kind='linear',
                fill_value='extrapolate'
            )
            angles = interp_func(np.arange(len(angles)))

        # 异常值检测（3-sigma原则）
        mean = np.mean(angles)
        std = np.std(angles)
        if std > 0:
            outliers = np.abs(angles - mean) > 3 * std
            if np.any(outliers):
                angles[outliers] = mean

        # 低通滤波
        if len(angles) > 6:
            b, a = butter(2, 0.2, btype='low')
            angles = filtfilt(b, a, angles)

        # Savitzky-Golay平滑
        if len(angles) > self.smooth_window:
            window = min(self.smooth_window, len(angles))
            if window % 2 == 0:
                window -= 1
            angles = savgol_filter(angles, window_length=window, polyorder=2)

        return angles.tolist()

    def _calculate_cadence_improved(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """改进的步频计算（侧面：仅踝关节主方法）"""
        # 方法1: 基于膝关节角度
        cadence1, step_count1 = self._cadence_from_knee_angle(keypoints_sequence, fps)

        # 方法2: 基于脚踝Y坐标
        cadence2, step_count2, ankle_quality = self._cadence_from_ankle_position(keypoints_sequence, fps)

        # 方法3: 基于髋部运动
        cadence3, step_count3 = self._cadence_from_hip_motion(keypoints_sequence, fps)

        # 只使用踝关节方法（用户要求）
        cadences = [cadence1, cadence2, cadence3]
        step_counts = [step_count1, step_count2, step_count3]
        weights = [0, 1, 0]  # 仅踝关节

        valid_data = [(c, s, w) for c, s, w in zip(cadences, step_counts, weights) if c > 0 and s > 0]
        if valid_data:
            total_weight = sum(w for _, _, w in valid_data)
            weighted_cadence = sum(c * w for c, _, w in valid_data) / total_weight
            # 步数也使用加权平均，保持与步频一致
            weighted_step_count = sum(s * w for _, s, w in valid_data) / total_weight
            avg_step_count = int(round(weighted_step_count))
        else:
            weighted_cadence = 0
            avg_step_count = 0

        duration = len(keypoints_sequence) / fps

        # 计算预期步数（基于加权步频）用于验证
        expected_steps = weighted_cadence * duration / 60 if weighted_cadence > 0 else 0

        confidence = float(ankle_quality.get('confidence', 0.0))
        if confidence <= 0:
            # 仅用于调试的回退置信度，不改变主算法选择
            confidence = self._calculate_cadence_confidence([cadence1, cadence2, cadence3])

        return {
            'cadence': float(weighted_cadence),
            'step_count': avg_step_count,
            'duration': float(duration),
            'expected_steps': float(expected_steps),  # 新增：理论步数（用于解释）
            'cadence_knee': float(cadence1),
            'cadence_ankle': float(cadence2),
            'cadence_hip': float(cadence3),
            'step_count_knee': step_count1,
            'step_count_ankle': step_count2,
            'step_count_hip': step_count3,
            'confidence': float(confidence),
            'rating': self._rate_cadence(weighted_cadence),
            'method': 'ankle_y_only',
            'reliability': ankle_quality.get('reliability', 'low'),
            'quality_flags': ankle_quality,
        }

    def _rate_cadence(self, cadence: float) -> Dict:
        """评估步频（新标准：5个等级）"""
        if cadence >= 185:
            return {'level': 'elite', 'score': 100, 'description': '精英'}
        elif cadence >= 175:
            return {'level': 'excellent', 'score': 90, 'description': '优秀'}
        elif cadence >= 165:
            return {'level': 'good', 'score': 75, 'description': '良好'}
        elif cadence >= 155:
            return {'level': 'fair', 'score': 60, 'description': '一般'}
        else:
            return {'level': 'poor', 'score': 45, 'description': '较差'}

    def _cadence_from_knee_angle(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[float, int]:
        """从膝关节角度计算步频"""
        knee_angles = []
        for kp in keypoints_sequence:
            angle = self._calculate_joint_angle_safe(
                kp['landmarks'][24], kp['landmarks'][26], kp['landmarks'][28]
            )
            knee_angles.append(angle if not np.isnan(angle) else 0)

        if len(knee_angles) < 10:
            return 0.0, 0

        knee_angles = self._smooth_signal_advanced(np.array(knee_angles))
        peaks, _ = find_peaks(-knee_angles, distance=int(fps * 0.3), prominence=5)

        step_count = len(peaks)
        duration = len(keypoints_sequence) / fps
        cadence = (step_count / duration) * 60 if duration > 0 else 0

        return cadence, step_count

    def _cadence_from_ankle_position(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[float, int, Dict]:
        """从脚踝位置计算步频（自适应峰值 + 可靠性评估）"""
        left_ankle_y = []
        right_ankle_y = []
        left_visible = []
        right_visible = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][27]
            right = kp['landmarks'][28]
            lv = left['visibility'] > 0.5
            rv = right['visibility'] > 0.5
            left_visible.append(lv)
            right_visible.append(rv)
            left_ankle_y.append(left['y_norm'] if lv else np.nan)
            right_ankle_y.append(right['y_norm'] if rv else np.nan)

        frame_count = len(keypoints_sequence)
        left_vis_ratio = float(np.mean(left_visible)) if frame_count > 0 else 0.0
        right_vis_ratio = float(np.mean(right_visible)) if frame_count > 0 else 0.0

        left_ankle_y = self._interpolate_nans(np.array(left_ankle_y))
        right_ankle_y = self._interpolate_nans(np.array(right_ankle_y))

        if len(left_ankle_y) < 10:
            return 0.0, 0, {
                'confidence': 0.0,
                'reliability': 'low',
                'rejection_reason': 'insufficient_frames',
            }

        left_ankle_y = self._smooth_signal_advanced(left_ankle_y)
        right_ankle_y = self._smooth_signal_advanced(right_ankle_y)
        duration = len(keypoints_sequence) / fps if fps > 0 else 0.0

        combined = np.concatenate([left_ankle_y, right_ankle_y])
        signal_std = float(np.nanstd(combined))
        p95 = float(np.nanpercentile(combined, 95))
        p05 = float(np.nanpercentile(combined, 5))
        signal_range = max(1e-6, p95 - p05)
        # 峰值法主路径：单脚设置更严格的最小间隔，减少“单步多峰”过检
        min_distance = max(5, int(fps * 0.32))
        prominence = max(0.0030, signal_std * 0.36, signal_range * 0.08)
        left_peaks, left_props = find_peaks(
            left_ankle_y,
            distance=min_distance,
            prominence=prominence,
        )
        right_peaks, right_props = find_peaks(
            right_ankle_y,
            distance=min_distance,
            prominence=prominence,
        )

        left_peak_count = int(len(left_peaks))
        right_peak_count = int(len(right_peaks))
        raw_step_count = left_peak_count + right_peak_count
        cadence_peak = (raw_step_count / duration) * 60 if duration > 0 else 0.0

        left_cadence = (left_peak_count / duration) * 60 if duration > 0 else 0.0
        right_cadence = (right_peak_count / duration) * 60 if duration > 0 else 0.0

        merged = np.sort(np.concatenate([left_peaks, right_peaks])) if raw_step_count > 1 else np.array([])
        interval_cv = 1.0
        if merged.size >= 3:
            intervals = np.diff(merged.astype(float))
            mean_interval = float(np.mean(intervals))
            if mean_interval > 1e-6:
                interval_cv = float(np.std(intervals) / mean_interval)

        left_prom = float(np.mean(left_props.get('prominences', [0.0]))) if left_peak_count > 0 else 0.0
        right_prom = float(np.mean(right_props.get('prominences', [0.0]))) if right_peak_count > 0 else 0.0
        avg_prom = (left_prom + right_prom) / 2.0

        # 自相关校验路径：对周期性更敏感，可抑制峰值法高估
        cadence_autocorr = 0.0
        autocorr_conf = 0.0
        try:
            support_signal = np.maximum(left_ankle_y, right_ankle_y)
            support_signal = self._detrend_signal(support_signal)
            support_filtered = self._bandpass_filter(support_signal, fps, low_freq=1.2, high_freq=4.2)
            cadence_autocorr, autocorr_conf = self._extract_cadence_autocorr(support_filtered, fps)
        except Exception:
            cadence_autocorr, autocorr_conf = 0.0, 0.0

        # 融合：当峰值法显著高于自相关时，偏向自相关结果，防止高估
        if cadence_peak > 0 and cadence_autocorr > 0:
            diff_ratio = abs(cadence_peak - cadence_autocorr) / max(cadence_peak, cadence_autocorr)
            if diff_ratio < 0.12:
                cadence = cadence_peak * 0.60 + cadence_autocorr * 0.40
            elif cadence_peak > cadence_autocorr * 1.18:
                cadence = cadence_peak * 0.25 + cadence_autocorr * 0.75
            elif cadence_autocorr > cadence_peak * 1.18:
                cadence = cadence_peak * 0.75 + cadence_autocorr * 0.25
            else:
                cadence = cadence_peak if autocorr_conf < 0.45 else cadence_autocorr
        elif cadence_peak > 0:
            cadence = cadence_peak
        elif cadence_autocorr > 0:
            cadence = cadence_autocorr
        else:
            cadence = 0.0

        # 跑步步频常见有效范围（侧面分析场景）
        cadence = float(np.clip(cadence, 120.0, 220.0)) if cadence > 0 else 0.0
        step_count = int(round(cadence * duration / 60.0)) if duration > 0 else 0
        method_name = 'peak_autocorr_fusion'

        lr_consistency = 1.0
        if max(left_cadence, right_cadence) > 0:
            lr_consistency = 1.0 - abs(left_cadence - right_cadence) / max(left_cadence, right_cadence)
            lr_consistency = float(np.clip(lr_consistency, 0.0, 1.0))

        prom_score = float(np.clip(avg_prom / (prominence * 2.5 + 1e-6), 0.0, 1.0))
        interval_score = float(np.clip(1.0 - interval_cv / 0.35, 0.0, 1.0))
        method_consistency = 0.5
        if cadence_peak > 0 and cadence_autocorr > 0:
            method_consistency = float(np.clip(1.0 - abs(cadence_peak - cadence_autocorr) / max(cadence_peak, cadence_autocorr), 0.0, 1.0))
        elif cadence_peak > 0 or cadence_autocorr > 0:
            method_consistency = 0.6

        visibility_score = float(np.clip(((left_vis_ratio + right_vis_ratio) / 2.0) / 0.85, 0.0, 1.0))
        expected_min_steps = max(4.0, duration * 2.5)
        count_score = float(np.clip(step_count / expected_min_steps, 0.0, 1.0))

        confidence = (
            visibility_score * 0.30 +
            lr_consistency * 0.20 +
            interval_score * 0.20 +
            prom_score * 0.10 +
            float(np.clip(autocorr_conf, 0.0, 1.0)) * 0.10 +
            method_consistency * 0.10
        )

        # 可见度差时限制过高置信度，避免误判
        if min(left_vis_ratio, right_vis_ratio) < 0.55:
            confidence = min(confidence, 0.45)

        # 若节奏过快且周期一致性不足，惩罚置信度（典型高估模式）
        if cadence > 205 and interval_score < 0.60:
            confidence *= 0.65

        if step_count < 4:
            confidence = min(confidence, 0.30)

        if confidence >= 0.78:
            reliability = 'high'
        elif confidence >= 0.55:
            reliability = 'medium'
        else:
            reliability = 'low'

        return float(cadence), int(step_count), {
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'reliability': reliability,
            'left_visible_ratio': float(left_vis_ratio),
            'right_visible_ratio': float(right_vis_ratio),
            'left_peaks': int(left_peak_count),
            'right_peaks': int(right_peak_count),
            'left_cadence': float(left_cadence),
            'right_cadence': float(right_cadence),
            'lr_consistency': float(lr_consistency),
            'interval_cv': float(interval_cv),
            'interval_score': float(interval_score),
            'prominence': float(prominence),
            'cadence_peak': float(cadence_peak),
            'cadence_autocorr': float(cadence_autocorr),
            'autocorr_confidence': float(autocorr_conf),
            'method_source': method_name,
            'method_consistency': float(method_consistency),
            'count_score': float(count_score),
            'adaptive': True,
        }

    def _cadence_from_hip_motion(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[float, int]:
        """从髋部运动计算步频"""
        hip_y = []
        for kp in keypoints_sequence:
            left_hip = kp['landmarks'][23]
            right_hip = kp['landmarks'][24]
            if left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
                hip_y.append((left_hip['y_norm'] + right_hip['y_norm']) / 2)

        if len(hip_y) < 10:
            return 0.0, 0

        hip_y = self._smooth_signal_advanced(np.array(hip_y))
        peaks, _ = find_peaks(hip_y, distance=int(fps * 0.15))

        step_count = len(peaks)
        duration = len(hip_y) / fps
        cadence = (step_count / duration) * 60 if duration > 0 else 0

        return cadence, step_count

    def _calculate_cadence_confidence(self, cadences: List[float]) -> float:
        """计算步频置信度"""
        valid_cadences = [c for c in cadences if c > 0]
        if len(valid_cadences) < 2:
            return 0.5

        mean = np.mean(valid_cadences)
        std = np.std(valid_cadences)
        cv = std / mean if mean > 0 else 1.0

        confidence = max(0, 1.0 - cv)
        return float(confidence)

    def _calculate_stride_info(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """步态信息计算"""
        left_ankle_x = []
        right_ankle_x = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][27]
            right = kp['landmarks'][28]
            if left['visibility'] > 0.5:
                left_ankle_x.append(left['x_norm'])
            if right['visibility'] > 0.5:
                right_ankle_x.append(right['x_norm'])

        stride_length = 0.0
        if left_ankle_x and right_ankle_x:
            left_range = np.max(left_ankle_x) - np.min(left_ankle_x)
            right_range = np.max(right_ankle_x) - np.min(right_ankle_x)
            stride_length = (left_range + right_range) / 2

        ground_contact_ratio = self._estimate_ground_contact_ratio(keypoints_sequence, fps)

        return {
            'stride_length_norm': float(stride_length),
            'ground_contact_ratio': float(ground_contact_ratio),
            'flight_time_ratio': float(1.0 - ground_contact_ratio)
        }

    def _estimate_ground_contact_ratio(self, keypoints_sequence: List[Dict], fps: float) -> float:
        """估算触地时间比例"""
        ankle_velocities = []
        for i in range(1, len(keypoints_sequence)):
            curr = keypoints_sequence[i]['landmarks'][27]
            prev = keypoints_sequence[i - 1]['landmarks'][27]
            if curr['visibility'] > 0.5 and prev['visibility'] > 0.5:
                vy = (curr['y_norm'] - prev['y_norm']) * fps
                ankle_velocities.append(abs(vy))

        if not ankle_velocities:
            return 0.5

        threshold = np.percentile(ankle_velocities, 50)
        ground_frames = sum(1 for v in ankle_velocities if v < threshold)
        return ground_frames / len(ankle_velocities)

    def _calculate_stability_side_view(self, keypoints_sequence: List[Dict]) -> Dict:
        """侧面视角稳定性计算（不包含左右对称性）"""
        trunk_stability = self._calculate_trunk_stability(keypoints_sequence)
        head_stability = self._calculate_head_stability(keypoints_sequence)

        # 侧面视角只评估躯干和头部稳定性
        overall = (trunk_stability * 0.6 + head_stability * 0.4)

        return {
            'overall': float(overall),
            'trunk': float(trunk_stability),
            'head': float(head_stability),
            'rating': self._rate_stability(overall)
        }

    def _calculate_stability_front_view(self, keypoints_sequence: List[Dict]) -> Dict:
        """正面视角稳定性计算（移除对称性，提高肩部晃动权重）"""
        trunk_stability = self._calculate_trunk_stability(keypoints_sequence)
        head_stability = self._calculate_head_stability(keypoints_sequence)
        shoulder_sway = self._calculate_shoulder_sway(keypoints_sequence)

        # 正面视角：提高肩部晃动权重（移除对称性）
        overall = (trunk_stability * 0.35 + head_stability * 0.15 + shoulder_sway * 0.50)

        return {
            'overall': float(overall),
            'trunk': float(trunk_stability),
            'head': float(head_stability),
            'shoulder_sway': float(shoulder_sway),
            'rating': self._rate_stability(overall)
        }

    def _calculate_stability_improved(self, keypoints_sequence: List[Dict]) -> Dict:
        """改进的稳定性计算（兼容用途）"""
        trunk_stability = self._calculate_trunk_stability(keypoints_sequence)
        head_stability = self._calculate_head_stability(keypoints_sequence)
        shoulder_sway = self._calculate_shoulder_sway(keypoints_sequence)

        # 移除对称性，使用肩部晃动
        overall = (trunk_stability * 0.40 + head_stability * 0.20 + shoulder_sway * 0.40)

        return {
            'overall': float(overall),
            'trunk': float(trunk_stability),
            'head': float(head_stability),
            'shoulder_sway': float(shoulder_sway),
            'rating': self._rate_stability(overall)
        }

    def _calculate_shoulder_sway(self, keypoints_sequence: List[Dict]) -> float:
        """计算肩部晃动幅度（用于正面视角对称性评估）"""
        left_shoulder_y = []
        right_shoulder_y = []
        shoulder_diff = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][11]
            right = kp['landmarks'][12]

            if left['visibility'] > 0.5 and right['visibility'] > 0.5:
                left_shoulder_y.append(left['y_norm'])
                right_shoulder_y.append(right['y_norm'])
                # 左右肩膀高度差（正常跑步时应该交替变化）
                shoulder_diff.append(abs(left['y_norm'] - right['y_norm']))

        if len(shoulder_diff) < 10:
            return 50.0

        # 肩部晃动分析
        # 1. 左右高度差的标准差（越小越稳定）
        diff_std = np.std(shoulder_diff) * 100
        # 2. 左右高度差的平均值（过大说明存在不对称）
        diff_mean = np.mean(shoulder_diff) * 100

        # 评分逻辑：差值小且稳定得分高
        score = 100 - min(diff_std * 5 + diff_mean * 3, 100)
        return max(0, score)

    def _rate_stability(self, score: float) -> Dict:
        """评估稳定性"""
        if score >= 85:
            return {'level': 'excellent', 'score': 100, 'description': '动作非常稳定'}
        elif score >= 70:
            return {'level': 'good', 'score': 80, 'description': '动作稳定性良好'}
        elif score >= 55:
            return {'level': 'fair', 'score': 60, 'description': '稳定性一般'}
        else:
            return {'level': 'poor', 'score': 40, 'description': '动作稳定性需要改善'}

    def _calculate_trunk_stability(self, keypoints_sequence: List[Dict]) -> float:
        """
        计算躯干稳定性（重写版）

        测量：躯干倾斜角度（肩-髋连线与垂直方向的夹角）的标准差
        角度变异越小 → 稳定性越高
        """
        trunk_angles = []
        vis_threshold = 0.5

        for kp in keypoints_sequence:
            ls, rs = kp['landmarks'][11], kp['landmarks'][12]
            lh, rh = kp['landmarks'][23], kp['landmarks'][24]

            if (ls['visibility'] > vis_threshold and rs['visibility'] > vis_threshold and
                lh['visibility'] > vis_threshold and rh['visibility'] > vis_threshold):
                shoulder_x = (ls['x_norm'] + rs['x_norm']) / 2
                shoulder_y = (ls['y_norm'] + rs['y_norm']) / 2
                hip_x = (lh['x_norm'] + rh['x_norm']) / 2
                hip_y = (lh['y_norm'] + rh['y_norm']) / 2

                dx = shoulder_x - hip_x
                dy = shoulder_y - hip_y  # y_norm中y向下为正
                # 躯干与垂直方向的夹角
                trunk_angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
                trunk_angles.append(trunk_angle)

        if len(trunk_angles) < 10:
            return 50.0

        trunk_angles = np.array(trunk_angles)

        # 轻度平滑（仅去噪，不压制振幅）
        if len(trunk_angles) > 5:
            window = min(5, len(trunk_angles) // 2 * 2 + 1)
            if window >= 3:
                trunk_angles = savgol_filter(trunk_angles, window, 1)

        angle_std = np.std(trunk_angles)

        # 阈值: <1.5°→95, 1.5-3°→95~80, 3-5°→80~65, >5°→<65
        if angle_std < 1.5:
            stability = 95
        elif angle_std < 3.0:
            stability = 95 - (angle_std - 1.5) * 10    # 95→80 线性
        elif angle_std < 5.0:
            stability = 80 - (angle_std - 3.0) * 7.5   # 80→65 线性
        else:
            stability = max(40, 65 - (angle_std - 5.0) * 5)

        return float(stability)

    def _calculate_head_stability(self, keypoints_sequence: List[Dict]) -> float:
        """
        计算头部稳定性（重写版）

        测量：融合多个头部关键点后，相对于肩部中点的垂直距离变异
        使用去趋势 + 归一化，降低检测抖动导致的系统性低分
        CV 越小 → 稳定性越高
        """
        relative_distances = []
        valid_weight_ratios = []
        vis_threshold = 0.5

        for kp in keypoints_sequence:
            ls, rs = kp['landmarks'][11], kp['landmarks'][12]

            if not (ls['visibility'] > vis_threshold and rs['visibility'] > vis_threshold):
                continue

            shoulder_y = (ls['y_norm'] + rs['y_norm']) / 2
            shoulder_w = abs(ls['x_norm'] - rs['x_norm'])
            scale = max(shoulder_w, 0.03)

            # 多头部关键点融合：nose + ear + eye
            weighted_sum = 0.0
            weighted_y = 0.0
            base_points = [
                (0, 0.35),   # nose
                (7, 0.20),   # left ear
                (8, 0.20),   # right ear
                (2, 0.125),  # left eye
                (5, 0.125),  # right eye
            ]
            for idx, base_weight in base_points:
                p = kp['landmarks'][idx]
                if p['visibility'] > 0.35:
                    w = base_weight * float(p['visibility'])
                    weighted_sum += w
                    weighted_y += p['y_norm'] * w

            if weighted_sum <= 0:
                continue

            head_y = weighted_y / weighted_sum
            relative_dist = (shoulder_y - head_y) / scale
            relative_distances.append(relative_dist)
            valid_weight_ratios.append(min(1.0, weighted_sum))

        if len(relative_distances) < 10:
            return 50.0

        relative_distances = np.array(relative_distances)
        baseline_mean = float(np.mean(np.abs(relative_distances)))

        # 去趋势：减少镜头或身体整体位移导致的慢漂移
        if len(relative_distances) >= 12:
            x = np.arange(len(relative_distances))
            coeff = np.polyfit(x, relative_distances, 1)
            relative_distances = relative_distances - np.polyval(coeff, x)

        # 轻度平滑（仅去噪，不压制振幅）
        if len(relative_distances) > 5:
            window = min(5, len(relative_distances) // 2 * 2 + 1)
            if window >= 3:
                relative_distances = savgol_filter(relative_distances, window, 1)

        std_dist = np.std(relative_distances)

        # 使用变异系数(CV)消除距离量级差异
        if baseline_mean > 1e-4:
            cv = (std_dist / baseline_mean) * 100  # 变异系数百分比
        else:
            cv = std_dist * 200  # 降级：绝对值×放大系数

        # 放宽阈值: <6%→95, 6-10%→95~80, 10-16%→80~65, >16%→<65
        if cv < 6:
            stability = 95
        elif cv < 10:
            stability = 95 - (cv - 6) * 3.75    # 95→80 线性
        elif cv < 16:
            stability = 80 - (cv - 10) * 3.75   # 80→65 线性
        else:
            stability = max(45, 65 - (cv - 16) * 2.5)

        visibility_factor = float(np.mean(valid_weight_ratios)) if valid_weight_ratios else 0.0
        stability *= (0.8 + 0.2 * visibility_factor)

        return float(np.clip(stability, 0, 100))

    def _calculate_gait_symmetry(self, keypoints_sequence: List[Dict]) -> float:
        """计算步态对称性"""
        left_knee_angles = []
        right_knee_angles = []

        for kp in keypoints_sequence:
            left_angle = self._calculate_joint_angle_safe(
                kp['landmarks'][23], kp['landmarks'][25], kp['landmarks'][27]
            )
            right_angle = self._calculate_joint_angle_safe(
                kp['landmarks'][24], kp['landmarks'][26], kp['landmarks'][28]
            )

            if not np.isnan(left_angle):
                left_knee_angles.append(left_angle)
            if not np.isnan(right_angle):
                right_knee_angles.append(right_angle)

        if len(left_knee_angles) < 10 or len(right_knee_angles) < 10:
            return 50.0

        min_len = min(len(left_knee_angles), len(right_knee_angles))
        correlation = np.corrcoef(left_knee_angles[:min_len], right_knee_angles[:min_len])[0, 1]

        if np.isnan(correlation):
            return 50.0

        symmetry = abs(correlation) * 100
        return max(0, min(symmetry, 100))

    def _calculate_body_lean_2d_frame(self, landmarks: List[Dict]) -> Optional[float]:
        """计算单帧2D躯干前倾角。"""
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        r_hip = landmarks[24]

        if (
            l_shoulder['visibility'] > 0.5 and r_shoulder['visibility'] > 0.5 and
            l_hip['visibility'] > 0.5 and r_hip['visibility'] > 0.5
        ):
            shoulder_x = (l_shoulder['x_norm'] + r_shoulder['x_norm']) / 2
            shoulder_y = (l_shoulder['y_norm'] + r_shoulder['y_norm']) / 2
            hip_x = (l_hip['x_norm'] + r_hip['x_norm']) / 2
            hip_y = (l_hip['y_norm'] + r_hip['y_norm']) / 2
        elif l_shoulder['visibility'] > 0.5 and l_hip['visibility'] > 0.5:
            shoulder_x, shoulder_y = l_shoulder['x_norm'], l_shoulder['y_norm']
            hip_x, hip_y = l_hip['x_norm'], l_hip['y_norm']
        elif r_shoulder['visibility'] > 0.5 and r_hip['visibility'] > 0.5:
            shoulder_x, shoulder_y = r_shoulder['x_norm'], r_shoulder['y_norm']
            hip_x, hip_y = r_hip['x_norm'], r_hip['y_norm']
        else:
            return None

        dx = shoulder_x - hip_x
        dy = shoulder_y - hip_y
        lean_angle = np.degrees(np.arctan2(dx, -dy))
        if self.view_angle == 'side':
            lean_angle = abs(lean_angle)
        return float(lean_angle) if np.isfinite(lean_angle) else None

    def _calculate_body_lean_3d_frame(self, pose_3d: Dict) -> Optional[float]:
        """计算单帧3D躯干前倾角。"""
        l_shoulder = get_3d_point_from_pose(pose_3d, 'l_shoulder')
        r_shoulder = get_3d_point_from_pose(pose_3d, 'r_shoulder')
        l_hip = get_3d_point_from_pose(pose_3d, 'l_hip')
        r_hip = get_3d_point_from_pose(pose_3d, 'r_hip')

        if not all(p is not None for p in [l_shoulder, r_shoulder, l_hip, r_hip]):
            return None

        shoulder_center = (l_shoulder + r_shoulder) / 2
        hip_center = (l_hip + r_hip) / 2

        if self.view_angle == 'side':
            lean_angle = self._calculate_side_lean_3d_rotated(
                l_shoulder, r_shoulder, l_hip, r_hip
            )
        else:
            trunk_vector = shoulder_center - hip_center
            vertical = np.array([0, -1, 0])
            cos_angle = np.dot(trunk_vector, vertical) / (np.linalg.norm(trunk_vector) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            lean_angle = np.degrees(np.arccos(cos_angle))

        return float(lean_angle) if np.isfinite(lean_angle) else None

    def _collect_body_lean_samples(
        self,
        keypoints_sequence: List[Dict],
        support_only: bool,
        fps: float,
    ) -> Tuple[List[float], List[str], List[float], int]:
        """收集稳健前倾角样本，优先保留支撑相关帧。"""
        n = len(keypoints_sequence)
        support_mask = np.ones(n, dtype=bool)

        if support_only and self.view_angle == 'side' and fps > 0 and n >= 10:
            phases = self._detect_gait_phases(keypoints_sequence, fps)
            if phases and len(phases) == n:
                support_mask = np.array([phase != self.PHASE_FLIGHT for phase in phases], dtype=bool)
                if int(np.sum(support_mask)) < 5:
                    support_mask = np.ones(n, dtype=bool)

        frame_indices = getattr(self, 'valid_frame_indices', []) or list(range(n))
        lean_angles: List[float] = []
        source_labels: List[str] = []
        consistency_diffs: List[float] = []
        selected_support_count = 0

        for local_idx, kp in enumerate(keypoints_sequence):
            if local_idx >= len(support_mask) or not support_mask[local_idx]:
                continue

            lean_2d = self._calculate_body_lean_2d_frame(kp['landmarks'])
            lean_3d = None

            if self.has_3d_data and self.keypoints_3d is not None:
                original_idx = frame_indices[local_idx] if local_idx < len(frame_indices) else local_idx
                if 0 <= original_idx < len(self.keypoints_3d):
                    kp_3d = self.keypoints_3d[original_idx]
                    if kp_3d.get('has_3d', False):
                        lean_3d = self._calculate_body_lean_3d_frame(kp_3d.get('pose_3d', {}))

            selected_angle = None
            selected_source = '2D'
            if lean_2d is not None and lean_3d is not None:
                diff = abs(lean_3d - lean_2d)
                consistency_diffs.append(float(diff))
                if diff <= 4.0:
                    selected_angle = 0.7 * lean_3d + 0.3 * lean_2d
                    selected_source = '3D'
                elif diff <= 8.0:
                    selected_angle = 0.55 * lean_3d + 0.45 * lean_2d
                    selected_source = '3D'
                else:
                    selected_angle = lean_2d
                    selected_source = '2D'
            elif lean_3d is not None:
                selected_angle = lean_3d
                selected_source = '3D'
            elif lean_2d is not None:
                selected_angle = lean_2d
                selected_source = '2D'

            if selected_angle is None or not np.isfinite(selected_angle):
                continue

            lean_angles.append(float(selected_angle))
            source_labels.append(selected_source)
            selected_support_count += 1

        return lean_angles, source_labels, consistency_diffs, selected_support_count

    def _calculate_body_lean(self, keypoints_sequence: List[Dict], fps: float = 0.0) -> Dict:
        """
        计算身体前倾角度（稳健版：3D优先 + 支撑相关帧 + 2D一致性约束）

        核心改进：
        1. 优先使用3D数据，但必须通过2D一致性检查
        2. 侧面视角优先使用支撑相关帧，减少摆动相对结果的污染
        3. 使用截尾均值而不是全帧直接平均，降低异常帧影响
        """
        lean_angles, source_labels, consistency_diffs, support_count = self._collect_body_lean_samples(
            keypoints_sequence,
            support_only=True,
            fps=fps,
        )
        if len(lean_angles) < 5:
            lean_angles, source_labels, consistency_diffs, support_count = self._collect_body_lean_samples(
                keypoints_sequence,
                support_only=False,
                fps=fps,
            )

        lean_angles = [float(a) for a in lean_angles if np.isfinite(a)]
        if self.view_angle == 'side':
            lean_angles = [a for a in lean_angles if 0 <= a <= 35]
        else:
            lean_angles = [a for a in lean_angles if 0 <= a <= 60]

        if not lean_angles:
            return {
                'mean_lean': 0.0,
                'std_lean': 0.0,
                'forward_lean': 0.0,
                'rating': {},
                'data_source': 'none',
                'debug_info': {'support_frame_count': 0, 'consistency_mean_abs_diff': None},
            }

        lean_array = np.array(lean_angles, dtype=float)
        if len(lean_array) >= 7:
            lean_array = self._smooth_signal_advanced(lean_array)

        if len(lean_array) >= 8:
            lower, upper = np.percentile(lean_array, [15, 85])
            trimmed = lean_array[(lean_array >= lower) & (lean_array <= upper)]
            if len(trimmed) >= 5:
                lean_array = trimmed

        mean_forward = float(np.mean(lean_array)) if len(lean_array) else 0.0
        std_forward = float(np.std(lean_array)) if len(lean_array) else 0.0
        source_3d_count = sum(1 for label in source_labels if label == '3D')
        data_source = '3D' if source_3d_count >= max(5, int(len(source_labels) * 0.5)) else '2D'

        return {
            'mean_lean': mean_forward,
            'std_lean': std_forward,
            'forward_lean': float(mean_forward),
            'rating': self._rate_body_lean(mean_forward),
            'data_source': data_source,
            'debug_info': {
                'support_frame_count': int(support_count),
                'selected_sample_count': int(len(lean_array)),
                'consistency_mean_abs_diff': (
                    float(np.mean(consistency_diffs)) if consistency_diffs else None
                ),
                'source_3d_ratio': float(source_3d_count / max(len(source_labels), 1)),
            },
        }

    def _safe_normalize_vector(self, vec: np.ndarray) -> Optional[np.ndarray]:
        """安全归一化向量"""
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return None
        return vec / norm

    def _calculate_side_lean_3d_rotated(self,
                                        l_shoulder: np.ndarray,
                                        r_shoulder: np.ndarray,
                                        l_hip: np.ndarray,
                                        r_hip: np.ndarray) -> float:
        """
        3D侧面前倾角（先对齐身体坐标系）

        使用髋部连线定义左右轴，构建身体局部矢状面，
        再计算躯干在该平面上的前倾角，减少机位偏差影响。
        """
        shoulder_center = (l_shoulder + r_shoulder) / 2.0
        hip_center = (l_hip + r_hip) / 2.0
        trunk_vec = shoulder_center - hip_center

        trunk_u = self._safe_normalize_vector(trunk_vec)
        if trunk_u is None:
            return 0.0

        lateral_vec = r_hip - l_hip
        lateral_u = self._safe_normalize_vector(lateral_vec)
        if lateral_u is None:
            return 0.0

        global_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        up_u = self._safe_normalize_vector(global_up)
        if up_u is None:
            return 0.0

        # 前向轴 = 左右轴 × 垂直轴；在退化情况下回退到与躯干叉乘
        forward_vec = np.cross(lateral_u, up_u)
        if np.linalg.norm(forward_vec) < 1e-6:
            forward_vec = np.cross(lateral_u, trunk_u)
        forward_u = self._safe_normalize_vector(forward_vec)
        if forward_u is None:
            return 0.0

        forward_comp = abs(float(np.dot(trunk_u, forward_u)))
        vertical_comp = abs(float(np.dot(trunk_u, up_u)))

        lean = np.degrees(np.arctan2(forward_comp, vertical_comp + 1e-6))
        return float(np.clip(lean, 0.0, 90.0))

    def _rate_body_lean(self, forward_lean: float) -> Dict:
        """评估身体前倾（中长跑/马拉松稳健阈值）。"""
        optimal_min = float(BODY_LEAN_THRESHOLDS['optimal_min'])
        optimal_max = float(BODY_LEAN_THRESHOLDS['optimal_max'])
        good_min = float(BODY_LEAN_THRESHOLDS['good_min'])
        good_max = float(BODY_LEAN_THRESHOLDS['good_max'])
        fair_min = float(BODY_LEAN_THRESHOLDS['fair_min'])
        fair_max = float(BODY_LEAN_THRESHOLDS['fair_max'])

        if optimal_min <= forward_lean <= optimal_max:
            return {'level': 'optimal', 'score': 100, 'description': '前倾角度理想，符合耐力跑经济性要求'}
        if good_min <= forward_lean <= good_max:
            return {'level': 'good', 'score': 85, 'description': '前倾角度较好，整体较为自然'}
        if fair_min <= forward_lean <= fair_max:
            return {'level': 'acceptable', 'score': 70, 'description': '前倾角度一般，存在改进空间'}
        if forward_lean < fair_min:
            return {'level': 'upright', 'score': 55, 'description': '躯干过于直立，前移不足'}
        return {'level': 'excessive', 'score': 55, 'description': '前倾过大，可能增加额外负担'}

    def _calculate_arm_swing(self, keypoints_sequence: List[Dict]) -> Dict:
        """
        计算手臂摆动幅度（2D主导、3D辅助策略）

        策略说明：
        1. 2D数据为主（可靠观测）：侧面视角2D检测手臂摆动正常
        2. 3D侧面手臂不可靠：远侧手臂被遮挡，3D提升结果不准
        3. 使用visibility过滤低置信度帧
        4. 当左右差异过大且3D置信度低时，标记远侧手臂检测可能不准确
        """
        left_elbow_angles = []
        right_elbow_angles = []
        high_vis_frame_count = 0  # 高可见度帧计数

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # 检查左手臂可见度（肩11、肘13、腕15）
            left_vis = min(landmarks[11]['visibility'],
                          landmarks[13]['visibility'],
                          landmarks[15]['visibility'])
            # 检查右手臂可见度（肩12、肘14、腕16）
            right_vis = min(landmarks[12]['visibility'],
                           landmarks[14]['visibility'],
                           landmarks[16]['visibility'])

            # 统计高可见度帧
            if left_vis > 0.7 and right_vis > 0.7:
                high_vis_frame_count += 1

            # 只使用可见度 > 0.5 的数据
            if left_vis > 0.5:
                left_angle = self._calculate_joint_angle_safe(
                    landmarks[11], landmarks[13], landmarks[15]
                )
                if not np.isnan(left_angle):
                    left_elbow_angles.append(left_angle)

            if right_vis > 0.5:
                right_angle = self._calculate_joint_angle_safe(
                    landmarks[12], landmarks[14], landmarks[16]
                )
                if not np.isnan(right_angle):
                    right_elbow_angles.append(right_angle)

        if not left_elbow_angles or not right_elbow_angles:
            return {
                'arm_swing_amplitude': 0.0,
                'rating': {},
                'data_source': '2D_primary',
                'warning': '手臂数据不足'
            }

        # 计算摆动范围
        left_range = np.max(left_elbow_angles) - np.min(left_elbow_angles)
        right_range = np.max(right_elbow_angles) - np.min(right_elbow_angles)
        avg_amplitude = (left_range + right_range) / 2

        # 计算3D置信度（基于高可见度帧比例）
        arm_3d_confidence = high_vis_frame_count / len(keypoints_sequence) if keypoints_sequence else 0

        return {
            'arm_swing_amplitude': float(avg_amplitude),
            'left_arm_range': float(left_range),
            'right_arm_range': float(right_range),
            'data_source': '2D_primary',
            'arm_3d_confidence': float(arm_3d_confidence),
            'rating': self._rate_arm_swing(avg_amplitude)
        }

    def _rate_arm_swing(self, amplitude: float) -> Dict:
        """评估手臂摆动"""
        if 30 <= amplitude <= 60:
            return {'level': 'optimal', 'score': 100, 'description': '手臂摆动幅度适中'}
        elif 20 <= amplitude < 30 or 60 < amplitude <= 80:
            return {'level': 'good', 'score': 80, 'description': '手臂摆动良好'}
        elif amplitude < 20:
            return {'level': 'restricted', 'score': 60, 'description': '手臂摆动受限'}
        else:
            return {'level': 'excessive', 'score': 60, 'description': '手臂摆动过大'}

    def _smooth_signal_advanced(self, signal: np.ndarray) -> np.ndarray:
        """高级信号平滑"""
        if len(signal) < 5:
            return signal

        from scipy.ndimage import median_filter
        signal = median_filter(signal, size=3)

        if len(signal) > 6:
            b, a = butter(2, 0.2, btype='low')
            signal = filtfilt(b, a, signal)

        if len(signal) > self.smooth_window:
            window = min(self.smooth_window, len(signal))
            if window % 2 == 0:
                window -= 1
            signal = savgol_filter(signal, window_length=window, polyorder=2)

        return signal

    def _filter_outliers_iqr(self, data: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """
        使用IQR方法过滤异常值

        Args:
            data: 输入数据数组
            factor: IQR倍数，默认1.5

        Returns:
            过滤后的数据数组
        """
        if len(data) < 4:
            return data
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return data
        lower = q25 - factor * iqr
        upper = q75 + factor * iqr
        mask = (data >= lower) & (data <= upper)
        return data[mask] if np.sum(mask) >= 3 else data

    def _interpolate_nans(self, arr: np.ndarray) -> np.ndarray:
        """插值处理NaN值"""
        if not np.any(np.isnan(arr)):
            return arr

        valid_idx = ~np.isnan(arr)
        if np.sum(valid_idx) < 2:
            return np.zeros_like(arr)

        interp_func = interp1d(
            np.where(valid_idx)[0],
            arr[valid_idx],
            kind='linear',
            fill_value='extrapolate'
        )
        return interp_func(np.arange(len(arr)))

    def _get_empty_analysis(self) -> Dict:
        """返回空分析结果"""
        return {
            'fps': 0,
            'total_frames': 0,
            'valid_frames': 0,
            'view_angle': 'unknown',
            'trunk_reference': 0,
            'angles': {},
            'vertical_motion': self._get_empty_vertical_motion(),
            'cadence': {},
            'stride_info': {},
            'stability': {},
            'body_lean': {},
            'arm_swing': {},
            'gait_cycle': {},
        }


# 模块测试
if __name__ == "__main__":
    print("=" * 60)
    print("测试重构版运动学分析模块")
    print("=" * 60)

    # 生成模拟数据
    mock_keypoints = []
    fps = 30
    duration = 3
    num_frames = fps * duration

    for i in range(num_frames):
        kp = {'detected': True, 'landmarks': []}
        t = i / fps
        phase = t * 2 * np.pi * 2

        for j in range(33):
            if j in [25, 26, 27, 28]:
                y_offset = np.sin(phase + (j % 2) * np.pi) * 0.1
            else:
                y_offset = 0

            kp['landmarks'].append({
                'id': j,
                'name': f'point_{j}',
                'x': 320,
                'y': 240 + y_offset * 100,
                'x_norm': 0.5,
                'y_norm': 0.5 + y_offset,
                'visibility': 0.9
            })
        mock_keypoints.append(kp)

    analyzer = KinematicAnalyzer()
    results = analyzer.analyze_sequence(mock_keypoints, fps, view_angle='side')

    print(f"\n步频: {results['cadence']['cadence']:.1f} 步/分")
    print(f"垂直振幅(归一化): {results['vertical_motion']['amplitude_normalized']:.2f}%")
    print(f"振幅评级: {results['vertical_motion']['amplitude_rating']}")
    print(f"\n膝关节分阶段分析:")
    phase_analysis = results['angles']['phase_analysis']
    print(f"  触地期平均角度: {phase_analysis['ground_contact']['mean']:.1f}°")
    print(f"  腾空期平均角度: {phase_analysis['flight']['mean']:.1f}°")
    print(f"  最大弯曲角度: {phase_analysis['max_flexion']:.1f}°")
    print(f"  关节活动范围: {phase_analysis['range_of_motion']:.1f}°")

    print("\n✅ 模块测试完成!")
