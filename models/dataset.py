"""
跑步动作合成数据集

特点：
1. 基于生物力学模型生成更真实的关键点序列
2. 支持多视角数据生成（侧面、正面、背面、混合）
3. 模拟真实跑步参数变化
4. 支持数据增强

适用于毕业设计：基于深度学习的跑步动作视频解析与技术质量评价系统
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class RunningParameters:
    """跑步参数配置"""
    cadence: float  # 步频 (步/分)
    stride_length: float  # 步长 (相对值)
    vertical_oscillation: float  # 垂直振幅 (相对躯干长度)
    trunk_lean: float  # 躯干前倾角度 (度)
    arm_swing: float  # 手臂摆动幅度
    ground_contact_time: float  # 触地时间比例
    asymmetry: float  # 左右不对称程度 (0=完全对称)
    noise_level: float  # 噪声水平


class BiomechanicalModel:
    """
    生物力学模型
    基于跑步运动学原理生成关键点运动
    """

    # MediaPipe关键点索引
    KEYPOINT_INDICES = {
        'nose': 0,
        'left_eye': 1, 'right_eye': 2,
        'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32
    }

    def __init__(self, view: str = 'side'):
        """
        Args:
            view: 视角类型 ('side', 'front', 'back')
        """
        self.view = view

        # 人体比例参数
        self.body_proportions = {
            'head_height': 0.12,
            'neck_height': 0.03,
            'torso_height': 0.30,
            'upper_leg': 0.23,
            'lower_leg': 0.22,
            'upper_arm': 0.15,
            'lower_arm': 0.13,
            'shoulder_width': 0.22,
            'hip_width': 0.18
        }

    def generate_keypoints(self, t: float, params: RunningParameters,
                           is_left_leg_forward: bool) -> np.ndarray:
        """
        生成单帧关键点

        Args:
            t: 时间点 (0-2π 为一个完整步态周期)
            params: 跑步参数
            is_left_leg_forward: 左腿是否在前

        Returns:
            关键点数组 (33, 2) - x, y 坐标
        """
        keypoints = np.zeros((33, 2))

        # 基础位置（身体中心）
        center_x = 0.5
        center_y = 0.5

        # 步态相位
        phase = t % (2 * np.pi)

        # 根据视角生成关键点
        if self.view == 'side':
            keypoints = self._generate_side_view(keypoints, center_x, center_y,
                                                  phase, params, is_left_leg_forward)
        elif self.view == 'front':
            keypoints = self._generate_front_view(keypoints, center_x, center_y,
                                                   phase, params, is_left_leg_forward)
        elif self.view == 'back':
            keypoints = self._generate_back_view(keypoints, center_x, center_y,
                                                  phase, params, is_left_leg_forward)

        # 添加噪声
        noise = np.random.randn(33, 2) * params.noise_level
        keypoints += noise

        # 限制在有效范围内
        keypoints = np.clip(keypoints, 0.01, 0.99)

        return keypoints

    def _generate_side_view(self, keypoints: np.ndarray, cx: float, cy: float,
                            phase: float, params: RunningParameters,
                            is_left_forward: bool) -> np.ndarray:
        """生成侧面视角关键点"""
        # 躯干前倾
        lean_rad = np.radians(params.trunk_lean)

        # 垂直振幅（基于步态相位）
        vertical_offset = np.sin(phase * 2) * params.vertical_oscillation

        # 髋部位置
        hip_y = cy + 0.05 + vertical_offset
        keypoints[23] = [cx - 0.02, hip_y]  # left_hip
        keypoints[24] = [cx + 0.02, hip_y]  # right_hip

        # 肩部位置（考虑前倾）
        shoulder_offset = np.sin(lean_rad) * self.body_proportions['torso_height']
        shoulder_y = hip_y - self.body_proportions['torso_height'] * np.cos(lean_rad)
        keypoints[11] = [cx - 0.03 + shoulder_offset, shoulder_y]  # left_shoulder
        keypoints[12] = [cx + 0.03 + shoulder_offset, shoulder_y]  # right_shoulder

        # 头部
        head_y = shoulder_y - self.body_proportions['head_height']
        keypoints[0] = [cx + shoulder_offset, head_y]  # nose

        # 腿部运动
        leg_phase_left = phase if is_left_forward else phase + np.pi
        leg_phase_right = phase + np.pi if is_left_forward else phase

        # 左腿
        knee_angle_left, ankle_pos_left = self._calculate_leg_position(
            leg_phase_left, params, hip_y
        )
        keypoints[25] = [cx + ankle_pos_left[0] * 0.5, hip_y + self.body_proportions['upper_leg'] * 0.8]
        keypoints[27] = ankle_pos_left

        # 右腿
        knee_angle_right, ankle_pos_right = self._calculate_leg_position(
            leg_phase_right, params, hip_y
        )
        keypoints[26] = [cx + ankle_pos_right[0] * 0.5, hip_y + self.body_proportions['upper_leg'] * 0.8]
        keypoints[28] = ankle_pos_right

        # 手臂摆动（与腿部反向）
        arm_phase_left = leg_phase_right  # 左臂与右腿同步
        arm_phase_right = leg_phase_left

        # 左臂
        elbow_x_left = cx - 0.05 + np.sin(arm_phase_left) * params.arm_swing * 0.5
        elbow_y_left = shoulder_y + 0.1
        keypoints[13] = [elbow_x_left, elbow_y_left]
        keypoints[15] = [elbow_x_left + np.sin(arm_phase_left) * params.arm_swing * 0.3, elbow_y_left + 0.08]

        # 右臂
        elbow_x_right = cx + 0.05 + np.sin(arm_phase_right) * params.arm_swing * 0.5
        elbow_y_right = shoulder_y + 0.1
        keypoints[14] = [elbow_x_right, elbow_y_right]
        keypoints[16] = [elbow_x_right + np.sin(arm_phase_right) * params.arm_swing * 0.3, elbow_y_right + 0.08]

        # 填充其他关键点（眼睛、耳朵等）
        keypoints = self._fill_auxiliary_keypoints(keypoints, self.view)

        return keypoints

    def _generate_front_view(self, keypoints: np.ndarray, cx: float, cy: float,
                             phase: float, params: RunningParameters,
                             is_left_forward: bool) -> np.ndarray:
        """生成正面视角关键点"""
        # 垂直振幅
        vertical_offset = np.sin(phase * 2) * params.vertical_oscillation

        # 髋部（左右分开可见）
        hip_width = self.body_proportions['hip_width']
        hip_y = cy + 0.05 + vertical_offset
        keypoints[23] = [cx - hip_width / 2, hip_y]
        keypoints[24] = [cx + hip_width / 2, hip_y]

        # 肩部
        shoulder_width = self.body_proportions['shoulder_width']
        shoulder_y = hip_y - self.body_proportions['torso_height']

        # 添加不对称性
        asym = params.asymmetry * np.sin(phase)
        keypoints[11] = [cx - shoulder_width / 2 + asym * 0.02, shoulder_y]
        keypoints[12] = [cx + shoulder_width / 2 + asym * 0.02, shoulder_y]

        # 头部
        keypoints[0] = [cx, shoulder_y - self.body_proportions['head_height']]

        # 腿部（前后深度变化体现在y坐标的微小变化和膝盖位置）
        leg_phase_left = phase if is_left_forward else phase + np.pi
        leg_phase_right = phase + np.pi if is_left_forward else phase

        # 左腿
        knee_depth_left = np.sin(leg_phase_left) * 0.02
        keypoints[25] = [cx - hip_width / 2, hip_y + self.body_proportions['upper_leg'] + knee_depth_left]
        keypoints[27] = [cx - hip_width / 2 * 0.9, hip_y + self.body_proportions['upper_leg'] + self.body_proportions['lower_leg']]

        # 右腿
        knee_depth_right = np.sin(leg_phase_right) * 0.02
        keypoints[26] = [cx + hip_width / 2, hip_y + self.body_proportions['upper_leg'] + knee_depth_right]
        keypoints[28] = [cx + hip_width / 2 * 0.9, hip_y + self.body_proportions['upper_leg'] + self.body_proportions['lower_leg']]

        # 手臂
        arm_phase_left = leg_phase_right
        arm_phase_right = leg_phase_left

        keypoints[13] = [cx - shoulder_width / 2 - 0.03, shoulder_y + 0.12]
        keypoints[15] = [cx - shoulder_width / 2 - 0.05, shoulder_y + 0.22]
        keypoints[14] = [cx + shoulder_width / 2 + 0.03, shoulder_y + 0.12]
        keypoints[16] = [cx + shoulder_width / 2 + 0.05, shoulder_y + 0.22]

        keypoints = self._fill_auxiliary_keypoints(keypoints, self.view)

        return keypoints

    def _generate_back_view(self, keypoints: np.ndarray, cx: float, cy: float,
                            phase: float, params: RunningParameters,
                            is_left_forward: bool) -> np.ndarray:
        """生成背面视角关键点（与正面类似但镜像）"""
        keypoints = self._generate_front_view(keypoints, cx, cy, phase, params, is_left_forward)
        # 背面看不到鼻子，调整相关关键点
        keypoints[0] = [cx, cy - 0.35]  # 头部后方
        return keypoints

    def _calculate_leg_position(self, phase: float, params: RunningParameters,
                                hip_y: float) -> Tuple[float, np.ndarray]:
        """
        计算腿部位置

        Returns:
            knee_angle: 膝关节角度
            ankle_position: 踝关节位置 [x, y]
        """
        # 步态周期划分
        # 0 - π/3: 触地期
        # π/3 - 2π/3: 蹬伸期
        # 2π/3 - π: 腾空前期
        # π - 4π/3: 腾空后期
        # 4π/3 - 5π/3: 落地准备
        # 5π/3 - 2π: 着地期

        normalized_phase = phase % (2 * np.pi)

        if normalized_phase < np.pi / 3:  # 触地期
            x_offset = -0.05 + normalized_phase / (np.pi / 3) * 0.1
            y_offset = self.body_proportions['upper_leg'] + self.body_proportions['lower_leg']
            knee_angle = 160 - normalized_phase / (np.pi / 3) * 20  # 160° -> 140°

        elif normalized_phase < 2 * np.pi / 3:  # 蹬伸期
            progress = (normalized_phase - np.pi / 3) / (np.pi / 3)
            x_offset = 0.05 + progress * 0.1
            y_offset = self.body_proportions['upper_leg'] + self.body_proportions['lower_leg'] - progress * 0.05
            knee_angle = 140 + progress * 30  # 140° -> 170°

        elif normalized_phase < np.pi:  # 腾空前期
            progress = (normalized_phase - 2 * np.pi / 3) / (np.pi / 3)
            x_offset = 0.15 - progress * 0.2
            y_offset = self.body_proportions['upper_leg'] + self.body_proportions['lower_leg'] * 0.7 - progress * 0.1
            knee_angle = 170 - progress * 70  # 170° -> 100°

        elif normalized_phase < 4 * np.pi / 3:  # 腾空后期
            progress = (normalized_phase - np.pi) / (np.pi / 3)
            x_offset = -0.05 - progress * 0.1
            y_offset = self.body_proportions['upper_leg'] * 0.6 + progress * 0.15
            knee_angle = 100 + progress * 20  # 100° -> 120°

        elif normalized_phase < 5 * np.pi / 3:  # 落地准备
            progress = (normalized_phase - 4 * np.pi / 3) / (np.pi / 3)
            x_offset = -0.15 + progress * 0.1
            y_offset = self.body_proportions['upper_leg'] * 0.75 + self.body_proportions['lower_leg'] * 0.5 + progress * 0.15
            knee_angle = 120 + progress * 30  # 120° -> 150°

        else:  # 着地期
            progress = (normalized_phase - 5 * np.pi / 3) / (np.pi / 3)
            x_offset = -0.05
            y_offset = self.body_proportions['upper_leg'] + self.body_proportions['lower_leg']
            knee_angle = 150 + progress * 10  # 150° -> 160°

        ankle_position = np.array([0.5 + x_offset * params.stride_length,
                                   hip_y + y_offset])

        return knee_angle, ankle_position

    def _fill_auxiliary_keypoints(self, keypoints: np.ndarray, view: str) -> np.ndarray:
        """填充辅助关键点（眼睛、耳朵等）"""
        nose = keypoints[0]

        if view == 'side':
            # 侧面只能看到一侧
            keypoints[1] = nose + np.array([0.02, 0.02])  # left_eye
            keypoints[2] = nose + np.array([0.02, 0.02])  # right_eye (重叠)
            keypoints[3] = nose + np.array([0.04, 0.01])  # left_ear
            keypoints[4] = nose + np.array([-0.02, 0.01])  # right_ear (不可见)
        elif view in ['front', 'back']:
            keypoints[1] = nose + np.array([-0.03, 0.02])
            keypoints[2] = nose + np.array([0.03, 0.02])
            keypoints[3] = nose + np.array([-0.06, 0.01])
            keypoints[4] = nose + np.array([0.06, 0.01])

        # 填充脚部关键点
        for ankle_idx, heel_idx, toe_idx in [(27, 29, 31), (28, 30, 32)]:
            ankle = keypoints[ankle_idx]
            keypoints[heel_idx] = ankle + np.array([-0.02, 0.02])
            keypoints[toe_idx] = ankle + np.array([0.03, 0.01])

        return keypoints

    def determine_phase(self, t: float) -> int:
        """
        确定步态阶段

        Returns:
            0: 触地期
            1: 腾空期
            2: 过渡期
        """
        phase = t % (2 * np.pi)

        if phase < np.pi / 3 or phase > 5 * np.pi / 3:
            return 0  # 触地
        elif 2 * np.pi / 3 < phase < 4 * np.pi / 3:
            return 1  # 腾空
        else:
            return 2  # 过渡


class RunningDataset(Dataset):
    """
    跑步动作数据集

    生成多视角、多参数的合成跑步数据
    """

    def __init__(self,
                 num_samples: int = 1000,
                 sequence_length: int = 30,
                 views: List[str] = None,
                 augment: bool = True):
        """
        Args:
            num_samples: 样本数量
            sequence_length: 序列长度
            views: 视角列表，默认 ['side', 'front', 'back']
            augment: 是否数据增强
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.views = views or ['side', 'front', 'back']
        self.augment = augment

        # 视角ID映射
        self.view_to_id = {'side': 0, 'front': 1, 'back': 2, 'mixed': 3}

        print(f"生成 {num_samples} 个合成跑步样本...")
        self.data = self._generate_dataset()
        print("✅ 数据集生成完成")

    def _generate_dataset(self) -> List[Dict]:
        """生成完整数据集"""
        dataset = []

        for i in range(self.num_samples):
            # 随机选择视角
            view = random.choice(self.views)

            # 随机生成跑步参数
            params = self._generate_random_params()

            # 生成序列
            sample = self._generate_sample(view, params)
            dataset.append(sample)

        return dataset

    def _generate_random_params(self) -> RunningParameters:
        """
        生成随机跑步参数（优化版：增加多样性）

        跑者类型分布：
        - 精英跑者 (20%): 高步频、短触地、低振幅
        - 优秀跑者 (25%): 较高步频、较短触地
        - 良好跑者 (25%): 中等参数
        - 一般跑者 (20%): 较低步频、较长触地
        - 较差跑者 (10%): 低步频、长触地、高振幅
        """
        # 随机选择跑者类型
        runner_type = np.random.choice(
            ['elite', 'excellent', 'good', 'fair', 'poor'],
            p=[0.20, 0.25, 0.25, 0.20, 0.10]
        )

        if runner_type == 'elite':
            # 精英跑者
            return RunningParameters(
                cadence=np.random.uniform(185, 210),
                stride_length=np.random.uniform(1.1, 1.4),
                vertical_oscillation=np.random.uniform(0.03, 0.06),
                trunk_lean=np.random.uniform(8, 15),
                arm_swing=np.random.uniform(0.15, 0.25),
                ground_contact_time=np.random.uniform(0.16, 0.22),
                asymmetry=np.random.uniform(0, 0.03),
                noise_level=np.random.uniform(0.003, 0.01)
            )
        elif runner_type == 'excellent':
            # 优秀跑者
            return RunningParameters(
                cadence=np.random.uniform(175, 190),
                stride_length=np.random.uniform(1.0, 1.3),
                vertical_oscillation=np.random.uniform(0.05, 0.08),
                trunk_lean=np.random.uniform(6, 16),
                arm_swing=np.random.uniform(0.12, 0.28),
                ground_contact_time=np.random.uniform(0.20, 0.26),
                asymmetry=np.random.uniform(0, 0.05),
                noise_level=np.random.uniform(0.005, 0.015)
            )
        elif runner_type == 'good':
            # 良好跑者
            return RunningParameters(
                cadence=np.random.uniform(165, 180),
                stride_length=np.random.uniform(0.9, 1.2),
                vertical_oscillation=np.random.uniform(0.06, 0.10),
                trunk_lean=np.random.uniform(5, 18),
                arm_swing=np.random.uniform(0.10, 0.30),
                ground_contact_time=np.random.uniform(0.24, 0.30),
                asymmetry=np.random.uniform(0, 0.08),
                noise_level=np.random.uniform(0.008, 0.018)
            )
        elif runner_type == 'fair':
            # 一般跑者
            return RunningParameters(
                cadence=np.random.uniform(155, 170),
                stride_length=np.random.uniform(0.8, 1.1),
                vertical_oscillation=np.random.uniform(0.08, 0.14),
                trunk_lean=np.random.uniform(3, 20),
                arm_swing=np.random.uniform(0.08, 0.32),
                ground_contact_time=np.random.uniform(0.27, 0.35),
                asymmetry=np.random.uniform(0.02, 0.12),
                noise_level=np.random.uniform(0.010, 0.022)
            )
        else:  # poor
            # 较差跑者
            return RunningParameters(
                cadence=np.random.uniform(140, 160),
                stride_length=np.random.uniform(0.7, 1.0),
                vertical_oscillation=np.random.uniform(0.12, 0.20),
                trunk_lean=np.random.uniform(0, 25),
                arm_swing=np.random.uniform(0.05, 0.35),
                ground_contact_time=np.random.uniform(0.30, 0.45),
                asymmetry=np.random.uniform(0.05, 0.18),
                noise_level=np.random.uniform(0.015, 0.030)
            )

    def _generate_sample(self, view: str, params: RunningParameters) -> Dict:
        """生成单个样本"""
        model = BiomechanicalModel(view)

        # 时间序列（覆盖约2个完整步态周期）
        freq = params.cadence / 60.0  # 步频转Hz
        duration = self.sequence_length / 30.0  # 假设30fps
        t_values = np.linspace(0, duration * freq * 2 * np.pi, self.sequence_length)

        # 生成关键点序列
        keypoints_sequence = []
        phase_labels = []
        is_left_forward = random.choice([True, False])

        for t in t_values:
            kp = model.generate_keypoints(t, params, is_left_forward)
            keypoints_sequence.append(kp.flatten())  # (66,)
            phase_labels.append(model.determine_phase(t))

        # 计算质量评分
        quality_scores = self._calculate_quality_scores(params)

        return {
            'keypoints': np.array(keypoints_sequence, dtype=np.float32),  # (seq_len, 66)
            'phase_labels': np.array(phase_labels, dtype=np.int64),  # (seq_len,)
            'quality_scores': np.array(quality_scores, dtype=np.float32),  # (5,)
            'view': view,
            'view_id': self.view_to_id[view],
            'params': params
        }

    def _calculate_quality_scores(self, params: RunningParameters) -> List[float]:
        """
        根据参数计算质量评分（优化版：更大区分度）

        Returns:
            [总分, 稳定性, 效率, 跑姿, 节奏]
        """
        # 1. 稳定性评分（基于垂直振幅和不对称性）
        # 精英：<6%, 优秀：6-8%, 良好：8-10%, 一般：10-14%, 较差：>14%
        if params.vertical_oscillation < 0.06:
            stability = 92 + np.random.uniform(-3, 5)
        elif params.vertical_oscillation < 0.08:
            stability = 82 + np.random.uniform(-4, 4)
        elif params.vertical_oscillation < 0.10:
            stability = 72 + np.random.uniform(-5, 5)
        elif params.vertical_oscillation < 0.14:
            stability = 58 + np.random.uniform(-6, 6)
        else:
            stability = 42 + np.random.uniform(-8, 5)

        # 不对称性惩罚
        stability -= params.asymmetry * 80

        # 2. 效率评分（基于步频和触地时间）
        # 步频评分：精英185+, 优秀175-185, 良好165-175, 一般155-165, 较差<155
        if params.cadence >= 185:
            cadence_score = 95
        elif params.cadence >= 175:
            cadence_score = 83
        elif params.cadence >= 165:
            cadence_score = 70
        elif params.cadence >= 155:
            cadence_score = 55
        else:
            cadence_score = 38

        # 触地时间评分：精英<0.22, 优秀0.22-0.26, 良好0.26-0.30, 一般0.30-0.35
        gc_time = params.ground_contact_time
        if gc_time < 0.22:
            gc_score = 95
        elif gc_time < 0.26:
            gc_score = 82
        elif gc_time < 0.30:
            gc_score = 68
        elif gc_time < 0.35:
            gc_score = 52
        else:
            gc_score = 35

        efficiency = (cadence_score * 0.6 + gc_score * 0.4) + np.random.uniform(-4, 4)

        # 3. 跑姿评分（躯干前倾和手臂摆动）
        # 最优前倾：8-15度
        lean_diff = abs(params.trunk_lean - 11)
        if lean_diff < 4:
            lean_score = 92
        elif lean_diff < 7:
            lean_score = 78
        elif lean_diff < 10:
            lean_score = 62
        else:
            lean_score = 45

        # 手臂摆动评分
        if 0.15 <= params.arm_swing <= 0.25:
            arm_score = 90
        elif 0.10 <= params.arm_swing <= 0.30:
            arm_score = 75
        else:
            arm_score = 55

        form = (lean_score * 0.7 + arm_score * 0.3) + np.random.uniform(-4, 4)

        # 4. 节奏评分（对称性和步长一致性）
        # 对称性越高越好
        rhythm = 95 - params.asymmetry * 300 + np.random.uniform(-5, 5)

        # 限制范围并增加变化
        stability = np.clip(stability, 25, 100)
        efficiency = np.clip(efficiency, 25, 100)
        form = np.clip(form, 25, 100)
        rhythm = np.clip(rhythm, 25, 100)

        # 总分（加权平均）
        total = (
            stability * 0.25 +
            efficiency * 0.30 +
            form * 0.25 +
            rhythm * 0.20
        )

        return [total, stability, efficiency, form, rhythm]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        获取样本

        Returns:
            keypoints: (seq_len, 66)
            phase_labels: (seq_len,)
            quality_scores: (5,)
            view_id: scalar
        """
        sample = self.data[idx]

        keypoints = torch.FloatTensor(sample['keypoints'])
        phase_labels = torch.LongTensor(sample['phase_labels'])
        quality_scores = torch.FloatTensor(sample['quality_scores'])
        view_id = torch.LongTensor([sample['view_id']])

        # 数据增强
        if self.augment:
            keypoints = self._augment(keypoints)

        # 数据标准化
        keypoints = self._normalize(keypoints)

        return keypoints, phase_labels, quality_scores, view_id.squeeze()

    def _augment(self, keypoints: torch.Tensor) -> torch.Tensor:
        """数据增强"""
        # 随机缩放
        if random.random() < 0.3:
            scale = 1.0 + random.uniform(-0.1, 0.1)
            keypoints = keypoints * scale

        # 随机平移
        if random.random() < 0.3:
            shift = random.uniform(-0.05, 0.05)
            keypoints = keypoints + shift

        # 随机噪声
        if random.random() < 0.3:
            noise = torch.randn_like(keypoints) * 0.01
            keypoints = keypoints + noise

        return keypoints

    def _normalize(self, keypoints: torch.Tensor) -> torch.Tensor:
        """数据标准化"""
        mean = keypoints.mean()
        std = keypoints.std() + 1e-6
        return (keypoints - mean) / std


class MixedViewDataset(Dataset):
    """
    混合视角数据集

    模拟视频中视角变化的情况
    """

    def __init__(self, num_samples: int = 500, sequence_length: int = 30):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.view_to_id = {'side': 0, 'front': 1, 'back': 2, 'mixed': 3}

        print(f"生成 {num_samples} 个混合视角样本...")
        self.data = self._generate_dataset()
        print("✅ 混合视角数据集生成完成")

    def _generate_dataset(self) -> List[Dict]:
        dataset = []

        for _ in range(self.num_samples):
            # 随机选择2-3个视角进行混合
            views = random.sample(['side', 'front', 'back'], k=random.randint(2, 3))
            params = RunningParameters(
                cadence=np.random.uniform(160, 200),
                stride_length=np.random.uniform(0.8, 1.3),
                vertical_oscillation=np.random.uniform(0.03, 0.12),
                trunk_lean=np.random.uniform(5, 20),
                arm_swing=np.random.uniform(0.1, 0.3),
                ground_contact_time=np.random.uniform(0.2, 0.4),
                asymmetry=np.random.uniform(0, 0.1),
                noise_level=np.random.uniform(0.005, 0.02)
            )

            sample = self._generate_mixed_sample(views, params)
            dataset.append(sample)

        return dataset

    def _generate_mixed_sample(self, views: List[str], params: RunningParameters) -> Dict:
        """生成混合视角样本"""
        # 随机分配每个视角的帧数
        total_frames = self.sequence_length
        num_segments = len(views)
        segment_lengths = np.random.multinomial(total_frames, [1 / num_segments] * num_segments)

        keypoints_sequence = []
        phase_labels = []
        frame_idx = 0

        for view, seg_len in zip(views, segment_lengths):
            model = BiomechanicalModel(view)
            freq = params.cadence / 60.0

            for i in range(seg_len):
                t = (frame_idx + i) / 30.0 * freq * 2 * np.pi
                kp = model.generate_keypoints(t, params, True)
                keypoints_sequence.append(kp.flatten())
                phase_labels.append(model.determine_phase(t))

            frame_idx += seg_len

        # 计算质量评分
        quality_scores = self._calculate_quality_scores(params)

        return {
            'keypoints': np.array(keypoints_sequence, dtype=np.float32),
            'phase_labels': np.array(phase_labels, dtype=np.int64),
            'quality_scores': np.array(quality_scores, dtype=np.float32),
            'view': 'mixed',
            'view_id': 3
        }

    def _calculate_quality_scores(self, params: RunningParameters) -> List[float]:
        """计算质量评分"""
        stability = 90 - params.vertical_oscillation * 300 + np.random.uniform(-5, 5)
        efficiency = 90 - abs(params.cadence - 185) * 0.5 + np.random.uniform(-5, 5)
        form = 90 - abs(params.trunk_lean - 12) * 2 + np.random.uniform(-5, 5)
        rhythm = 90 - params.asymmetry * 200 + np.random.uniform(-5, 5)

        stability = np.clip(stability, 40, 100)
        efficiency = np.clip(efficiency, 40, 100)
        form = np.clip(form, 40, 100)
        rhythm = np.clip(rhythm, 40, 100)

        total = stability * 0.3 + efficiency * 0.3 + form * 0.2 + rhythm * 0.2

        return [total, stability, efficiency, form, rhythm]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample = self.data[idx]

        keypoints = torch.FloatTensor(sample['keypoints'])
        phase_labels = torch.LongTensor(sample['phase_labels'])
        quality_scores = torch.FloatTensor(sample['quality_scores'])
        view_id = torch.LongTensor([sample['view_id']])

        # 标准化
        mean = keypoints.mean()
        std = keypoints.std() + 1e-6
        keypoints = (keypoints - mean) / std

        return keypoints, phase_labels, quality_scores, view_id.squeeze()


def create_dataloaders(batch_size: int = 32,
                       num_train: int = 2000,
                       num_val: int = 500,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Returns:
        train_loader, val_loader
    """
    # 创建数据集
    train_dataset = RunningDataset(num_samples=num_train, augment=True)
    val_dataset = RunningDataset(num_samples=num_val, augment=False)

    # 添加混合视角数据
    mixed_train = MixedViewDataset(num_samples=num_train // 4)
    mixed_val = MixedViewDataset(num_samples=num_val // 4)

    # 合并数据集
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([train_dataset, mixed_train])
    val_dataset = ConcatDataset([val_dataset, mixed_val])

    # 创建加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试跑步动作数据集")
    print("=" * 70)

    # 测试单视角数据集
    print("\n1. 测试 RunningDataset:")
    dataset = RunningDataset(num_samples=100, sequence_length=30)
    print(f"   数据集大小: {len(dataset)}")

    keypoints, phases, quality, view_id = dataset[0]
    print(f"   关键点形状: {keypoints.shape}")
    print(f"   阶段标签形状: {phases.shape}")
    print(f"   质量评分形状: {quality.shape}")
    print(f"   视角ID: {view_id.item()}")
    print(f"   质量评分: {quality.numpy()}")

    # 测试混合视角数据集
    print("\n2. 测试 MixedViewDataset:")
    mixed_dataset = MixedViewDataset(num_samples=50)
    keypoints, phases, quality, view_id = mixed_dataset[0]
    print(f"   数据集大小: {len(mixed_dataset)}")
    print(f"   视角ID: {view_id.item()} (应为3=mixed)")

    # 测试数据加载器
    print("\n3. 测试 DataLoader:")
    train_loader, val_loader = create_dataloaders(batch_size=16, num_train=200, num_val=50)
    print(f"   训练集批次数: {len(train_loader)}")
    print(f"   验证集批次数: {len(val_loader)}")

    for batch in train_loader:
        keypoints, phases, quality, view_ids = batch
        print(f"   批次关键点形状: {keypoints.shape}")
        print(f"   批次阶段形状: {phases.shape}")
        print(f"   批次质量形状: {quality.shape}")
        print(f"   批次视角ID: {view_ids.shape}")
        break

    print("\n" + "=" * 70)
    print("✅ 所有测试通过!")
    print("=" * 70)
