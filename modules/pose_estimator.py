# modules/pose_estimator.py
"""
姿态估计器模块 - 统一接口设计

支持多种后端：MediaPipe（当前）、MMPose（预留）
新增功能：集成MotionBERT进行2D→3D姿态提升

架构：
  视频帧 → MediaPipe(2D) → MotionBERT(3D提升) → 3D关键点
"""
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from config.config import POSE_CONFIG


class BasePoseEstimator(ABC):
    """
    姿态估计器基类
    定义统一接口，方便后续替换不同的姿态估计后端
    """

    # 统一的关键点定义（基于COCO格式）
    KEYPOINT_NAMES = {
        0: 'nose',
        1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
        4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
        7: 'left_ear', 8: 'right_ear',
        9: 'mouth_left', 10: 'mouth_right',
        11: 'left_shoulder', 12: 'right_shoulder',
        13: 'left_elbow', 14: 'right_elbow',
        15: 'left_wrist', 16: 'right_wrist',
        17: 'left_pinky', 18: 'right_pinky',
        19: 'left_index', 20: 'right_index',
        21: 'left_thumb', 22: 'right_thumb',
        23: 'left_hip', 24: 'right_hip',
        25: 'left_knee', 26: 'right_knee',
        27: 'left_ankle', 28: 'right_ankle',
        29: 'left_heel', 30: 'right_heel',
        31: 'left_foot_index', 32: 'right_foot_index'
    }

    # 跑步分析关键关节
    RUNNING_KEYPOINTS = {
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'nose': 0
    }

    @abstractmethod
    def process_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """处理视频帧序列，返回关键点时间序列"""
        pass

    @abstractmethod
    def process_single_frame(self, frame: np.ndarray) -> Dict:
        """处理单帧图像"""
        pass

    @abstractmethod
    def visualize_pose(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """可视化姿态"""
        pass

    @abstractmethod
    def close(self):
        """释放资源"""
        pass

    def get_keypoint_by_name(self, keypoints: Dict, name: str) -> Optional[Dict]:
        """根据名称获取关键点"""
        for kp in keypoints['landmarks']:
            if kp['name'] == name:
                return kp
        return None

    def get_running_keypoints(self, keypoints: Dict) -> Dict:
        """提取跑步分析所需的关键关节"""
        running_kps = {}
        for name, idx in self.RUNNING_KEYPOINTS.items():
            kp = keypoints['landmarks'][idx]
            if kp['visibility'] > 0.5:
                running_kps[name] = kp
        return running_kps


class MediaPipePoseEstimator(BasePoseEstimator):
    """
    MediaPipe姿态估计器
    当前主要使用的后端
    """

    def __init__(self, config: Dict = None):
        """初始化MediaPipe姿态估计器"""
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        config = config or POSE_CONFIG

        self.pose = self.mp_pose.Pose(
            model_complexity=config.get('model_complexity', 1),
            min_detection_confidence=config.get('min_detection_confidence', 0.5),
            min_tracking_confidence=config.get('min_tracking_confidence', 0.5),
            static_image_mode=config.get('static_image_mode', False)
        )

        self.backend = 'mediapipe'
        self.num_keypoints = 33

    def process_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        处理视频帧序列

        改进：对于未检测到的帧（特别是初始帧），使用静态模式重试

        Args:
            frames: 视频帧列表
        Returns:
            keypoints_sequence: 关键点时间序列
        """
        keypoints_sequence = []
        consecutive_failures = 0
        max_consecutive_failures = 5  # 如果连续失败超过5帧，可能真的没有人

        for idx, frame in enumerate(frames):
            keypoints = self.process_single_frame(frame)
            keypoints['frame_idx'] = idx

            # 如果初始几帧检测失败，使用静态模式重试
            if not keypoints.get('detected', False):
                consecutive_failures += 1
                # 对前10帧或连续失败的帧进行重试
                if idx < 10 or consecutive_failures <= max_consecutive_failures:
                    retry_kp = self._retry_detection_static(frame)
                    if retry_kp.get('detected', False):
                        keypoints = retry_kp
                        keypoints['frame_idx'] = idx
                        keypoints['detection_method'] = 'static_retry'
                        consecutive_failures = 0
            else:
                consecutive_failures = 0

            keypoints_sequence.append(keypoints)

        return keypoints_sequence

    def _retry_detection_static(self, frame: np.ndarray) -> Dict:
        """
        使用静态图像模式重试检测

        静态模式会对每帧独立进行检测，不依赖前帧的跟踪，
        可能在初始帧或跟踪丢失时获得更好的检测结果。

        Args:
            frame: BGR格式的图像

        Returns:
            关键点数据字典
        """
        import mediapipe as mp

        # 创建临时的静态模式检测器
        try:
            with mp.solutions.pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.3,  # 降低阈值以提高召回率
                min_tracking_confidence=0.3,
                static_image_mode=True  # 静态模式
            ) as static_pose:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = static_pose.process(image_rgb)

                if results.pose_landmarks:
                    keypoints = self._extract_keypoints(results.pose_landmarks, frame.shape)
                    keypoints['detected'] = True
                    return keypoints

        except Exception as e:
            pass

        return self._get_empty_keypoints(frame.shape)

    def process_single_frame(self, frame: np.ndarray) -> Dict:
        """
        处理单帧图像
        Args:
            frame: BGR格式的图像
        Returns:
            关键点数据字典
        """
        # BGR转RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe姿态估计
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = self._extract_keypoints(results.pose_landmarks, frame.shape)
            keypoints['detected'] = True
        else:
            keypoints = self._get_empty_keypoints(frame.shape)
            keypoints['detected'] = False

        return keypoints

    def _extract_keypoints(self, landmarks, image_shape: Tuple) -> Dict:
        """
        提取关键点坐标
        """
        h, w = image_shape[:2]
        keypoints = {'landmarks': [], 'visibility': []}

        for idx, landmark in enumerate(landmarks.landmark):
            # 归一化坐标
            x_norm = landmark.x
            y_norm = landmark.y
            z_norm = landmark.z
            visibility = landmark.visibility

            # 像素坐标
            x_pixel = int(x_norm * w)
            y_pixel = int(y_norm * h)

            keypoints['landmarks'].append({
                'id': idx,
                'name': self.KEYPOINT_NAMES.get(idx, f'point_{idx}'),
                'x': x_pixel,
                'y': y_pixel,
                'z': z_norm,
                'x_norm': x_norm,
                'y_norm': y_norm,
                'visibility': visibility
            })
            keypoints['visibility'].append(visibility)

        return keypoints

    def _get_empty_keypoints(self, image_shape: Tuple) -> Dict:
        """获取空关键点（检测失败时）"""
        return {
            'landmarks': [
                {
                    'id': i,
                    'name': self.KEYPOINT_NAMES.get(i, f'point_{i}'),
                    'x': 0, 'y': 0, 'z': 0,
                    'x_norm': 0, 'y_norm': 0,
                    'visibility': 0
                }
                for i in range(33)
            ],
            'visibility': [0] * 33
        }

    def visualize_pose(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """
        可视化姿态（火柴人）
        """
        vis_frame = frame.copy()

        if not keypoints.get('detected', False):
            return vis_frame

        # 绘制关键点
        for kp in keypoints['landmarks']:
            if kp['visibility'] > 0.5:
                color = self._get_keypoint_color(kp['id'])
                cv2.circle(vis_frame, (kp['x'], kp['y']), 5, color, -1)

        # 绘制骨架连接
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start_kp = keypoints['landmarks'][start_idx]
            end_kp = keypoints['landmarks'][end_idx]

            if start_kp['visibility'] > 0.5 and end_kp['visibility'] > 0.5:
                color = self._get_connection_color(start_idx, end_idx)
                cv2.line(vis_frame,
                         (start_kp['x'], start_kp['y']),
                         (end_kp['x'], end_kp['y']),
                         color, 2)

        return vis_frame

    def _get_keypoint_color(self, kp_id: int) -> Tuple[int, int, int]:
        """根据关键点ID获取颜色"""
        # 躯干：蓝色
        if kp_id in [11, 12, 23, 24]:
            return (255, 128, 0)
        # 腿部：绿色
        elif kp_id in [25, 26, 27, 28, 29, 30, 31, 32]:
            return (0, 255, 0)
        # 手臂：红色
        elif kp_id in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            return (0, 0, 255)
        # 头部：黄色
        else:
            return (0, 255, 255)

    def _get_connection_color(self, start_id: int, end_id: int) -> Tuple[int, int, int]:
        """根据连接获取颜色"""
        # 腿部连接
        leg_ids = {23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
        if start_id in leg_ids and end_id in leg_ids:
            return (0, 200, 0)
        # 手臂连接
        arm_ids = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
        if start_id in arm_ids and end_id in arm_ids:
            return (200, 0, 0)
        # 躯干连接
        return (200, 128, 0)

    def close(self):
        """释放资源"""
        if self.pose is not None:
            self.pose.close()
            self.pose = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# 工厂函数
def create_pose_estimator(backend: str = 'mediapipe', config: Dict = None) -> BasePoseEstimator:
    """
    创建姿态估计器的工厂函数

    Args:
        backend: 后端类型 ('mediapipe')
        config: 配置字典

    Returns:
        姿态估计器实例
    """
    if backend == 'mediapipe':
        return MediaPipePoseEstimator(config)
    else:
        raise ValueError(f"不支持的后端: {backend}，目前仅支持 'mediapipe'")


# 兼容性别名
PoseEstimator = MediaPipePoseEstimator


# ============================================================================
# 集成3D姿态提升
# ============================================================================

class Pose3DEstimator:
    """
    集成式3D姿态估计器

    流程：MediaPipe(2D) → MotionBERT(3D提升)

    针对4GB显存优化
    """

    def __init__(
        self,
        backend_2d: str = 'mediapipe',
        enable_3d: bool = True,
        device: str = 'auto',
        config: Dict = None
    ):
        """
        初始化3D姿态估计器

        Args:
            backend_2d: 2D后端 ('mediapipe', 'mmpose')
            enable_3d: 是否启用3D提升
            device: 推理设备 ('auto', 'cuda', 'cpu')
            config: 配置字典
        """
        self.enable_3d = enable_3d

        # 创建2D估计器
        self.estimator_2d = create_pose_estimator(backend_2d, config)

        # 创建3D提升器
        self.lifter = None
        if enable_3d:
            try:
                from modules.pose_lifter import PoseLifter
                self.lifter = PoseLifter(device=device)
                print("✅ 3D姿态提升已启用")
            except Exception as e:
                print(f"⚠️ 3D姿态提升初始化失败: {e}")
                print("   将只使用2D姿态估计")
                self.enable_3d = False

    def process_frames(
        self,
        frames: List[np.ndarray],
        lift_to_3d: bool = True,
        view_angle: str = 'side'
    ) -> Dict:
        """
        处理视频帧序列

        Args:
            frames: 视频帧列表
            lift_to_3d: 是否进行3D提升
            view_angle: 视角类型 ('side' 或 'front')，用于3D校正

        Returns:
            结果字典，包含：
            - keypoints_2d: 2D关键点序列
            - keypoints_3d: 3D关键点序列（如启用）
            - poses_3d: (T, 17, 3) numpy数组
            - confidence_3d: 置信度
        """
        # 2D姿态估计
        keypoints_2d = self.estimator_2d.process_frames(frames)

        result = {
            'keypoints_2d': keypoints_2d,
            'keypoints_3d': None,
            'poses_3d': None,
            'confidence_3d': None,
            'has_3d': False
        }

        # 3D提升
        if self.enable_3d and self.lifter and lift_to_3d:
            try:
                lift_result = self.lifter.lift_sequence(
                    keypoints_2d,
                    view_angle=view_angle,
                    apply_2d_correction=True
                )

                if lift_result.get('success', False):
                    keypoints_3d_raw = lift_result['keypoints_3d']
                    valid_ratio = lift_result.get('valid_frames_ratio', 0)

                    # 提取poses_3d数组
                    poses_3d = np.array([
                        kp['keypoints_h36m'] for kp in keypoints_3d_raw
                    ])  # (T, 17, 3)

                    # 生成置信度（基于valid_mask）
                    confidence = np.array([
                        np.ones(17) if kp.get('detected', False) else np.zeros(17)
                        for kp in keypoints_3d_raw
                    ])

                    # 转换为关键点格式
                    keypoints_3d = self._convert_3d_to_keypoints(
                        keypoints_2d, poses_3d, confidence
                    )

                    result['keypoints_3d'] = keypoints_3d
                    result['poses_3d'] = poses_3d
                    result['confidence_3d'] = confidence
                    result['has_3d'] = True
                    result['lift_info'] = {
                        'success': True,
                        'valid_frames_ratio': valid_ratio
                    }

                    detected_3d = sum(1 for kp in keypoints_3d if kp.get('has_3d', False))
                    print(f"   3D姿态提升完成: {detected_3d}/{len(keypoints_3d)} 帧 ({valid_ratio*100:.1f}% 有效)")
                else:
                    error_msg = lift_result.get('error', 'Unknown error')
                    print(f"   ⚠️ 3D提升失败: {error_msg}")
                    result['lift_info'] = {'success': False, 'error': error_msg}

            except Exception as e:
                print(f"⚠️ 3D提升失败: {e}")
                result['lift_info'] = {'success': False, 'error': str(e)}

        return result

    def _convert_3d_to_keypoints(
        self,
        keypoints_2d: List[Dict],
        poses_3d: np.ndarray,
        confidence: np.ndarray
    ) -> List[Dict]:
        """
        将3D姿态转换为关键点格式

        合并2D和3D信息到统一的数据结构
        """
        from modules.pose_lifter import KeypointMapper

        keypoints_3d = []

        for i, kp_2d in enumerate(keypoints_2d):
            kp_3d = kp_2d.copy()
            kp_3d['has_3d'] = False

            if i < len(poses_3d) and np.any(poses_3d[i] != 0):
                # 添加3D信息
                pose = poses_3d[i]
                conf = confidence[i] if i < len(confidence) else np.ones(17)

                # H36M格式的3D关键点
                kp_3d['pose_3d'] = KeypointMapper.h36m_to_analysis_format(pose)
                kp_3d['confidence_3d'] = conf.tolist()
                kp_3d['has_3d'] = True

                # 将3D坐标映射回MediaPipe格式的landmarks
                # 用于与现有代码兼容
                kp_3d['landmarks_3d'] = self._map_h36m_to_mediapipe_3d(pose)

            keypoints_3d.append(kp_3d)

        return keypoints_3d

    def _map_h36m_to_mediapipe_3d(self, pose_h36m: np.ndarray) -> Dict:
        """
        将H36M 3D关键点映射为类MediaPipe格式

        用于与现有分析代码兼容
        """
        # H36M到MediaPipe的反向映射
        H36M_TO_MEDIAPIPE = {
            4: 23,   # L_Hip -> left_hip
            1: 24,   # R_Hip -> right_hip
            5: 25,   # L_Knee -> left_knee
            2: 26,   # R_Knee -> right_knee
            6: 27,   # L_Ankle -> left_ankle
            3: 28,   # R_Ankle -> right_ankle
            11: 11,  # L_Shoulder -> left_shoulder
            14: 12,  # R_Shoulder -> right_shoulder
            12: 13,  # L_Elbow -> left_elbow
            15: 14,  # R_Elbow -> right_elbow
            13: 15,  # L_Wrist -> left_wrist
            16: 16,  # R_Wrist -> right_wrist
            10: 0,   # Head -> nose
        }

        landmarks_3d = {}
        for h36m_idx, mp_idx in H36M_TO_MEDIAPIPE.items():
            landmarks_3d[mp_idx] = {
                'x_3d': float(pose_h36m[h36m_idx, 0]),
                'y_3d': float(pose_h36m[h36m_idx, 1]),
                'z_3d': float(pose_h36m[h36m_idx, 2])
            }

        return landmarks_3d

    def visualize_pose(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """可视化姿态"""
        return self.estimator_2d.visualize_pose(frame, keypoints)

    def close(self):
        """释放资源"""
        self.estimator_2d.close()
        self.lifter = None


def create_pose_estimator_3d(
    backend_2d: str = 'mediapipe',
    enable_3d: bool = True,
    device: str = 'auto'
) -> Pose3DEstimator:
    """
    创建3D姿态估计器的工厂函数

    Args:
        backend_2d: 2D后端类型
        enable_3d: 是否启用3D提升
        device: 推理设备

    Returns:
        Pose3DEstimator实例
    """
    return Pose3DEstimator(
        backend_2d=backend_2d,
        enable_3d=enable_3d,
        device=device
    )


# 模块测试
if __name__ == "__main__":
    import sys
    from modules.video_processor import VideoProcessor

    print("=" * 60)
    print("测试姿态估计模块")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("用法: python pose_estimator.py <video_path>")
        print("\n测试工厂函数...")

        # 测试MediaPipe
        estimator = create_pose_estimator('mediapipe')
        print(f"创建成功: {estimator.backend}")
        estimator.close()

        print("\n✅ 模块基本测试完成!")
    else:
        video_path = sys.argv[1]

        try:
            # 加载视频
            print("加载视频...")
            processor = VideoProcessor(video_path)
            frames, fps = processor.extract_frames(target_fps=30, max_frames=30)
            print(f"提取了 {len(frames)} 帧")

            # 姿态估计
            print("\n进行姿态估计...")
            estimator = create_pose_estimator('mediapipe')
            keypoints_seq = estimator.process_frames(frames)

            detected_count = sum(1 for kp in keypoints_seq if kp['detected'])
            print(f"检测成功: {detected_count}/{len(keypoints_seq)} 帧")

            # 可视化
            saved = False
            for frame, kps in zip(frames, keypoints_seq):
                if kps['detected']:
                    vis_frame = estimator.visualize_pose(frame, kps)
                    cv2.imwrite('pose_test_output.jpg', vis_frame)
                    print("\n可视化结果已保存: pose_test_output.jpg")
                    saved = True
                    break

            if not saved:
                print("\n未检测到姿态")

            estimator.close()
            processor.release()
            print("\n✅ 模块测试完成!")

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
