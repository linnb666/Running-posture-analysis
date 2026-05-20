"""
AthletePose3D 数据工具模块

功能：
1. 加载和解析 AP3D 数据集
2. 建立 frame_81 与 train.pkl 的映射关系
3. 数据归一化和预处理
4. MediaPipe 噪声模拟增强
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import warnings


class AP3DDataUtils:
    """AthletePose3D 数据工具类"""

    # H36M 17 关节定义（与 pose_lifter.py 保持一致）
    H36M_JOINTS = [
        'hip',           # 0 - 髋部中心
        'right_hip',     # 1
        'right_knee',    # 2
        'right_ankle',   # 3
        'left_hip',      # 4
        'left_knee',     # 5
        'left_ankle',    # 6
        'spine',         # 7 - 脊柱中心
        'thorax',        # 8 - 胸腔
        'neck',          # 9 - 颈部
        'head',          # 10 - 头顶
        'left_shoulder', # 11
        'left_elbow',    # 12
        'left_wrist',    # 13
        'right_shoulder',# 14
        'right_elbow',   # 15
        'right_wrist'    # 16
    ]

    def __init__(self, data_dir: str):
        """
        初始化 AP3D 数据工具

        Args:
            data_dir: pose_3d_v3 目录路径
        """
        self.data_dir = Path(data_dir)
        self.frame81_train_dir = self.data_dir / 'frame_81' / 'train'
        self.frame81_test_dir = self.data_dir / 'frame_81' / 'test'
        self.train_pkl_path = self.data_dir / 'train.pkl'
        self.valid_pkl_path = self.data_dir / 'valid.pkl'

        # 缓存
        self._train_data = None
        self._valid_data = None
        self._action_indices = None
        self._frame81_mapping = None

    def load_train_pkl(self, force_reload: bool = False) -> List[Dict]:
        """
        加载 train.pkl

        Returns:
            样本列表，每个样本包含 action, image_path, videoid, fps 等字段
        """
        if self._train_data is not None and not force_reload:
            return self._train_data

        if not self.train_pkl_path.exists():
            raise FileNotFoundError(f"train.pkl not found at {self.train_pkl_path}")

        print(f"加载 train.pkl ({self.train_pkl_path})...")
        with open(self.train_pkl_path, 'rb') as f:
            self._train_data = pickle.load(f)

        print(f"  加载完成: {len(self._train_data)} 个样本")
        return self._train_data

    def load_valid_pkl(self, force_reload: bool = False) -> List[Dict]:
        """加载 valid.pkl"""
        if self._valid_data is not None and not force_reload:
            return self._valid_data

        if not self.valid_pkl_path.exists():
            raise FileNotFoundError(f"valid.pkl not found at {self.valid_pkl_path}")

        print(f"加载 valid.pkl ({self.valid_pkl_path})...")
        with open(self.valid_pkl_path, 'rb') as f:
            self._valid_data = pickle.load(f)

        print(f"  加载完成: {len(self._valid_data)} 个样本")
        return self._valid_data

    def get_action_indices(self, force_rebuild: bool = False) -> Dict[str, List[int]]:
        """
        获取各 action 类型在 train.pkl 中的索引

        Returns:
            {action: [indices...]} 字典
        """
        if self._action_indices is not None and not force_rebuild:
            return self._action_indices

        train_data = self.load_train_pkl()
        self._action_indices = defaultdict(list)

        for i, sample in enumerate(train_data):
            action = sample.get('action', 'unknown')
            self._action_indices[action].append(i)

        # 转为普通 dict
        self._action_indices = dict(self._action_indices)
        return self._action_indices

    def build_frame81_index_map(self,
                                 actions: Optional[List[str]] = None,
                                 split: str = 'train') -> Dict[str, Dict]:
        """
        建立 frame_81 文件索引映射

        基于分析结果：
        - train.pkl 样本按 action 类型连续排列
        - frame_81 文件是从 train.pkl 按滑动窗口生成的
        - 滑动窗口步长约 37 帧

        Args:
            actions: 要筛选的 action 类型列表，如 ['rm']，None 表示全部
            split: 'train' 或 'test'

        Returns:
            {
                'file_indices': [int...],  # frame_81 文件索引
                'file_paths': [str...],    # frame_81 文件路径
                'action_ranges': {action: (start, end)},  # 各 action 对应的文件范围
                'total_files': int
            }
        """
        # 获取 action 索引
        action_indices = self.get_action_indices()

        # 确定 frame_81 目录
        if split == 'train':
            frame81_dir = self.frame81_train_dir
            pkl_len = len(self.load_train_pkl())
        else:
            frame81_dir = self.frame81_test_dir
            pkl_len = len(self.load_valid_pkl())

        # 获取 frame_81 文件列表
        all_files = sorted([f for f in os.listdir(frame81_dir) if f.endswith('.pkl')])
        total_files = len(all_files)

        print(f"frame_81/{split} 共有 {total_files} 个文件")

        # 计算滑动窗口步长
        stride = pkl_len / total_files
        print(f"估算滑动窗口步长: {stride:.2f} 帧")

        # 计算各 action 对应的 frame_81 文件范围
        action_ranges = {}
        for action, indices in sorted(action_indices.items()):
            if indices:
                start_idx = min(indices)
                end_idx = max(indices)
                f81_start = int(start_idx / stride)
                f81_end = min(int(end_idx / stride) + 1, total_files)
                action_ranges[action] = (f81_start, f81_end)
                print(f"  {action}: train.pkl [{start_idx}, {end_idx}] -> frame_81 [{f81_start}, {f81_end}]")

        # 根据 actions 筛选文件
        if actions is None:
            selected_indices = list(range(total_files))
        else:
            selected_indices = []
            for action in actions:
                if action in action_ranges:
                    start, end = action_ranges[action]
                    selected_indices.extend(range(start, end))
            selected_indices = sorted(set(selected_indices))

        # 构建文件路径列表
        file_paths = [str(frame81_dir / all_files[i]) for i in selected_indices]

        result = {
            'file_indices': selected_indices,
            'file_paths': file_paths,
            'action_ranges': action_ranges,
            'total_files': total_files,
            'selected_count': len(selected_indices),
            'stride': stride
        }

        print(f"筛选后文件数: {len(selected_indices)}")
        return result

    def load_frame81_sample(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        加载单个 frame_81 pkl 文件

        Args:
            file_path: pkl 文件路径

        Returns:
            {'data_input': (81, 17, 3), 'data_label': (81, 17, 3)}
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def normalize_2d_for_model(data_input: np.ndarray,
                               clip_range: Tuple[float, float] = (-1.0, 2.0),
                               fill_nan: bool = True) -> np.ndarray:
        """
        归一化 2D 输入数据

        Args:
            data_input: (T, 17, 3) 2D 关键点数据 (x, y, conf)
            clip_range: 裁剪范围
            fill_nan: 是否填充 NaN/Inf

        Returns:
            归一化后的数据 (T, 17, 3)
        """
        data = data_input.copy().astype(np.float32)

        # 处理 NaN 和 Inf
        if fill_nan:
            nan_mask = np.isnan(data) | np.isinf(data)
            if nan_mask.any():
                warnings.warn(f"Found {nan_mask.sum()} NaN/Inf values, filling with 0")
                data[nan_mask] = 0.0

        # 裁剪到合理范围
        data = np.clip(data, clip_range[0], clip_range[1])

        return data

    @staticmethod
    def make_root_relative(pose_3d: np.ndarray, root_idx: int = 0) -> np.ndarray:
        """
        将 3D 姿态转换为相对于根节点 (pelvis) 的坐标

        Args:
            pose_3d: (T, 17, 3) 或 (17, 3) 3D 姿态
            root_idx: 根节点索引，默认 0 (hip/pelvis)

        Returns:
            相对坐标的 3D 姿态
        """
        if pose_3d.ndim == 2:  # (17, 3)
            root = pose_3d[root_idx:root_idx+1, :]  # (1, 3)
            return pose_3d - root
        elif pose_3d.ndim == 3:  # (T, 17, 3)
            root = pose_3d[:, root_idx:root_idx+1, :]  # (T, 1, 3)
            return pose_3d - root
        else:
            raise ValueError(f"Invalid pose_3d shape: {pose_3d.shape}")

    @staticmethod
    def simulate_mediapipe_noise(data_2d: np.ndarray,
                                  coord_noise_std: float = 0.01,
                                  dropout_prob: float = 0.05,
                                  conf_noise_std: float = 0.1) -> np.ndarray:
        """
        模拟 MediaPipe 检测噪声用于数据增强

        Args:
            data_2d: (T, 17, 3) 2D 关键点 (x, y, conf)
            coord_noise_std: 坐标高斯噪声标准差
            dropout_prob: 随机丢点概率
            conf_noise_std: 置信度噪声标准差

        Returns:
            添加噪声后的数据
        """
        data = data_2d.copy()
        T, J, C = data.shape

        # 1. 坐标高斯噪声
        if coord_noise_std > 0:
            noise = np.random.randn(T, J, 2) * coord_noise_std
            data[:, :, :2] += noise

        # 2. 随机丢点（设置为 0）
        if dropout_prob > 0:
            dropout_mask = np.random.rand(T, J) < dropout_prob
            data[dropout_mask, :] = 0.0

        # 3. 置信度噪声
        if conf_noise_std > 0 and C >= 3:
            conf_noise = np.random.randn(T, J) * conf_noise_std
            data[:, :, 2] = np.clip(data[:, :, 2] + conf_noise, 0.0, 1.0)

        return data

    @staticmethod
    def calculate_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        计算 Mean Per Joint Position Error (MPJPE)

        Args:
            pred: (T, 17, 3) 或 (17, 3) 预测 3D 姿态
            gt: (T, 17, 3) 或 (17, 3) 真值 3D 姿态

        Returns:
            MPJPE (与输入单位一致)
        """
        diff = pred - gt
        dist = np.linalg.norm(diff, axis=-1)  # (T, 17) or (17,)
        return float(np.mean(dist))

    @staticmethod
    def calculate_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        计算三点形成的角度（p2 为顶点）

        Args:
            p1, p2, p3: 3D 坐标点

        Returns:
            角度（度）
        """
        v1 = p1 - p2
        v2 = p3 - p2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.degrees(np.arccos(cos_angle)))

    def calculate_knee_angles(self, pose_3d: np.ndarray) -> Tuple[float, float]:
        """
        计算膝关节角度

        Args:
            pose_3d: (17, 3) 3D 姿态

        Returns:
            (left_knee_angle, right_knee_angle)
        """
        # 左膝: l_hip(4) - l_knee(5) - l_ankle(6)
        left_angle = self.calculate_joint_angle(
            pose_3d[4], pose_3d[5], pose_3d[6]
        )

        # 右膝: r_hip(1) - r_knee(2) - r_ankle(3)
        right_angle = self.calculate_joint_angle(
            pose_3d[1], pose_3d[2], pose_3d[3]
        )

        return left_angle, right_angle


def run_unit_tests():
    """运行单元测试"""
    print("=" * 70)
    print("AP3D Utils Unit Tests")
    print("=" * 70)

    # 初始化
    data_dir = r'E:\PycharmProjects\running\pose_3d_v3'
    utils = AP3DDataUtils(data_dir)

    # 测试 1: 加载 train.pkl
    print("\n[Test 1] Load train.pkl")
    try:
        train_data = utils.load_train_pkl()
        print(f"  [OK] Loaded {len(train_data)} samples")

        # 检查样本结构
        sample = train_data[0]
        print(f"  Sample keys: {list(sample.keys())}")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试 2: 获取 action 索引
    print("\n[Test 2] Get action indices")
    try:
        action_indices = utils.get_action_indices()
        for action, indices in action_indices.items():
            print(f"  {action}: {len(indices)} samples, index range [{min(indices)}, {max(indices)}]")
        print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试 3: 建立 frame_81 映射（仅 rm）
    print("\n[Test 3] Build frame_81 mapping (rm only)")
    try:
        mapping = utils.build_frame81_index_map(actions=['rm'], split='train')
        print(f"  [OK] Found {mapping['selected_count']} rm-related files")
        print(f"  File index range: [{mapping['file_indices'][0]}, {mapping['file_indices'][-1]}]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试 4: 加载 frame_81 样本
    print("\n[Test 4] Load frame_81 sample")
    try:
        if mapping['file_paths']:
            sample_path = mapping['file_paths'][0]
            sample = utils.load_frame81_sample(sample_path)
            data_input = sample['data_input']
            data_label = sample['data_label']

            print(f"  data_input shape: {data_input.shape}, dtype: {data_input.dtype}")
            print(f"  data_input range: [{data_input.min():.4f}, {data_input.max():.4f}]")
            print(f"  data_label shape: {data_label.shape}, dtype: {data_label.dtype}")
            print(f"  data_label range: [{data_label.min():.4f}, {data_label.max():.4f}]")
            print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试 5: 数据归一化
    print("\n[Test 5] Normalize 2D data")
    try:
        normalized = utils.normalize_2d_for_model(data_input)
        print(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
        print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试 6: Root-relative 转换
    print("\n[Test 6] Root-relative transform")
    try:
        root_rel = utils.make_root_relative(data_label)
        print(f"  Before: hip (joint 0) = {data_label[0, 0]}")
        print(f"  After:  hip (joint 0) = {root_rel[0, 0]} (should be 0)")
        assert np.allclose(root_rel[:, 0, :], 0.0), "Root should be 0"
        print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试 7: MediaPipe 噪声模拟
    print("\n[Test 7] MediaPipe noise simulation")
    try:
        noisy = utils.simulate_mediapipe_noise(data_input, coord_noise_std=0.01)
        diff = np.abs(noisy - data_input)
        print(f"  Mean noise difference: {diff.mean():.6f}")
        print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试 8: MPJPE 计算
    print("\n[Test 8] MPJPE calculation")
    try:
        # 使用 data_label 作为预测和真值（应该为 0）
        mpjpe = utils.calculate_mpjpe(data_label, data_label)
        print(f"  MPJPE (same data): {mpjpe:.6f} (should be 0)")

        # 添加一些偏移测试
        pred_offset = data_label + 0.01
        mpjpe_offset = utils.calculate_mpjpe(pred_offset, data_label)
        print(f"  MPJPE (offset 0.01): {mpjpe_offset:.6f}")
        print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试 9: 膝关节角度计算
    print("\n[Test 9] Knee angle calculation")
    try:
        pose_3d = data_label[0]  # 第一帧
        left_knee, right_knee = utils.calculate_knee_angles(pose_3d)
        print(f"  Left knee angle: {left_knee:.2f} deg")
        print(f"  Right knee angle: {right_knee:.2f} deg")
        print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n" + "=" * 70)
    print("Unit Tests Complete")
    print("=" * 70)


if __name__ == '__main__':
    run_unit_tests()
