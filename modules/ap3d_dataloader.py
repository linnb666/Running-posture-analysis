"""
AthletePose3D DataLoader 模块

功能：
1. AP3DFrame81Dataset: 加载 frame_81 格式的训练数据
2. 支持按 action 类型筛选（如 rm=跑步）
3. 数据增强（MediaPipe 噪声模拟）
4. Root-relative 3D 坐标转换
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.ap3d_utils import AP3DDataUtils


class AP3DFrame81Dataset(Dataset):
    """
    AthletePose3D Frame-81 数据集

    每个样本包含:
    - data_input: (81, 17, 3) - 2D 关键点 (x, y, confidence)
    - data_label: (81, 17, 3) - 3D 关键点 (x, y, z)

    Args:
        data_dir: pose_3d_v3 目录路径
        split: 'train' 或 'test'
        actions: 筛选的 action 类型列表，如 ['rm']，None 表示全部
        transform: 2D 输入的数据变换/增强函数
        root_relative: 是否将 3D 标签转换为 root-relative 坐标
        normalize_2d: 是否归一化 2D 输入
        augment: 是否启用数据增强
        augment_config: 增强参数配置
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 actions: Optional[List[str]] = None,
                 transform: Optional[Callable] = None,
                 root_relative: bool = True,
                 normalize_2d: bool = True,
                 augment: bool = False,
                 augment_config: Optional[Dict] = None):

        self.data_dir = Path(data_dir)
        self.split = split
        self.actions = actions
        self.transform = transform
        self.root_relative = root_relative
        self.normalize_2d = normalize_2d
        self.augment = augment

        # 默认增强配置
        self.augment_config = augment_config or {
            'coord_noise_std': 0.005,   # 坐标噪声标准差
            'dropout_prob': 0.02,       # 随机丢点概率
            'conf_noise_std': 0.05,     # 置信度噪声
            'scale_range': (0.95, 1.05), # 缩放范围
            'flip_prob': 0.5,           # 水平翻转概率
        }

        # 初始化工具类
        self.utils = AP3DDataUtils(data_dir)

        # 建立文件映射
        print(f"Building AP3D {split} dataset with actions={actions}...")
        self.mapping = self.utils.build_frame81_index_map(
            actions=actions,
            split=split
        )

        self.file_paths = self.mapping['file_paths']
        self.num_samples = len(self.file_paths)

        print(f"Dataset initialized: {self.num_samples} samples")

        # 预加载数据到内存（如果数据量不大）
        self._cache = {}
        self._use_cache = self.num_samples < 5000  # 小于5000个样本时使用缓存

        if self._use_cache:
            print("Pre-loading data to memory...")
            self._preload_data()

    def _preload_data(self):
        """预加载所有数据到内存"""
        for i, path in enumerate(self.file_paths):
            with open(path, 'rb') as f:
                self._cache[i] = pickle.load(f)
            if (i + 1) % 500 == 0:
                print(f"  Loaded {i + 1}/{self.num_samples} samples")
        print(f"  Pre-loading complete: {len(self._cache)} samples in memory")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Returns:
            pose_2d: (81, 17, 3) 2D 关键点 tensor
            pose_3d: (81, 17, 3) 3D 关键点 tensor
        """
        # 加载数据
        if self._use_cache and idx in self._cache:
            sample = self._cache[idx]
        else:
            with open(self.file_paths[idx], 'rb') as f:
                sample = pickle.load(f)

        data_input = sample['data_input'].astype(np.float32)  # (81, 17, 3)
        data_label = sample['data_label'].astype(np.float32)  # (81, 17, 3)

        # 归一化 2D 输入
        if self.normalize_2d:
            data_input = self.utils.normalize_2d_for_model(data_input)

        # 转换为 root-relative 3D 坐标
        if self.root_relative:
            data_label = self.utils.make_root_relative(data_label)

        # 数据增强
        if self.augment and self.split == 'train':
            data_input, data_label = self._apply_augmentation(data_input, data_label)

        # 自定义变换
        if self.transform is not None:
            data_input = self.transform(data_input)

        # 转换为 tensor
        pose_2d = torch.from_numpy(data_input)
        pose_3d = torch.from_numpy(data_label)

        return pose_2d, pose_3d

    def _apply_augmentation(self,
                            data_2d: np.ndarray,
                            data_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用数据增强

        Args:
            data_2d: (T, 17, 3) 2D 数据
            data_3d: (T, 17, 3) 3D 数据

        Returns:
            增强后的 (data_2d, data_3d)
        """
        cfg = self.augment_config

        # 1. 坐标噪声（仅对 2D）
        if cfg.get('coord_noise_std', 0) > 0:
            data_2d = self.utils.simulate_mediapipe_noise(
                data_2d,
                coord_noise_std=cfg['coord_noise_std'],
                dropout_prob=cfg.get('dropout_prob', 0),
                conf_noise_std=cfg.get('conf_noise_std', 0)
            )

        # 2. 随机缩放（同时应用于 2D 和 3D 的 x, y）
        if 'scale_range' in cfg:
            scale = np.random.uniform(*cfg['scale_range'])
            data_2d[:, :, :2] *= scale
            data_3d[:, :, :2] *= scale

        # 3. 水平翻转
        if np.random.rand() < cfg.get('flip_prob', 0):
            data_2d, data_3d = self._horizontal_flip(data_2d, data_3d)

        return data_2d, data_3d

    def _horizontal_flip(self,
                         data_2d: np.ndarray,
                         data_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        水平翻转（需要交换左右关节）

        H36M 关节映射:
        - 左右髋: 4 <-> 1
        - 左右膝: 5 <-> 2
        - 左右踝: 6 <-> 3
        - 左右肩: 11 <-> 14
        - 左右肘: 12 <-> 15
        - 左右腕: 13 <-> 16
        """
        # 翻转 x 坐标
        data_2d[:, :, 0] = 1.0 - data_2d[:, :, 0]  # 假设 x 在 [0, 1] 范围
        data_3d[:, :, 0] = -data_3d[:, :, 0]       # 对于 root-relative，取反

        # 交换左右关节
        swap_pairs = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]
        for left, right in swap_pairs:
            data_2d[:, [left, right], :] = data_2d[:, [right, left], :]
            data_3d[:, [left, right], :] = data_3d[:, [right, left], :]

        return data_2d, data_3d


def create_ap3d_dataloaders(
    data_dir: str,
    actions: Optional[List[str]] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    augment_train: bool = True,
    root_relative: bool = True,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    创建 AP3D 训练和验证 DataLoader

    Args:
        data_dir: pose_3d_v3 目录路径
        actions: 筛选的 action 类型，如 ['rm']
        batch_size: 批次大小
        num_workers: 数据加载进程数
        augment_train: 是否对训练集进行增强
        root_relative: 是否使用 root-relative 坐标
        train_ratio: 训练集占比
        seed: 随机种子

    Returns:
        (train_loader, val_loader)
    """
    # 创建完整数据集
    full_dataset = AP3DFrame81Dataset(
        data_dir=data_dir,
        split='train',
        actions=actions,
        root_relative=root_relative,
        augment=False  # 先不增强，后面单独处理
    )

    # 计算分割点
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    print(f"Splitting dataset: {train_size} train, {val_size} val")

    # 使用随机分割
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    # 为训练集包装增强
    if augment_train:
        # 创建增强版数据集
        train_dataset = AP3DFrame81DatasetWrapper(
            train_subset,
            augment=True,
            augment_config={
                'coord_noise_std': 0.005,
                'dropout_prob': 0.02,
                'conf_noise_std': 0.05,
                'flip_prob': 0.3,
            }
        )
    else:
        train_dataset = train_subset

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, val_loader


class AP3DFrame81DatasetWrapper(Dataset):
    """
    用于在 Subset 上应用数据增强的包装器
    """

    def __init__(self,
                 subset: torch.utils.data.Subset,
                 augment: bool = True,
                 augment_config: Optional[Dict] = None):
        self.subset = subset
        self.augment = augment
        self.augment_config = augment_config or {}
        self.utils = AP3DDataUtils.__new__(AP3DDataUtils)  # 只需要静态方法

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        pose_2d, pose_3d = self.subset[idx]

        if self.augment:
            pose_2d_np = pose_2d.numpy()
            pose_3d_np = pose_3d.numpy()

            # 应用增强
            cfg = self.augment_config

            # 坐标噪声
            if cfg.get('coord_noise_std', 0) > 0:
                pose_2d_np = AP3DDataUtils.simulate_mediapipe_noise(
                    pose_2d_np,
                    coord_noise_std=cfg['coord_noise_std'],
                    dropout_prob=cfg.get('dropout_prob', 0),
                    conf_noise_std=cfg.get('conf_noise_std', 0)
                )

            pose_2d = torch.from_numpy(pose_2d_np)
            pose_3d = torch.from_numpy(pose_3d_np)

        return pose_2d, pose_3d


def run_dataloader_tests():
    """运行 DataLoader 测试"""
    print("=" * 70)
    print("AP3D DataLoader Tests")
    print("=" * 70)

    data_dir = r'E:\PycharmProjects\running\pose_3d_v3'

    # 测试 1: 创建数据集
    print("\n[Test 1] Create AP3DFrame81Dataset (rm only)")
    try:
        dataset = AP3DFrame81Dataset(
            data_dir=data_dir,
            split='train',
            actions=['rm'],
            root_relative=True,
            augment=False
        )
        print(f"  [OK] Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        return

    # 测试 2: 获取单个样本
    print("\n[Test 2] Get single sample")
    try:
        pose_2d, pose_3d = dataset[0]
        print(f"  pose_2d shape: {pose_2d.shape}, dtype: {pose_2d.dtype}")
        print(f"  pose_3d shape: {pose_3d.shape}, dtype: {pose_3d.dtype}")
        print(f"  pose_2d range: [{pose_2d.min():.4f}, {pose_2d.max():.4f}]")
        print(f"  pose_3d range: [{pose_3d.min():.4f}, {pose_3d.max():.4f}]")
        print(f"  pose_3d hip (should be ~0): {pose_3d[0, 0]}")
        print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试 3: 创建 DataLoader
    print("\n[Test 3] Create DataLoader")
    try:
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch_2d, batch_3d = next(iter(loader))
        print(f"  Batch pose_2d shape: {batch_2d.shape}")
        print(f"  Batch pose_3d shape: {batch_3d.shape}")
        print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试 4: 创建带增强的 DataLoader
    print("\n[Test 4] Create augmented DataLoader")
    try:
        aug_dataset = AP3DFrame81Dataset(
            data_dir=data_dir,
            split='train',
            actions=['rm'],
            root_relative=True,
            augment=True
        )
        aug_loader = DataLoader(aug_dataset, batch_size=4, shuffle=True)
        batch_2d, batch_3d = next(iter(aug_loader))
        print(f"  Augmented batch pose_2d shape: {batch_2d.shape}")
        print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试 5: 使用 create_ap3d_dataloaders
    print("\n[Test 5] Create train/val dataloaders")
    try:
        train_loader, val_loader = create_ap3d_dataloaders(
            data_dir=data_dir,
            actions=['rm'],
            batch_size=8,
            augment_train=True,
            train_ratio=0.8
        )
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

        # 检查一个训练批次
        for batch_2d, batch_3d in train_loader:
            print(f"  Train batch shape: {batch_2d.shape}")
            break

        print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n" + "=" * 70)
    print("DataLoader Tests Complete")
    print("=" * 70)


if __name__ == '__main__':
    run_dataloader_tests()
