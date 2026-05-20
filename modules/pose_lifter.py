# modules/pose_lifter.py
"""
MotionBERT 2D→3D 姿态提升模块

核心功能：
1. 将MediaPipe 2D关键点转换为Human3.6M格式
2. 使用MotionBERT将2D序列提升为3D序列
3. 在3D空间中计算关节角度

【重要修复】完全重写DSTformer架构以100%匹配官方MotionBERT权重
- 基于checkpoint_weights.txt分析的精确权重结构
- 每个block包含独立的spatial和temporal attention (norm1_s/t, attn_s/t, norm2_s/t, mlp_s/t)
- head逐关节点应用，输出维度为3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from functools import partial
import warnings
import math

# ============================================================================
# MotionBERT 官方架构实现 (DSTformer) - 100%匹配官方权重
# ============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer - 匹配官方命名 fc1, fc2"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self attention - 匹配官方命名 qkv, proj"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DSTBlock(nn.Module):
    """
    DSTformer Block - 100%匹配官方权重命名

    每个block包含独立的spatial和temporal attention:
    - norm1_s, attn_s, norm2_s, mlp_s (spatial)
    - norm1_t, attn_t, norm2_t, mlp_t (temporal)
    """
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, st_mode='st'):
        super().__init__()
        self.st_mode = st_mode  # 'st' = spatial-first, 'ts' = temporal-first

        # Spatial attention components
        self.norm1_s = norm_layer(dim)
        self.attn_s = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_s = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_s = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                         act_layer=act_layer, drop=drop)

        # Temporal attention components
        self.norm1_t = norm_layer(dim)
        self.attn_t = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_t = norm_layer(dim)
        self.mlp_t = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                         act_layer=act_layer, drop=drop)

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward_spatial(self, x):
        """Spatial attention across joints at each frame"""
        B, T, J, C = x.shape
        x = x.reshape(B * T, J, C)
        x = x + self.drop_path(self.attn_s(self.norm1_s(x)))
        x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        x = x.reshape(B, T, J, C)
        return x

    def forward_temporal(self, x):
        """Temporal attention across frames for each joint"""
        B, T, J, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * J, T, C)  # (B*J, T, C)
        x = x + self.drop_path(self.attn_t(self.norm1_t(x)))
        x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        x = x.reshape(B, J, T, C).permute(0, 2, 1, 3)  # (B, T, J, C)
        return x

    def forward(self, x):
        """Forward pass - order depends on st_mode"""
        if self.st_mode == 'st':
            # Spatial first, then Temporal
            x = self.forward_spatial(x)
            x = self.forward_temporal(x)
        else:  # 'ts'
            # Temporal first, then Spatial
            x = self.forward_temporal(x)
            x = self.forward_spatial(x)
        return x


class DSTformer(nn.Module):
    """
    Dual-Stream Spatial-Temporal Transformer

    100%匹配官方MotionBERT权重结构:
    - blocks_st: Spatial attention先，然后Temporal attention
    - blocks_ts: Temporal attention先，然后Spatial attention
    - ts_attn: 融合两个流的注意力权重
    - head: 逐关节点应用，输出维度为3 (不是51)
    """
    def __init__(self, dim_in=3, dim_out=3, dim_feat=512, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=2., num_joints=17, maxlen=243,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, att_fuse=True):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.dim_rep = dim_rep
        self.num_joints = num_joints
        self.maxlen = maxlen
        self.att_fuse = att_fuse

        # 【官方命名】joints_embed
        self.joints_embed = nn.Linear(dim_in, dim_feat)

        # 【官方命名】位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # 【官方命名】blocks_st - Spatial first, then Temporal
        self.blocks_st = nn.ModuleList([
            DSTBlock(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode='st'
            ) for i in range(depth)
        ])

        # 【官方命名】blocks_ts - Temporal first, then Spatial
        self.blocks_ts = nn.ModuleList([
            DSTBlock(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode='ts'
            ) for i in range(depth)
        ])

        # 【官方命名】ts_attn: 融合两个流的注意力
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([
                nn.Linear(dim_feat * 2, 2) for _ in range(depth)
            ])

        # 【官方命名】norm
        self.norm = norm_layer(dim_feat)

        # 【官方命名】pre_logits - 使用Sequential包装Linear层
        # 权重命名差异在_load_weights中处理: pre_logits.fc.* -> pre_logits.0.*
        if dim_rep:
            self.pre_logits = nn.Sequential(
                nn.Linear(dim_feat, dim_rep),
            )
        else:
            self.pre_logits = nn.Identity()

        # 【官方命名】head - 逐关节点应用，输出3 (xyz坐标)
        self.head = nn.Linear(dim_rep if dim_rep else dim_feat, dim_out)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.temp_embed, std=.02)
        self.apply(self._init_weights_module)

    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_rep=False):
        """
        Forward pass - 100%匹配官方实现

        Args:
            x: Input tensor (B, T, J, C) where C is dim_in (3 for x,y,confidence)
            return_rep: Whether to return representation instead of output

        Returns:
            Output tensor (B, T, J, 3) - 3D coordinates
        """
        B, T, J, C = x.shape

        # 1. Joint embedding
        x = self.joints_embed(x)  # (B, T, J, dim_feat)

        # 2. Add position embeddings
        x = x + self.pos_embed  # broadcast over B and T
        x = x + self.temp_embed[:, :T, :, :]  # temporal position

        x = self.pos_drop(x)

        # 3. Dual-stream processing with attention fusion
        for i in range(len(self.blocks_st)):
            # Stream 1 (ST): Spatial first, then Temporal
            x_st = self.blocks_st[i](x)

            # Stream 2 (TS): Temporal first, then Spatial
            x_ts = self.blocks_ts[i](x)

            # Attention fusion
            if self.att_fuse:
                # Concatenate and compute fusion weights
                alpha = torch.cat([x_st, x_ts], dim=-1)  # (B, T, J, dim_feat*2)
                alpha = self.ts_attn[i](alpha)  # (B, T, J, 2)
                alpha = alpha.softmax(dim=-1)
                x = x_st * alpha[..., 0:1] + x_ts * alpha[..., 1:2]
            else:
                x = (x_st + x_ts) * 0.5

        # 4. Normalize
        x = self.norm(x)  # (B, T, J, dim_feat)

        # 5. Get representation (per-joint)
        x = self.pre_logits(x)  # (B, T, J, dim_rep)

        if return_rep:
            return x

        # 6. Output head: predict 3D coordinates per joint
        x = self.head(x)  # (B, T, J, 3)

        return x


# ============================================================================
# 关键点映射：MediaPipe → Human3.6M
# ============================================================================

class KeypointMapper:
    """
    MediaPipe 33关键点 → Human3.6M 17关键点映射
    """

    # Human3.6M 17个关键点定义
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

    # 【修复】短名称映射，用于kinematic_analyzer兼容
    H36M_SHORT_NAMES = {
        'hip': 'hip',
        'right_hip': 'r_hip',
        'right_knee': 'r_knee',
        'right_ankle': 'r_ankle',
        'left_hip': 'l_hip',
        'left_knee': 'l_knee',
        'left_ankle': 'l_ankle',
        'spine': 'spine',
        'thorax': 'thorax',
        'neck': 'neck',
        'head': 'head',
        'left_shoulder': 'l_shoulder',
        'left_elbow': 'l_elbow',
        'left_wrist': 'l_wrist',
        'right_shoulder': 'r_shoulder',
        'right_elbow': 'r_elbow',
        'right_wrist': 'r_wrist'
    }

    # MediaPipe关键点索引
    MP_INDICES = {
        'nose': 0,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
    }

    @classmethod
    def mediapipe_to_h36m(cls, mp_keypoints: Dict) -> np.ndarray:
        """
        将MediaPipe关键点转换为H36M格式

        Args:
            mp_keypoints: MediaPipe格式的关键点字典

        Returns:
            H36M格式的17个关键点 (17, 3) - (x, y, confidence)
            【修复】官方MotionBERT期望3维输入
        """
        landmarks = mp_keypoints.get('landmarks', [])
        if not landmarks or len(landmarks) < 33:
            return np.zeros((17, 3), dtype=np.float32)

        h36m = np.zeros((17, 3), dtype=np.float32)

        def get_point(idx):
            """获取点坐标和置信度"""
            if idx < len(landmarks):
                lm = landmarks[idx]
                conf = lm.get('visibility', 1.0)
                return np.array([lm['x_norm'], lm['y_norm'], conf])
            return np.zeros(3)

        def get_point_xy(idx):
            """只获取xy坐标（用于计算中点）"""
            if idx < len(landmarks):
                lm = landmarks[idx]
                return np.array([lm['x_norm'], lm['y_norm']])
            return np.zeros(2)

        def get_conf(idx):
            """获取置信度"""
            if idx < len(landmarks):
                return landmarks[idx].get('visibility', 1.0)
            return 0.0

        # 直接映射的关节
        h36m[1] = get_point(24)   # right_hip
        h36m[2] = get_point(26)   # right_knee
        h36m[3] = get_point(28)   # right_ankle
        h36m[4] = get_point(23)   # left_hip
        h36m[5] = get_point(25)   # left_knee
        h36m[6] = get_point(27)   # left_ankle
        h36m[11] = get_point(11)  # left_shoulder
        h36m[12] = get_point(13)  # left_elbow
        h36m[13] = get_point(15)  # left_wrist
        h36m[14] = get_point(12)  # right_shoulder
        h36m[15] = get_point(14)  # right_elbow
        h36m[16] = get_point(16)  # right_wrist

        # 计算的关节（中点）- xy坐标取平均，confidence取最小值
        # hip (center of left_hip and right_hip)
        h36m[0, :2] = (h36m[1, :2] + h36m[4, :2]) / 2
        h36m[0, 2] = min(h36m[1, 2], h36m[4, 2])

        # spine (between hip and thorax)
        thorax_xy = (h36m[11, :2] + h36m[14, :2]) / 2
        h36m[7, :2] = (h36m[0, :2] + thorax_xy) / 2
        h36m[7, 2] = min(h36m[0, 2], h36m[11, 2], h36m[14, 2])

        # thorax (shoulder center)
        h36m[8, :2] = thorax_xy
        h36m[8, 2] = min(h36m[11, 2], h36m[14, 2])

        # neck (use nose position)
        h36m[9] = get_point(0)

        # head (use nose position)
        h36m[10] = get_point(0)

        return h36m

    @classmethod
    def batch_mediapipe_to_h36m(cls, keypoints_sequence: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量转换MediaPipe关键点序列到H36M格式

        Args:
            keypoints_sequence: MediaPipe关键点序列

        Returns:
            h36m_sequence: (T, 17, 3) H36M格式序列 (x, y, confidence)
            valid_mask: (T,) 有效帧标记
        """
        T = len(keypoints_sequence)
        h36m_sequence = np.zeros((T, 17, 3), dtype=np.float32)  # 【修复】改为3维
        valid_mask = np.zeros(T, dtype=bool)

        for i, kp in enumerate(keypoints_sequence):
            if kp.get('detected', False):
                h36m_sequence[i] = cls.mediapipe_to_h36m(kp)
                valid_mask[i] = True

        return h36m_sequence, valid_mask

    @classmethod
    def h36m_to_analysis_format(cls, h36m_pose: np.ndarray,
                                 visibility: np.ndarray = None) -> Dict:
        """
        将H36M格式的3D姿态转换为分析友好的字典格式

        Args:
            h36m_pose: (17, 3) H36M格式3D关键点
            visibility: (17,) 可选的原始2D visibility值

        Returns:
            字典格式的3D姿态，使用短名称（如 'l_hip', 'r_knee'）
            如果提供visibility，会额外包含 '{joint}_vis' 字段
        """
        result = {}
        for i, joint_name in enumerate(cls.H36M_JOINTS):
            short_name = cls.H36M_SHORT_NAMES[joint_name]
            result[short_name] = h36m_pose[i]
            # 保留原始2D visibility信息（用于可视化判断数据可靠性）
            if visibility is not None:
                result[f'{short_name}_vis'] = float(visibility[i])
        return result


# ============================================================================
# PoseLifter: 主接口
# ============================================================================

class PoseLifter:
    """
    2D → 3D姿态提升器

    使用MotionBERT DSTformer将2D关键点序列提升为3D
    """

    def __init__(self, checkpoint_path: str = None, device: str = 'auto'):
        """
        初始化PoseLifter

        Args:
            checkpoint_path: MotionBERT权重文件路径
            device: 推理设备 ('auto', 'cuda', 'cpu')
        """
        from config.config import MOTIONBERT_CONFIG

        self.config = MOTIONBERT_CONFIG['model']
        self.device = self._setup_device(device)
        self.mapper = KeypointMapper()

        # 获取权重路径
        if checkpoint_path is None:
            checkpoint_path = MOTIONBERT_CONFIG['checkpoint_path']

        # 创建模型 - 使用正确的参数
        self.model = DSTformer(
            dim_in=self.config['dim_in'],
            dim_out=self.config['dim_out'],
            dim_feat=self.config['dim_feat'],
            dim_rep=self.config['dim_rep'],
            depth=self.config['depth'],
            num_heads=self.config['num_heads'],
            mlp_ratio=2.0,  # 【修复】官方使用2.0，不是4.0
            num_joints=self.config['num_joints'],
            maxlen=self.config['maxlen'],
            qkv_bias=self.config.get('qkv_bias', True),
            drop_rate=self.config.get('drop_rate', 0.),
            attn_drop_rate=self.config.get('attn_drop_rate', 0.),
            drop_path_rate=self.config.get('drop_path_rate', 0.),
            att_fuse=self.config.get('att_fuse', True)
        )

        self.model.to(self.device)
        self.model.eval()

        # 加载权重
        if Path(checkpoint_path).exists():
            self._load_weights(checkpoint_path)
        else:
            print(f"   [WARNING] Weights file not found: {checkpoint_path}")
            print("   Model will use random initialization (unreliable results)")

    def _setup_device(self, device: str) -> torch.device:
        """设置推理设备"""
        if device == 'cpu':
            print("   PoseLifter设备: cpu (手动指定)")
            return torch.device('cpu')

        cuda_available = torch.cuda.is_available()

        if not cuda_available:
            print("   [INFO] CUDA not available, using CPU")
            print("   Tip: For GPU acceleration, install CUDA PyTorch:")
            print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
            return torch.device('cpu')

        if device == 'auto' or device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   [OK] GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"   PoseLifter device: cuda")
            return torch.device('cuda')

        return torch.device('cpu')

    def _load_weights(self, path: str):
        """加载预训练权重 - 支持多种checkpoint格式"""
        checkpoint = torch.load(path, map_location=self.device)

        # 支持多种checkpoint格式:
        # 1. 官方MotionBERT: 'model_pos' 或 'model' 键
        # 2. AP3D微调版本: 'model_state_dict' 键
        # 3. 直接state_dict
        if 'model_pos' in checkpoint:
            state_dict = checkpoint['model_pos']
            print(f"   Loading from 'model_pos' key")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   Loading from 'model_state_dict' key (AP3D format)")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"   Loading from 'model' key")
        else:
            state_dict = checkpoint
            print(f"   Loading state_dict directly")

        # 清理权重键名（移除 'module.' 前缀）
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                cleaned_state_dict[k[7:]] = v
            else:
                cleaned_state_dict[k] = v

        # 处理 pre_logits.fc 命名差异
        # 官方: pre_logits.fc.weight -> 我们的: pre_logits.0.weight
        renamed_state_dict = {}
        for k, v in cleaned_state_dict.items():
            if k == 'pre_logits.fc.weight':
                renamed_state_dict['pre_logits.0.weight'] = v
            elif k == 'pre_logits.fc.bias':
                renamed_state_dict['pre_logits.0.bias'] = v
            else:
                renamed_state_dict[k] = v
        cleaned_state_dict = renamed_state_dict

        # 详细检查权重匹配情况
        model_state = self.model.state_dict()
        matched = 0
        missing = []
        unexpected = []
        shape_mismatch = []

        for k, v in cleaned_state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    model_state[k] = v
                    matched += 1
                else:
                    shape_mismatch.append(f"{k}: checkpoint={v.shape}, model={model_state[k].shape}")
            else:
                unexpected.append(k)

        for k in model_state.keys():
            found = False
            for ck in cleaned_state_dict.keys():
                if ck == k:
                    found = True
                    break
            if not found:
                missing.append(k)

        # 加载匹配的权重
        self.model.load_state_dict(model_state, strict=False)

        # 打印详细信息
        print(f"   [OK] Loaded weights: {path}")
        print(f"   Matched weights: {matched}/{len(model_state)} ({matched/len(model_state)*100:.1f}%)")

        if missing:
            print(f"   [WARN] Missing weights in model: {len(missing)}")
            if len(missing) <= 10:
                for m in missing:
                    print(f"      - {m}")

        if unexpected:
            print(f"   [WARN] Extra weights in checkpoint: {len(unexpected)}")
            if len(unexpected) <= 10:
                for u in unexpected:
                    print(f"      - {u}")

        if shape_mismatch:
            print(f"   [WARN] Shape mismatch: {len(shape_mismatch)}")
            for s in shape_mismatch[:10]:
                print(f"      - {s}")

        # 检查权重匹配率
        match_rate = matched / len(model_state) * 100
        if match_rate < 50:
            print(f"\n   [ERROR] Weight match rate too low ({match_rate:.1f}%)!")
            print("   Model architecture incompatible with weights, output may be incorrect")

    @torch.no_grad()
    def lift_2d_to_3d(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        将2D关键点序列提升为3D

        Args:
            keypoints_2d: (T, 17, 3) 2D关键点序列 (x, y, confidence)

        Returns:
            poses_3d: (T, 17, 3) 3D姿态序列
        """
        from config.config import MOTIONBERT_CONFIG

        T, J, C = keypoints_2d.shape
        maxlen = self.config.get('maxlen', 243)  # 模型最大序列长度

        # 情况1: 序列长度 <= maxlen，直接处理（可能需要填充）
        if T <= maxlen:
            return self._lift_single_chunk(keypoints_2d, T)

        # 情况2: 序列长度 > maxlen，分块处理
        # 使用滑动窗口，重叠部分使用线性插值混合（保留更多高频细节）
        overlap = maxlen // 4  # 25%重叠
        stride = maxlen - overlap

        poses_3d = np.zeros((T, J, 3), dtype=np.float32)
        filled = np.zeros(T, dtype=bool)  # 标记已填充的帧

        start = 0
        chunk_idx = 0
        prev_chunk_3d = None
        prev_end = 0

        while start < T:
            end = min(start + maxlen, T)
            chunk_len = end - start

            # 提取块
            chunk = keypoints_2d[start:end]

            # 处理块
            chunk_3d = self._lift_single_chunk(chunk, chunk_len)

            if chunk_idx == 0:
                # 第一个块，直接填充
                poses_3d[start:end] = chunk_3d
                filled[start:end] = True
            else:
                # 后续块：重叠区域使用线性插值混合
                overlap_start = start
                overlap_end = prev_end

                if overlap_end > overlap_start:
                    overlap_len = overlap_end - overlap_start
                    # 创建线性插值权重：从0到1
                    blend_weights = np.linspace(0, 1, overlap_len)[:, np.newaxis, np.newaxis]

                    # 混合重叠区域：线性过渡，保留更多动态细节
                    poses_3d[overlap_start:overlap_end] = (
                        poses_3d[overlap_start:overlap_end] * (1 - blend_weights) +
                        chunk_3d[:overlap_len] * blend_weights
                    )

                # 填充非重叠区域
                if overlap_end < end:
                    poses_3d[overlap_end:end] = chunk_3d[overlap_end - start:]
                    filled[overlap_end:end] = True

            prev_end = end
            chunk_idx += 1

            # 如果已经处理到末尾，退出
            if end >= T:
                break

            start += stride

        return poses_3d

    def _correct_3d_with_2d(self, poses_3d: np.ndarray, poses_2d: np.ndarray,
                            view_angle: str = 'side',
                            blend_ratio: float = 0.7) -> np.ndarray:
        """
        使用2D信息校正3D输出（后处理）

        核心思想：
        - 3D模型的价值 = 深度估计(Z坐标) + 骨骼比例关系
        - 2D检测的价值 = 精确的平面位置(XY坐标)
        - 使用混合策略：blend_ratio * 2D + (1-blend_ratio) * 3D

        Args:
            poses_3d: (T, 17, 3) 3D提升结果
            poses_2d: (T, 17, 3) 2D输入 (x, y, confidence)
            view_angle: 'side' 或 'front'
            blend_ratio: 2D权重 (0-1)，默认0.7表示70%使用2D位置

        Returns:
            校正后的3D姿态 (T, 17, 3)
        """
        T, J, _ = poses_3d.shape
        corrected = poses_3d.copy()

        # 下肢关节索引（需要重点校正）
        lower_body_joints = [1, 2, 3, 4, 5, 6]  # r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle

        # 首先计算全局缩放因子（使用所有帧的中位数，更稳定）
        scales = []
        for t in range(T):
            p3d = poses_3d[t]
            p2d = poses_2d[t, :, :2]

            hip_2d = (p2d[1] + p2d[4]) / 2
            hip_3d = p3d[0]

            trunk_2d = np.linalg.norm(p2d[8] - hip_2d)
            trunk_3d = np.linalg.norm(p3d[8, :2] - hip_3d[:2])

            if trunk_2d > 0.01 and trunk_3d > 0.001:
                scales.append(trunk_3d / trunk_2d)

        if not scales:
            return corrected  # 无法计算缩放，返回原始数据

        # 使用中位数作为全局缩放因子（抗异常值）
        global_scale = np.median(scales)

        for t in range(T):
            # 获取当前帧数据
            p3d = poses_3d[t]
            p2d = poses_2d[t, :, :2]
            conf = poses_2d[t, :, 2]  # 置信度

            # 计算髋部中心
            hip_2d = (p2d[1] + p2d[4]) / 2
            hip_3d = p3d[0]

            # 对下肢关节进行校正
            for joint_idx in lower_body_joints:
                # 检查2D置信度，低置信度时减少校正
                joint_conf = conf[joint_idx]
                if joint_conf < 0.3:
                    continue  # 置信度太低，不校正

                # 计算2D的相对位置（相对于髋部中心）
                relative_2d = p2d[joint_idx] - hip_2d

                # 缩放到3D空间
                scaled_xy = relative_2d * global_scale

                # 计算2D建议的位置
                pos_2d_suggested = np.array([
                    hip_3d[0] + scaled_xy[0],
                    hip_3d[1] + scaled_xy[1],
                    p3d[joint_idx, 2]  # Z保持3D的值
                ])

                # 根据置信度调整混合比例
                effective_blend = blend_ratio * joint_conf

                if view_angle == 'side':
                    # 侧面视角：混合X和Y，保留Z
                    corrected[t, joint_idx, 0] = (
                        effective_blend * pos_2d_suggested[0] +
                        (1 - effective_blend) * p3d[joint_idx, 0]
                    )
                    corrected[t, joint_idx, 1] = (
                        effective_blend * pos_2d_suggested[1] +
                        (1 - effective_blend) * p3d[joint_idx, 1]
                    )
                    # Z保持不变
                else:
                    # 正面视角：主要校正Y（垂直位置）
                    corrected[t, joint_idx, 1] = (
                        effective_blend * pos_2d_suggested[1] +
                        (1 - effective_blend) * p3d[joint_idx, 1]
                    )

        return corrected

    def _lift_single_chunk(self, chunk: np.ndarray, original_len: int) -> np.ndarray:
        """处理单个块的3D提升"""
        T, J, C = chunk.shape
        maxlen = self.config.get('maxlen', 243)

        # 如果块长度小于maxlen，需要填充
        if T < maxlen:
            padded = np.zeros((maxlen, J, C), dtype=np.float32)
            padded[:T] = chunk
            chunk = padded

        # 转换为tensor
        x = torch.from_numpy(chunk).float().to(self.device)
        x = x.unsqueeze(0)  # Add batch dimension: (1, T, J, C)

        # 推理
        output = self.model(x)  # (1, maxlen, J, 3)

        # 转换回numpy并移除填充
        poses_3d = output[0, :original_len].cpu().numpy()

        return poses_3d

    def process_sequence(self, keypoints_sequence: List[Dict],
                         view_angle: str = 'side',
                         apply_2d_correction: bool = True) -> Dict:
        """
        处理MediaPipe关键点序列，输出3D姿态

        Args:
            keypoints_sequence: MediaPipe格式的关键点序列
            view_angle: 视角类型 ('side' 或 'front')
            apply_2d_correction: 是否应用2D校正（用2D的XY + 3D的Z）

        Returns:
            包含3D姿态数据的字典
        """
        # 1. 转换为H36M格式
        h36m_2d, valid_mask = self.mapper.batch_mediapipe_to_h36m(keypoints_sequence)

        if not valid_mask.any():
            return {
                'has_3d': False,
                'poses_3d': None,
                'keypoints_3d': []
            }

        # 【保存原始2D visibility】用于可视化时判断数据可靠性
        visibility_2d = h36m_2d[:, :, 2].copy()  # (T, 17)

        # 2. 2D到3D提升
        poses_3d = self.lift_2d_to_3d(h36m_2d)

        # 3. 【新增】应用2D校正（保留3D的Z，用2D的XY）
        if apply_2d_correction:
            poses_3d = self._correct_3d_with_2d(poses_3d, h36m_2d, view_angle)

        # 4. 转换为分析格式（传递visibility信息）
        keypoints_3d = []
        for i, (pose_3d, valid) in enumerate(zip(poses_3d, valid_mask)):
            if valid:
                # 传递原始2D visibility用于可视化
                kp_dict = self.mapper.h36m_to_analysis_format(
                    pose_3d, visibility=visibility_2d[i]
                )
                keypoints_3d.append({
                    'frame_idx': i,
                    'has_3d': True,
                    'pose_3d': kp_dict
                })
            else:
                keypoints_3d.append({
                    'frame_idx': i,
                    'has_3d': False,
                    'pose_3d': {}
                })

        return {
            'has_3d': True,
            'poses_3d': poses_3d,
            'keypoints_3d': keypoints_3d
        }

    def lift_sequence(self, keypoints_sequence: List[Dict],
                      view_angle: str = 'side',
                      apply_2d_correction: bool = True) -> Dict:
        """
        处理MediaPipe关键点序列，输出3D姿态（兼容旧接口）

        Args:
            keypoints_sequence: MediaPipe格式的关键点序列
            view_angle: 视角类型 ('side' 或 'front')
            apply_2d_correction: 是否应用2D校正（用2D的XY + 3D的Z）

        Returns:
            包含3D姿态数据的字典，兼容pose_estimator调用格式
        """
        # 1. 转换为H36M格式
        h36m_2d, valid_mask = self.mapper.batch_mediapipe_to_h36m(keypoints_sequence)

        if not valid_mask.any():
            return {
                'success': False,
                'keypoints_3d': [],
                'valid_frames_ratio': 0.0
            }

        # 【保存原始2D visibility】用于可视化时判断数据可靠性
        visibility_2d = h36m_2d[:, :, 2].copy()  # (T, 17)

        # 2. 2D到3D提升
        poses_3d = self.lift_2d_to_3d(h36m_2d)

        # 3. 【新增】应用2D校正（保留3D的Z，用2D的XY）
        if apply_2d_correction:
            poses_3d = self._correct_3d_with_2d(poses_3d, h36m_2d, view_angle)

        # 4. 转换为pose_estimator期望的格式（传递visibility信息）
        keypoints_3d = []
        valid_count = 0
        for i, (pose_3d, valid) in enumerate(zip(poses_3d, valid_mask)):
            if valid:
                valid_count += 1
                keypoints_3d.append({
                    'frame_idx': i,
                    'detected': True,
                    'keypoints_h36m': pose_3d,  # (17, 3) array
                    'pose_3d': self.mapper.h36m_to_analysis_format(
                        pose_3d, visibility=visibility_2d[i]
                    )
                })
            else:
                keypoints_3d.append({
                    'frame_idx': i,
                    'detected': False,
                    'keypoints_h36m': np.zeros((17, 3), dtype=np.float32),
                    'pose_3d': {}
                })

        valid_ratio = valid_count / len(keypoints_sequence) if keypoints_sequence else 0.0

        return {
            'success': True,
            'keypoints_3d': keypoints_3d,
            'valid_frames_ratio': valid_ratio,
            'poses_3d': poses_3d  # 额外提供原始数组
        }

    def cleanup(self):
        """
        清理GPU资源

        在不再需要PoseLifter时调用此方法释放显存
        """
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None

        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("   PoseLifter resources cleaned up")

    def __del__(self):
        """析构函数：确保GPU资源被释放"""
        try:
            self.cleanup()
        except Exception:
            pass  # 析构时静默处理异常


# ============================================================================
# 3D角度计算工具
# ============================================================================

def calculate_3d_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三个3D点形成的角度（p2为顶点）

    Args:
        p1, p2, p3: 3D坐标点

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


def calculate_knee_angles_3d(pose_3d: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    从3D姿态计算膝关节角度

    Args:
        pose_3d: 3D姿态字典（使用短名称如 'l_hip', 'l_knee', 'l_ankle'）

    Returns:
        (left_knee_angle, right_knee_angle) 元组
    """
    left_angle = None
    right_angle = None

    # 左膝角度
    if all(k in pose_3d for k in ['l_hip', 'l_knee', 'l_ankle']):
        left_angle = calculate_3d_angle(
            pose_3d['l_hip'],
            pose_3d['l_knee'],
            pose_3d['l_ankle']
        )

    # 右膝角度
    if all(k in pose_3d for k in ['r_hip', 'r_knee', 'r_ankle']):
        right_angle = calculate_3d_angle(
            pose_3d['r_hip'],
            pose_3d['r_knee'],
            pose_3d['r_ankle']
        )

    return left_angle, right_angle
