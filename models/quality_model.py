"""
跑步质量评估深度学习模型

架构特点：
1. 多尺度时序卷积 - 捕捉不同时间尺度的运动模式
2. 视角感知 - 根据视角调整评估策略
3. 注意力机制 - 聚焦关键时刻
4. 多任务输出 - 同时评估多个质量维度
5. 不确定性估计 - 输出置信度

适用于毕业设计：基于深度学习的跑步动作视频解析与技术质量评价系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
from config.config import MODEL_CONFIG


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation模块
    通道注意力机制，自适应地重新校准通道特征
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, seq_len)
        """
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class TemporalBlock(nn.Module):
    """
    时序卷积块
    包含：膨胀卷积 + 残差连接 + SE注意力
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout(dropout)

        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        residual = self.residual(x)

        out = self.dropout(F.gelu(self.bn1(self.conv1(x))))
        out = self.dropout(F.gelu(self.bn2(self.conv2(out))))
        out = self.se(out)

        return out + residual


class MultiScaleTCN(nn.Module):
    """
    多尺度时序卷积网络
    使用不同膨胀率捕捉多尺度时序特征
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_levels: int = 4, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # 多尺度时序卷积
        self.levels = nn.ModuleList()
        for i in range(num_levels):
            dilation = 2 ** i  # 1, 2, 4, 8
            self.levels.append(
                TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout)
            )

        # 多尺度融合
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * num_levels, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, hidden_dim, seq_len)
        """
        # 转换维度
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)

        # 输入投影
        x = self.input_proj(x)

        # 多尺度特征
        features = []
        for level in self.levels:
            x = level(x)
            features.append(x)

        # 融合
        multi_scale = torch.cat(features, dim=1)
        return self.fusion(multi_scale)


class TemporalAttentionPooling(nn.Module):
    """
    时序注意力池化
    学习每个时间步的重要性权重
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, hidden_dim, seq_len)
        Returns:
            pooled: (batch, hidden_dim)
            attention_weights: (batch, seq_len)
        """
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        # 计算注意力权重
        attn_scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1)

        # 加权求和
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)

        return pooled, attn_weights


class ViewConditionedHead(nn.Module):
    """
    视角条件化的评估头
    根据视角调整评估权重
    """

    def __init__(self, hidden_dim: int, num_views: int = 4, num_outputs: int = 5):
        super().__init__()

        # 视角嵌入
        self.view_embedding = nn.Embedding(num_views, hidden_dim // 4)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_outputs)
        )

    def forward(self, x: torch.Tensor, view_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 特征 (batch, hidden_dim)
            view_ids: 视角ID (batch,)
        Returns:
            评分 (batch, num_outputs)
        """
        view_emb = self.view_embedding(view_ids)
        combined = torch.cat([x, view_emb], dim=-1)
        fused = self.fusion(combined)
        return self.output_head(fused)


class RunningQualityModel(nn.Module):
    """
    跑步质量评估模型（升级版）

    架构：
    1. 多尺度时序卷积网络 - 提取时序特征
    2. 时序注意力池化 - 聚焦关键帧
    3. 视角条件化评估头 - 根据视角调整评估

    输入：
        x: (batch, seq_len, 66) - 关键点序列
        view_ids: (batch,) - 视角ID

    输出：
        scores: (batch, 5) - [总分, 稳定性, 效率, 跑姿, 节奏]
        attention_weights: (batch, seq_len) - 注意力权重
    """

    def __init__(self,
                 input_dim: int = MODEL_CONFIG['input_dim'],
                 hidden_dim: int = 128,
                 num_levels: int = 4,
                 dropout: float = 0.2):
        super().__init__()

        # 多尺度时序卷积
        self.tcn = MultiScaleTCN(input_dim, hidden_dim, num_levels, dropout=dropout)

        # 时序注意力池化
        self.attention_pool = TemporalAttentionPooling(hidden_dim)

        # 视角条件化评估头
        self.quality_head = ViewConditionedHead(hidden_dim, num_views=4, num_outputs=5)

        # 输出激活
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor,
                view_ids: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入序列 (batch, seq_len, input_dim)
            view_ids: 视角ID (batch,)，默认为side视角
            return_attention: 是否返回注意力权重

        Returns:
            字典包含:
            - scores: (batch, 5) 质量评分
            - attention_weights: (batch, seq_len) 可选
        """
        batch_size = x.size(0)

        # 默认视角
        if view_ids is None:
            view_ids = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        # 多尺度时序特征
        features = self.tcn(x)  # (batch, hidden_dim, seq_len)

        # 注意力池化
        pooled, attn_weights = self.attention_pool(features)  # (batch, hidden_dim)

        # 视角条件化评估
        raw_scores = self.quality_head(pooled, view_ids)  # (batch, 5)

        # 转换为0-100分
        scores = self.output_activation(raw_scores) * 100

        result = {'scores': scores}
        if return_attention:
            result['attention_weights'] = attn_weights

        return result


class RunningQualityModelWithUncertainty(nn.Module):
    """
    带不确定性估计的跑步质量评估模型

    使用MC Dropout估计预测不确定性
    """

    def __init__(self,
                 input_dim: int = MODEL_CONFIG['input_dim'],
                 hidden_dim: int = 128,
                 num_levels: int = 4,
                 dropout: float = 0.2):
        super().__init__()

        self.base_model = RunningQualityModel(input_dim, hidden_dim, num_levels, dropout)
        self.dropout = nn.Dropout(dropout)
        self.num_mc_samples = 10

    def forward(self, x: torch.Tensor,
                view_ids: Optional[torch.Tensor] = None,
                estimate_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入序列 (batch, seq_len, input_dim)
            view_ids: 视角ID (batch,)
            estimate_uncertainty: 是否估计不确定性

        Returns:
            字典包含:
            - scores: (batch, 5) 质量评分
            - uncertainty: (batch, 5) 不确定性估计（可选）
        """
        if not estimate_uncertainty:
            return self.base_model(x, view_ids)

        # MC Dropout采样
        self.train()  # 启用dropout
        samples = []
        for _ in range(self.num_mc_samples):
            with torch.no_grad():
                result = self.base_model(x, view_ids)
                samples.append(result['scores'])

        samples = torch.stack(samples, dim=0)  # (num_samples, batch, 5)

        # 计算均值和不确定性
        mean_scores = samples.mean(dim=0)
        uncertainty = samples.std(dim=0)

        self.eval()

        return {
            'scores': mean_scores,
            'uncertainty': uncertainty
        }


# ============================================================================
# 联合模型：同时进行阶段分类和质量评估
# ============================================================================

class JointPhaseQualityModel(nn.Module):
    """
    联合阶段分类和质量评估模型

    共享底层特征提取，分别进行：
    1. 阶段分类（逐帧）
    2. 质量评估（全局）

    优势：
    - 共享特征提取，减少计算量
    - 多任务学习，相互增强
    """

    def __init__(self,
                 input_dim: int = MODEL_CONFIG['input_dim'],
                 hidden_dim: int = 128,
                 num_levels: int = 4,
                 num_phases: int = 3,
                 num_quality_dims: int = 5,
                 dropout: float = 0.2):
        super().__init__()

        # 共享的多尺度时序卷积
        self.shared_tcn = MultiScaleTCN(input_dim, hidden_dim, num_levels, dropout=dropout)

        # 视角嵌入（共享）
        self.view_embedding = nn.Embedding(4, hidden_dim // 4)

        # 阶段分类头（逐帧）
        self.phase_head = nn.Sequential(
            nn.Conv1d(hidden_dim + hidden_dim // 4, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, num_phases, 1)
        )

        # 质量评估头（全局）
        self.attention_pool = TemporalAttentionPooling(hidden_dim + hidden_dim // 4)
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_quality_dims)
        )

        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor,
                view_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入序列 (batch, seq_len, input_dim)
            view_ids: 视角ID (batch,)

        Returns:
            字典包含:
            - phase_logits: (batch, seq_len, num_phases) 阶段分类logits
            - quality_scores: (batch, num_quality_dims) 质量评分
            - attention_weights: (batch, seq_len) 注意力权重
        """
        batch_size, seq_len, _ = x.size()

        # 默认视角
        if view_ids is None:
            view_ids = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        # 共享特征提取
        features = self.shared_tcn(x)  # (batch, hidden_dim, seq_len)

        # 视角嵌入（扩展到每个时间步）
        view_emb = self.view_embedding(view_ids)  # (batch, hidden_dim//4)
        view_emb = view_emb.unsqueeze(2).expand(-1, -1, seq_len)  # (batch, hidden_dim//4, seq_len)

        # 融合视角信息
        fused_features = torch.cat([features, view_emb], dim=1)  # (batch, hidden_dim + hidden_dim//4, seq_len)

        # 阶段分类
        phase_logits = self.phase_head(fused_features)  # (batch, num_phases, seq_len)
        phase_logits = phase_logits.transpose(1, 2)  # (batch, seq_len, num_phases)

        # 质量评估
        pooled, attn_weights = self.attention_pool(fused_features)
        raw_scores = self.quality_head(pooled)
        quality_scores = self.output_activation(raw_scores) * 100

        return {
            'phase_logits': phase_logits,
            'quality_scores': quality_scores,
            'attention_weights': attn_weights
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                     phase_targets: torch.Tensor,
                     quality_targets: torch.Tensor,
                     phase_weight: float = 1.0,
                     quality_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        计算联合损失

        Args:
            outputs: 模型输出
            phase_targets: 阶段标签 (batch, seq_len)
            quality_targets: 质量标签 (batch, num_quality_dims)
            phase_weight: 阶段分类损失权重
            quality_weight: 质量评估损失权重

        Returns:
            字典包含各损失项
        """
        # 阶段分类损失
        phase_logits = outputs['phase_logits'].reshape(-1, outputs['phase_logits'].size(-1))
        phase_targets_flat = phase_targets.reshape(-1)
        phase_loss = F.cross_entropy(phase_logits, phase_targets_flat)

        # 质量评估损失
        quality_loss = F.mse_loss(outputs['quality_scores'], quality_targets)

        # 总损失
        total_loss = phase_weight * phase_loss + quality_weight * quality_loss

        return {
            'total_loss': total_loss,
            'phase_loss': phase_loss,
            'quality_loss': quality_loss
        }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试跑步质量评估模型")
    print("=" * 70)

    batch_size, seq_len = 4, 30
    x = torch.randn(batch_size, seq_len, 66)
    view_ids = torch.tensor([0, 1, 2, 3])

    # 测试基础质量模型
    print("\n1. 测试 RunningQualityModel:")
    model = RunningQualityModel()
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    result = model(x, view_ids, return_attention=True)
    print(f"   输入形状: {x.shape}")
    print(f"   评分形状: {result['scores'].shape}")
    print(f"   注意力形状: {result['attention_weights'].shape}")
    print(f"   评分示例: {result['scores'][0].detach().numpy()}")

    # 测试带不确定性的模型
    print("\n2. 测试 RunningQualityModelWithUncertainty:")
    uncertainty_model = RunningQualityModelWithUncertainty()
    result = uncertainty_model(x, view_ids, estimate_uncertainty=True)
    print(f"   评分: {result['scores'][0].detach().numpy()}")
    print(f"   不确定性: {result['uncertainty'][0].detach().numpy()}")

    # 测试联合模型
    print("\n3. 测试 JointPhaseQualityModel:")
    joint_model = JointPhaseQualityModel()
    print(f"   参数量: {sum(p.numel() for p in joint_model.parameters()):,}")

    result = joint_model(x, view_ids)
    print(f"   阶段logits形状: {result['phase_logits'].shape}")
    print(f"   质量评分形状: {result['quality_scores'].shape}")
    print(f"   注意力权重形状: {result['attention_weights'].shape}")

    # 测试损失计算
    phase_targets = torch.randint(0, 3, (batch_size, seq_len))
    quality_targets = torch.rand(batch_size, 5) * 100
    losses = joint_model.compute_loss(result, phase_targets, quality_targets)
    print(f"   总损失: {losses['total_loss'].item():.4f}")
    print(f"   阶段损失: {losses['phase_loss'].item():.4f}")
    print(f"   质量损失: {losses['quality_loss'].item():.4f}")

    # 测试反向传播
    print("\n4. 测试反向传播:")
    losses['total_loss'].backward()
    print("   ✅ 反向传播成功!")

    print("\n" + "=" * 70)
    print("✅ 所有测试通过!")
    print("=" * 70)
