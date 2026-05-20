"""
基于Transformer的跑步阶段分类模型

特点：
1. 视角感知 - 融合视角信息作为条件输入
2. 位置编码 - 学习时序位置关系
3. 多头自注意力 - 捕捉长距离依赖
4. 相对位置编码 - 增强时序建模能力

适用于毕业设计：基于深度学习的跑步动作视频解析与技术质量评价系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config.config import MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    为序列中的每个位置添加位置信息
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ViewAwareEmbedding(nn.Module):
    """
    视角感知嵌入层
    将视角信息融合到特征表示中
    """

    def __init__(self, d_model: int, num_views: int = 4):
        """
        Args:
            d_model: 模型维度
            num_views: 视角数量 (side=0, front=1, back=2, mixed=3)
        """
        super().__init__()
        self.view_embedding = nn.Embedding(num_views, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, view_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (batch, seq_len, d_model)
            view_ids: 视角ID (batch,) 取值 0-3
        Returns:
            融合视角信息的特征 (batch, seq_len, d_model)
        """
        # 获取视角嵌入
        view_emb = self.view_embedding(view_ids)  # (batch, d_model)
        view_emb = view_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch, seq_len, d_model)

        # 融合
        combined = torch.cat([x, view_emb], dim=-1)  # (batch, seq_len, d_model*2)
        return self.fusion(combined)


class RelativePositionBias(nn.Module):
    """
    相对位置偏置
    增强模型对相对时序关系的建模能力
    """

    def __init__(self, num_heads: int, max_len: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * max_len - 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 相对位置索引
        coords = torch.arange(max_len)
        relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)  # (max_len, max_len)
        relative_coords = relative_coords + max_len - 1  # 偏移到正数
        self.register_buffer('relative_position_index', relative_coords)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: 序列长度
        Returns:
            相对位置偏置 (1, num_heads, seq_len, seq_len)
        """
        relative_position_index = self.relative_position_index[:seq_len, :seq_len].contiguous()
        relative_position_bias = self.relative_position_bias_table[relative_position_index.reshape(-1)].reshape(
            seq_len, seq_len, -1
        )
        return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

class TransformerEncoderLayer(nn.Module):
    """
    改进的Transformer编码器层
    - 使用Pre-LN结构（更稳定）
    - 支持相对位置偏置
    - GELU激活函数
    """

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int,
                 dropout: float = 0.1, max_len: int = 128):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.relative_position_bias = RelativePositionBias(num_heads, max_len)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Pre-LN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            src_mask: 注意力掩码
        Returns:
            (batch, seq_len, d_model)
        """
        # 获取相对位置偏置
        seq_len = x.size(1)
        rel_pos_bias = self.relative_position_bias(seq_len)

        # Pre-LN + Self-Attention
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=src_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_output)

        # Pre-LN + FFN
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x


class RunningPhaseTransformer(nn.Module):
    """
    基于Transformer的跑步阶段分类模型

    架构：
    1. 输入嵌入层 - 将关键点坐标映射到高维空间
    2. 视角感知嵌入 - 融合视角条件信息
    3. 位置编码 - 添加时序位置信息
    4. Transformer编码器 - 多层自注意力
    5. 分类头 - 输出每帧的阶段预测

    输入：(batch, seq_len, 66) - 33个关键点 × 2D坐标
    输出：(batch, seq_len, 3) - 3个阶段的logits
    """

    def __init__(self,
                 input_dim: int = MODEL_CONFIG['input_dim'],
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 num_classes: int = 3,
                 max_len: int = 128):
        super().__init__()

        self.d_model = d_model

        # 输入嵌入
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 视角感知嵌入
        self.view_embedding = ViewAwareEmbedding(d_model, num_views=4)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, max_len)
            for _ in range(num_layers)
        ])

        # 输出层归一化
        self.output_norm = nn.LayerNorm(d_model)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor,
                view_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列 (batch, seq_len, input_dim)
            view_ids: 视角ID (batch,)，取值 0=side, 1=front, 2=back, 3=mixed
                      如果为None，默认使用side视角

        Returns:
            logits: (batch, seq_len, num_classes)
        """
        batch_size = x.size(0)

        # 默认视角
        if view_ids is None:
            view_ids = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        # 输入嵌入
        x = self.input_embedding(x)

        # 视角感知嵌入
        x = self.view_embedding(x, view_ids)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer编码器
        for layer in self.encoder_layers:
            x = layer(x)

        # 输出归一化
        x = self.output_norm(x)

        # 分类
        logits = self.classifier(x)

        return logits

    def predict(self, x: torch.Tensor,
                view_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测阶段

        Args:
            x: 输入序列 (batch, seq_len, input_dim)
            view_ids: 视角ID (batch,)

        Returns:
            predictions: 预测的阶段 (batch, seq_len)
            probabilities: 各阶段概率 (batch, seq_len, num_classes)
        """
        logits = self.forward(x, view_ids)
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities


class GaitPhaseTransformerWithCRF(nn.Module):
    """
    带条件随机场(CRF)的步态阶段分类模型

    在Transformer基础上添加CRF层，建模阶段之间的转移约束：
    - 触地 -> 过渡 (合理)
    - 过渡 -> 腾空 (合理)
    - 腾空 -> 过渡 (合理)
    - 触地 -> 腾空 (不合理，需惩罚)
    """

    def __init__(self,
                 input_dim: int = MODEL_CONFIG['input_dim'],
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 num_classes: int = 3,
                 max_len: int = 128):
        super().__init__()

        # Transformer骨干网络
        self.transformer = RunningPhaseTransformer(
            input_dim, d_model, num_heads, num_layers,
            dim_feedforward, dropout, num_classes, max_len
        )

        # CRF转移矩阵
        self.num_classes = num_classes
        self.transitions = nn.Parameter(torch.zeros(num_classes, num_classes))

        # 初始化转移矩阵（基于步态先验知识）
        self._init_transitions()

    def _init_transitions(self):
        """初始化转移矩阵（基于步态生物力学先验）"""
        # 状态: 0=触地, 1=腾空, 2=过渡
        # 合理转移给高分，不合理转移给低分
        with torch.no_grad():
            # 自转移（同一状态持续）
            self.transitions[0, 0] = 1.0  # 触地 -> 触地
            self.transitions[1, 1] = 1.0  # 腾空 -> 腾空
            self.transitions[2, 2] = 0.5  # 过渡 -> 过渡（持续时间短）

            # 合理转移
            self.transitions[0, 2] = 1.0  # 触地 -> 过渡
            self.transitions[2, 1] = 1.0  # 过渡 -> 腾空
            self.transitions[1, 2] = 1.0  # 腾空 -> 过渡
            self.transitions[2, 0] = 1.0  # 过渡 -> 触地

            # 不合理转移（惩罚）
            self.transitions[0, 1] = -1.0  # 触地 -> 腾空（跳过过渡）
            self.transitions[1, 0] = -1.0  # 腾空 -> 触地（跳过过渡）

    def forward(self, x: torch.Tensor,
                view_ids: Optional[torch.Tensor] = None,
                tags: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列 (batch, seq_len, input_dim)
            view_ids: 视角ID (batch,)
            tags: 真实标签 (batch, seq_len)，训练时使用

        Returns:
            如果提供tags，返回负对数似然损失
            否则返回emission scores
        """
        emissions = self.transformer(x, view_ids)  # (batch, seq_len, num_classes)

        if tags is not None:
            # 训练模式：计算CRF损失
            return self._compute_loss(emissions, tags)
        else:
            # 推理模式：返回发射分数
            return emissions

    def _compute_loss(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """计算CRF负对数似然损失"""
        batch_size, seq_len, num_classes = emissions.shape

        # 计算正确路径的分数
        score = self._score_sentence(emissions, tags)

        # 计算所有路径的分数（前向算法）
        forward_score = self._forward_algorithm(emissions)

        # 负对数似然
        loss = forward_score - score
        return loss.mean()

    def _score_sentence(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """计算给定标签序列的分数"""
        batch_size, seq_len, _ = emissions.shape

        # 发射分数
        score = torch.zeros(batch_size, device=emissions.device)
        for i in range(seq_len):
            score += emissions[torch.arange(batch_size), i, tags[:, i]]
            if i > 0:
                score += self.transitions[tags[:, i - 1], tags[:, i]]

        return score

    def _forward_algorithm(self, emissions: torch.Tensor) -> torch.Tensor:
        """前向算法计算配分函数"""
        batch_size, seq_len, num_classes = emissions.shape

        # 初始化
        alpha = emissions[:, 0, :]  # (batch, num_classes)

        for i in range(1, seq_len):
            alpha_expand = alpha.unsqueeze(2)  # (batch, num_classes, 1)
            emit = emissions[:, i, :].unsqueeze(1)  # (batch, 1, num_classes)
            trans = self.transitions.unsqueeze(0)  # (1, num_classes, num_classes)

            scores = alpha_expand + trans + emit  # (batch, num_classes, num_classes)
            alpha = torch.logsumexp(scores, dim=1)  # (batch, num_classes)

        return torch.logsumexp(alpha, dim=1)

    def decode(self, x: torch.Tensor,
               view_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Viterbi解码

        Args:
            x: 输入序列 (batch, seq_len, input_dim)
            view_ids: 视角ID (batch,)

        Returns:
            最优路径 (batch, seq_len)
        """
        emissions = self.transformer(x, view_ids)
        return self._viterbi_decode(emissions)

    def _viterbi_decode(self, emissions: torch.Tensor) -> torch.Tensor:
        """Viterbi解码找最优路径"""
        batch_size, seq_len, num_classes = emissions.shape

        # 初始化
        viterbi = emissions[:, 0, :]  # (batch, num_classes)
        backpointers = []

        for i in range(1, seq_len):
            viterbi_expand = viterbi.unsqueeze(2)  # (batch, num_classes, 1)
            emit = emissions[:, i, :].unsqueeze(1)  # (batch, 1, num_classes)
            trans = self.transitions.unsqueeze(0)  # (1, num_classes, num_classes)

            scores = viterbi_expand + trans + emit  # (batch, num_classes, num_classes)
            viterbi, bp = torch.max(scores, dim=1)  # (batch, num_classes)
            backpointers.append(bp)

        # 回溯
        best_path = [torch.argmax(viterbi, dim=1)]
        for bp in reversed(backpointers):
            best_tag = bp[torch.arange(batch_size), best_path[-1]]
            best_path.append(best_tag)

        best_path.reverse()
        return torch.stack(best_path, dim=1)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 Transformer 跑步阶段分类模型")
    print("=" * 70)

    # 测试基础Transformer模型
    print("\n1. 测试 RunningPhaseTransformer:")
    model = RunningPhaseTransformer()
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建测试输入
    batch_size, seq_len = 4, 30
    x = torch.randn(batch_size, seq_len, 66)
    view_ids = torch.tensor([0, 1, 2, 3])  # 4种视角各一个

    # 前向传播
    logits = model(x, view_ids)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {logits.shape}")

    # 预测
    preds, probs = model.predict(x, view_ids)
    print(f"   预测形状: {preds.shape}")
    print(f"   概率形状: {probs.shape}")

    # 测试带CRF的模型
    print("\n2. 测试 GaitPhaseTransformerWithCRF:")
    crf_model = GaitPhaseTransformerWithCRF()
    print(f"   参数量: {sum(p.numel() for p in crf_model.parameters()):,}")

    # 创建标签
    tags = torch.randint(0, 3, (batch_size, seq_len))

    # 训练模式（计算损失）
    loss = crf_model(x, view_ids, tags)
    print(f"   CRF损失: {loss.item():.4f}")

    # 推理模式（Viterbi解码）
    best_path = crf_model.decode(x, view_ids)
    print(f"   最优路径形状: {best_path.shape}")
    print(f"   示例路径: {best_path[0].tolist()}")

    # 测试反向传播
    print("\n3. 测试反向传播:")
    loss.backward()
    print("   ✅ 反向传播成功!")

    print("\n" + "=" * 70)
    print("✅ 所有测试通过!")
    print("=" * 70)

