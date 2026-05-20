import torch
import torch.nn as nn
from config.config import MODEL_CONFIG


class RunningPhaseLSTM(nn.Module):
    """
    改进的跑步阶段分类模型
    增加注意力机制和残差连接
    """

    def __init__(self,
                 input_dim=MODEL_CONFIG['input_dim'],
                 hidden_dim=MODEL_CONFIG['hidden_dim'],
                 num_layers=MODEL_CONFIG['num_layers'],
                 output_dim=MODEL_CONFIG['output_dim'],
                 dropout=MODEL_CONFIG['dropout']):
        super(RunningPhaseLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 输入归一化
        self.input_norm = nn.LayerNorm(input_dim)

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # 双向LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: (batch, sequence_length, input_dim)
        Returns:
            output: (batch, sequence_length, output_dim)
        """
        # 输入归一化
        x = self.input_norm(x)

        # 特征提取
        x = self.feature_extractor(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # 自注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 残差连接
        out = lstm_out + attn_out

        # 分类
        output = self.classifier(out)

        return output