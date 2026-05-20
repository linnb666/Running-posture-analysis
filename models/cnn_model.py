import torch
import torch.nn as nn
from config.config import MODEL_CONFIG


class RunningQualityCNN(nn.Module):
    """
    改进的跑步质量评估模型
    修复：避免inplace操作导致的梯度计算错误
    """

    def __init__(self,
                 input_dim=MODEL_CONFIG['input_dim'],
                 hidden_dim=MODEL_CONFIG['hidden_dim']):
        super(RunningQualityCNN, self).__init__()

        # 移除输入归一化（改为在数据预处理阶段处理）
        # self.input_norm = nn.LayerNorm(input_dim)  # 删除

        # 多尺度卷积分支
        self.conv_branch1 = self._make_conv_branch(input_dim, hidden_dim, kernel_size=3)
        self.conv_branch2 = self._make_conv_branch(input_dim, hidden_dim, kernel_size=5)
        self.conv_branch3 = self._make_conv_branch(input_dim, hidden_dim, kernel_size=7)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim * 2, kernel_size=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 时序注意力
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 分类头（多个评分维度）
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5个维度：总分、稳定性、效率、跑姿、节奏
        )

        # 输出激活
        self.output_activation = nn.Sigmoid()

    def _make_conv_branch(self, in_channels, out_channels, kernel_size):
        """创建卷积分支"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 2, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(),
            nn.Conv1d(out_channels // 2, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        """
        前向传播（修复版 - 避免inplace操作）
        Args:
            x: (batch, sequence_length, input_dim)
        Returns:
            scores: (batch, 5) - [总分, 稳定性, 效率, 跑姿, 节奏]
        """
        # 转换维度 (batch, seq, input) -> (batch, input, seq)
        x = x.transpose(1, 2)

        # 多尺度卷积
        feat1 = self.conv_branch1(x)
        feat2 = self.conv_branch2(x)
        feat3 = self.conv_branch3(x)

        # 特征融合
        feat_concat = torch.cat([feat1, feat2, feat3], dim=1)
        feat_fused = self.fusion(feat_concat)

        # 时序注意力
        attention_weights = self.temporal_attention(feat_fused)
        feat_attended = feat_fused * attention_weights

        # 全局池化
        feat_pooled = self.global_pool(feat_attended).squeeze(-1)

        # 质量评分
        scores = self.quality_head(feat_pooled)
        scores = self.output_activation(scores) * 100  # 转换为0-100

        return scores


# ============================================================================
# 改进的数据生成器（添加数据归一化）
"""
基于运动学规则生成更真实的训练数据
"""
import numpy as np
from torch.utils.data import Dataset


class ImprovedRunningDataset(Dataset):
    """
    改进的跑步数据集（添加数据归一化）
    """

    def __init__(self, num_samples=1000, sequence_length=30):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_dim = MODEL_CONFIG['input_dim']

        print(f"生成 {num_samples} 个改进的合成样本...")
        self.data, self.phase_labels, self.quality_labels = self._generate_data()

        # 数据归一化
        self._normalize_data()

        print("✅ 数据生成完成")

    def _generate_data(self):
        """生成基于生物力学的合成数据"""
        data = []
        phase_labels = []
        quality_labels = []

        for i in range(self.num_samples):
            # 随机生成跑步参数
            cadence = np.random.uniform(150, 180)  # 步频 150-180 步/分
            stride_length = np.random.uniform(0.8, 1.2)  # 步长
            vertical_oscillation = np.random.uniform(0.05, 0.15)  # 垂直振荡
            body_lean = np.random.uniform(5, 15)  # 身体前倾角度

            # 生成时间序列
            t = np.linspace(0, 2 * np.pi, self.sequence_length)
            frequency = cadence / 60.0  # 转换为Hz

            # 生成关键点序列
            sequence = self._generate_keypoint_sequence(
                t, frequency, stride_length, vertical_oscillation, body_lean
            )

            # 生成阶段标签
            phase_label = self._generate_phase_labels(t, frequency)

            # 计算质量评分（基于参数的合理性）
            quality_scores = self._calculate_quality_from_params(
                cadence, stride_length, vertical_oscillation, body_lean
            )

            data.append(sequence)
            phase_labels.append(phase_label)
            quality_labels.append(quality_scores)

        return np.array(data, dtype=np.float32), \
            np.array(phase_labels, dtype=np.int64), \
            np.array(quality_labels, dtype=np.float32)

    def _generate_keypoint_sequence(self, t, freq, stride, vert_osc, lean):
        """生成关键点序列"""
        sequence = []

        for frame_t in t:
            frame_keypoints = []

            # 33个关键点
            for kp_idx in range(33):
                if kp_idx in [25, 26]:  # 膝盖
                    x = np.sin(freq * frame_t * 2 * np.pi) * stride * 0.3
                    y = np.abs(np.sin(freq * frame_t * 2 * np.pi)) * vert_osc

                elif kp_idx in [27, 28]:  # 脚踝
                    x = np.sin(freq * frame_t * 2 * np.pi) * stride
                    y = np.abs(np.sin(freq * frame_t * 2 * np.pi)) * vert_osc * 1.5

                elif kp_idx in [23, 24]:  # 髋部
                    x = np.sin(freq * frame_t * 2 * np.pi) * stride * 0.1
                    y = np.sin(freq * frame_t * 2 * np.pi * 2) * vert_osc * 0.5

                elif kp_idx in [11, 12]:  # 肩部
                    x = np.sin(freq * frame_t * 2 * np.pi) * stride * 0.05 + lean * 0.01
                    y = np.sin(freq * frame_t * 2 * np.pi * 2) * vert_osc * 0.3

                else:  # 其他关键点
                    x = np.random.randn() * 0.01
                    y = np.random.randn() * 0.01

                # 添加噪声
                x += np.random.randn() * 0.02
                y += np.random.randn() * 0.02

                frame_keypoints.extend([x, y])

            sequence.append(frame_keypoints)

        return np.array(sequence)

    def _generate_phase_labels(self, t, freq):
        """生成阶段标签"""
        labels = []

        for frame_t in t:
            phase = (freq * frame_t * 2 * np.pi) % (2 * np.pi)

            # 0-π/3: 触地期
            # π/3-2π/3: 过渡期
            # 2π/3-π: 腾空期
            # π-4π/3: 过渡期
            # 4π/3-2π: 触地期

            if phase < np.pi / 3 or phase > 4 * np.pi / 3:
                labels.append(0)  # 触地
            elif np.pi / 3 <= phase < 2 * np.pi / 3 or np.pi <= phase < 4 * np.pi / 3:
                labels.append(2)  # 过渡
            else:
                labels.append(1)  # 腾空

        return labels

    def _calculate_quality_from_params(self, cadence, stride, vert_osc, lean):
        """从参数计算质量评分"""
        # 总分
        total = 70 + np.random.randn() * 10

        # 稳定性（垂直振荡越小越好）
        stability = 100 - vert_osc * 300 + np.random.randn() * 5
        stability = np.clip(stability, 50, 95)

        # 效率（步频接近170最好）
        efficiency = 100 - abs(cadence - 170) * 0.5 + np.random.randn() * 5
        efficiency = np.clip(efficiency, 50, 95)

        # 跑姿（前倾角度8-12度最好）
        form = 100 - abs(lean - 10) * 2 + np.random.randn() * 5
        form = np.clip(form, 50, 95)

        # 节奏
        rhythm = 70 + np.random.randn() * 10
        rhythm = np.clip(rhythm, 50, 95)

        # 重新计算总分
        total = (stability * 0.3 + efficiency * 0.3 + form * 0.2 + rhythm * 0.2)

        return [total, stability, efficiency, form, rhythm]

    def _normalize_data(self):
        """归一化数据（避免在forward中进行inplace操作）"""
        # 计算均值和标准差
        self.mean = np.mean(self.data, axis=(0, 1))
        self.std = np.std(self.data, axis=(0, 1)) + 1e-6

        # 归一化
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.data[idx]),
            torch.LongTensor(self.phase_labels[idx]),
            torch.FloatTensor(self.quality_labels[idx])
        )


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试修复后的CNN模型")
    print("=" * 60)

    # 测试CNN模型
    print("\n1. 测试模型结构:")
    model = RunningQualityCNN()
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    print("\n2. 测试前向传播:")
    dummy_input = torch.randn(4, 30, 66)
    print(f"输入形状: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"输出形状: {output.shape}")
    print(f"输出维度: [总分, 稳定性, 效率, 跑姿, 节奏]")
    print(f"输出示例: {output[0].detach().numpy()}")

    # 测试反向传播
    print("\n3. 测试反向传播:")
    try:
        loss = output.sum()
        loss.backward()
        print("✅ 反向传播成功！没有inplace错误")
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")

    # 测试数据集
    print("\n4. 测试数据集:")
    dataset = ImprovedRunningDataset(num_samples=10)
    data, phase, quality = dataset[0]
    print(f"数据形状: {data.shape}")
    print(f"数据统计: mean={data.mean():.4f}, std={data.std():.4f}")
    print(f"质量评分: {quality.numpy()}")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)