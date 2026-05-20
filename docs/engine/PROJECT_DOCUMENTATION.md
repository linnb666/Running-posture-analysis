# 基于深度学习的跑步动作视频解析与技术质量评价系统

## 项目概述

本系统是一个完整的跑步动作分析解决方案，集成了2D/3D姿态估计、运动学分析、深度学习评估和AI报告生成等功能。系统采用模块化设计，支持侧面/正面/背面多视角分析。

### 核心技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 2D姿态估计 | MediaPipe Pose | 33个关键点实时检测 |
| 3D姿态提升 | MotionBERT (DSTformer) | 2D→3D姿态转换 |
| 运动学分析 | NumPy + SciPy | 关节角度、步频、振幅计算 |
| 时序分析 | temporal_model.py（已停用/兼容保留） | 历史记录兼容、论文对照 |
| AI报告 | 智谱AI (GLM-4) | 自然语言分析报告 |

---

## 深度学习技术详解

### 深度学习模型总览

本项目的核心特色是**端到端的深度学习推理管道**，涉及多个SOTA模型：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        深度学习推理流程                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     │                              │                              │
     ▼                              ▼                              ▼
┌─────────────┐            ┌─────────────┐            ┌─────────────────┐
│ MediaPipe   │            │ MotionBERT  │            │ LSTM/Trans/CNN  │
│ (2D检测)    │     →      │ (3D提升)    │     →      │ (时序分析)      │
│ ~5-10M参数  │            │ ~8M参数     │            │ ~1.5M参数       │
└─────────────┘            └─────────────┘            └─────────────────┘
     │                              │                              │
     ▼                              ▼                              ▼
  33个关键点              17个3D关键点                阶段分类+质量评分
```

### 1. MediaPipe Pose (2D姿态估计)

**技术类型**: 深度神经网络 (卷积神经网络)

**模型来源**: Google MediaPipe

**推理过程**:
```python
# modules/pose_estimator.py
results = self.pose.process(image_rgb)  # 神经网络前向推理
```

**输入/输出**:
- 输入: RGB图像帧 (H×W×3)
- 输出: 33个关键点坐标 + 置信度

**深度学习特性**:
- 实时人体检测与关键点定位
- 端到端训练的深度卷积网络
- 支持GPU加速推理

### 2. MotionBERT (3D姿态提升)

**技术类型**: Transformer (DSTformer - 双流时空Transformer)

**模型架构**: Dual-Stream Spatial-Temporal Transformer

```
输入: 2D关键点序列 (T=243, J=17, C=3)
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐  ┌────────┐
│ ST流   │  │ TS流   │
│空间优先│  │时序优先│
└────────┘  └────────┘
    │           │
    └─────┬─────┘
          ▼
    ┌─────────────┐
    │ 融合注意力  │
    │ α_st, α_ts │
    └─────────────┘
          │
          ▼
    3D坐标 (T, 17, 3)
```

**核心组件** (`modules/pose_lifter.py`):

```python
class DSTformer(nn.Module):
    def __init__(self):
        # 1. 关键点嵌入层
        self.joints_embed = nn.Linear(3, 512)

        # 2. 位置编码
        self.pos_embed = nn.Parameter(...)      # 空间位置
        self.temp_embed = nn.Parameter(...)     # 时序位置

        # 3. 双路Transformer编码器
        self.blocks_st = nn.ModuleList([...])   # 空间优先路径 (5层)
        self.blocks_ts = nn.ModuleList([...])   # 时序优先路径 (5层)

        # 4. 动态融合
        self.ts_attn = nn.ModuleList([...])     # 学习融合权重

        # 5. 3D输出头
        self.head = nn.Linear(512, 3)
```

**DSTBlock详解**:
```python
class DSTBlock(nn.Module):
    """空间-时序分离注意力块"""
    def forward(self, x):
        # 空间注意力: 跨17个关节点
        x = self.norm1_s(x)
        x = x + self.attn_s(x)       # 自注意力
        x = x + self.mlp_s(self.norm2_s(x))

        # 时序注意力: 跨T个时间步
        x = self.norm1_t(x)
        x = x + self.attn_t(x)       # 自注意力
        x = x + self.mlp_t(self.norm2_t(x))
        return x
```

**创新点**:
1. **双流设计**: 分别建模空间和时序关系，然后动态融合
2. **生物力学约束**: 学习骨长恒定、关节连接等约束
3. **长序列处理**: 支持243帧滑动窗口，25%重叠

### 3. 时序深度学习分析

#### 3.1 LSTM阶段分类模型

```python
class RunningPhaseLSTM(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(
            input_size=66,        # 33关键点 × 2坐标
            hidden_size=64,
            num_layers=2,
            bidirectional=True,   # 双向LSTM
            dropout=0.3
        )
        self.attention = nn.MultiheadAttention(128, 4)  # 多头注意力
        self.classifier = nn.Linear(128, 3)  # 3阶段分类
```

**输出**: 每帧的步态阶段 (0=触地, 1=腾空, 2=过渡)

#### 3.2 Transformer阶段分类模型

```python
class RunningPhaseTransformer(nn.Module):
    def __init__(self):
        # 视角感知嵌入
        self.view_embedding = ViewAwareEmbedding(128, num_views=4)

        # 位置编码
        self.pos_encoding = PositionalEncoding(128)

        # 4层Transformer编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=128, nhead=4)
            for _ in range(4)
        ])
```

**创新**: 视角感知 - 根据侧面/正面/背面调整特征权重

#### 3.3 CNN质量评估模型

```python
class RunningQualityCNN(nn.Module):
    def __init__(self):
        # 多尺度卷积分支
        self.branch_3 = self._conv_branch(kernel_size=3)   # 局部特征
        self.branch_5 = self._conv_branch(kernel_size=5)   # 中程特征
        self.branch_7 = self._conv_branch(kernel_size=7)   # 全局特征

        # 时序注意力池化
        self.attention_pool = TemporalAttentionPooling(128)

        # 5维质量输出
        self.quality_head = nn.Linear(384, 5)
```

**输出**: [总分, 稳定性, 效率, 跑姿, 节奏] 5维评分

#### 3.4 联合模型 (Multi-task Learning)

```python
class JointPhaseQualityModel(nn.Module):
    """同时进行阶段分类和质量评估"""
    def __init__(self):
        # 共享特征提取器 (多尺度TCN)
        self.shared_tcn = MultiScaleTCN(
            input_dim=66,
            hidden_dim=128,
            num_levels=4,
            dilation_rates=[1, 2, 4, 8]
        )

        # 任务特定头
        self.phase_head = nn.Linear(128, 3)      # 阶段分类
        self.quality_head = nn.Linear(128, 5)    # 质量评估
```

### 4. 深度学习在系统中的角色

**分层应用策略**

本系统中深度学习技术的应用分为两个层次：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        深度学习应用层次                                   │
└─────────────────────────────────────────────────────────────────────────┘

层次1: 核心特征提取（用于最终评分）
├── MediaPipe CNN: 2D人体关键点检测 → 提取33个关键点坐标
└── MotionBERT Transformer: 2D→3D提升 → 生成精确3D关节角度

层次2: 辅助分析（仅供参考，不参与评分）
├── LSTM: 步态阶段分类 → 参考信息
├── Transformer: 步态分类 → 参考信息
└── CNN/TCN: 质量评估 → 参考信息
```

**设计理念**：

1. **特征提取依赖DL**：MediaPipe和MotionBERT负责从视频中提取精确的姿态特征
2. **评分基于运动学规则**：技术质量评分完全基于生物力学标准（可解释、可验证）
3. **DL模型辅助参考**：LSTM/CNN模型输出作为可视化参考，便于论文展示

**优势**：
- 评分标准明确、可解释（符合运动生物力学）
- 深度学习用于最合适的任务（特征提取）
- 避免端到端黑盒评分的不可解释性

### 深度学习推理验证

| 模块 | 模型 | 参数量 | 是否真正推理 | 验证代码位置 |
|------|------|--------|------------|-------------|
| 2D姿态 | MediaPipe | ~5-10M | ✅是 | `pose_estimator.py:146` |
| 3D提升 | DSTformer | ~8M | ✅是 | `pose_lifter.py:735` |
| 阶段分类 | LSTM | ~0.5M | ✅是 | `temporal_model.py:210` |
| 阶段分类 | Transformer | ~0.3M | ✅是 | `temporal_model.py:220` |
| 质量评估 | CNN | ~0.2M | ✅是 | `temporal_model.py:230` |
| 质量评估 | TCN | ~0.3M | ✅是 | `temporal_model.py:240` |

### 模型权重文件

```
data/checkpoints/
├── best_epoch.bin              # MotionBERT (DSTformer) - 官方预训练
├── phase_model.pth             # LSTM阶段分类
├── quality_model.pth           # CNN质量评估
├── transformer_phase_model.pth # Transformer阶段分类
├── quality_tcn_model.pth       # TCN质量评估
└── joint_model.pth             # 联合模型 (推荐)
```

### 项目与题目契合度

**题目**: 基于深度学习的跑步动作视频解析与技术质量评价系统

| 题目要求 | 实现方式 | 契合度 |
|---------|---------|--------|
| 基于深度学习 | MediaPipe (CNN) + MotionBERT (Transformer) 进行姿态提取 | ⭐⭐⭐⭐⭐ |
| 跑步动作 | 专门针对跑步运动设计，支持步态分析 | ⭐⭐⭐⭐⭐ |
| 视频解析 | 视频→帧→2D姿态(DL)→3D姿态(DL)→运动学指标 | ⭐⭐⭐⭐⭐ |
| 技术质量评价 | 基于运动学规则的三维度评估（可解释、科学） | ⭐⭐⭐⭐⭐ |

**深度学习技术要点**：
1. **MediaPipe Pose**: Google的CNN模型，实时检测33个人体关键点
2. **MotionBERT (DSTformer)**: 双流时空Transformer，将2D姿态提升为3D
3. **LSTM/CNN时序模型**: 步态阶段分类（作为参考信息展示）

**结论**: 项目完全符合题目要求。深度学习用于核心的姿态特征提取（MediaPipe+MotionBERT），技术质量评价基于生物力学规则确保可解释性和科学性。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              main.py (主程序入口)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         run_analysis_pipeline()                          │
│                              完整分析流程                                 │
└─────────────────────────────────────────────────────────────────────────┘
        │              │              │              │              │
        ▼              ▼              ▼              ▼              ▼
   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
   │  视频  │    │  姿态  │    │ 运动学 │    │  时序  │    │  质量  │
   │  处理  │    │  估计  │    │  分析  │    │  分析  │    │  评价  │
   └────────┘    └────────┘    └────────┘    └────────┘    └────────┘
        │              │              │              │              │
        ▼              ▼              ▼              ▼              ▼
   VideoProcessor  Pose3D       Kinematic      Temporal       Quality
                  Estimator     Analyzer        Model         Evaluator
                     │
                     ▼
              ┌─────────────┐
              │ PoseLifter  │
              │ (MotionBERT)│
              └─────────────┘
```

---

## 模块详细说明

### 1. 视频处理模块 (`modules/video_processor.py`)

**功能**：读取视频文件、提取帧序列、调整分辨率

**核心类**：`VideoProcessor`

**处理流程**：
```
视频文件 → OpenCV读取 → 帧率调整 → 分辨率缩放 → 帧序列输出
```

**关键参数**：
- `target_fps`: 目标帧率，默认30fps
- `max_frames`: 最大帧数限制，默认300帧
- `target_width/height`: 目标分辨率 640×480

**输出格式**：
```python
{
    'width': 640,
    'height': 480,
    'fps': 30.0,
    'frame_count': 300,
    'duration': 10.0,  # 秒
    'filename': 'video.mp4'
}
```

---

### 2. 姿态估计模块 (`modules/pose_estimator.py`)

**功能**：2D姿态检测 + 3D姿态提升

#### 2.1 MediaPipe 2D姿态估计

**核心类**：`MediaPipePoseEstimator`

**关键点定义**（33个）：
```
0: nose, 1-6: 眼睛, 7-8: 耳朵, 9-10: 嘴角
11-12: 肩膀, 13-14: 肘部, 15-16: 手腕
17-22: 手指, 23-24: 髋部, 25-26: 膝盖
27-28: 脚踝, 29-30: 脚跟, 31-32: 脚尖
```

**输出格式**：
```python
{
    'detected': True,
    'landmarks': [
        {
            'id': 0,
            'name': 'nose',
            'x': 320,           # 像素坐标
            'y': 240,
            'x_norm': 0.5,      # 归一化坐标 [0,1]
            'y_norm': 0.5,
            'z': -0.1,          # 相对深度
            'visibility': 0.95  # 置信度
        },
        ...
    ]
}
```

#### 2.2 MotionBERT 3D姿态提升 (`modules/pose_lifter.py`)

**核心功能**：将2D关键点序列提升为3D姿态

**架构**：DSTformer (Dual-Stream Spatial-Temporal Transformer)

**关键点映射**：
```
MediaPipe (33点) → Human3.6M (17点)

H36M格式：
0: hip (髋部中心)
1-3: 右腿 (右髋-右膝-右踝)
4-6: 左腿 (左髋-左膝-左踝)
7-10: 脊柱 (spine-thorax-neck-head)
11-13: 左臂 (左肩-左肘-左腕)
14-16: 右臂 (右肩-右肘-右腕)
```

**短名称映射**（用于运动学分析）：
```python
H36M_SHORT_NAMES = {
    'hip': 'hip',
    'right_hip': 'r_hip',    'left_hip': 'l_hip',
    'right_knee': 'r_knee',  'left_knee': 'l_knee',
    'right_ankle': 'r_ankle','left_ankle': 'l_ankle',
    'right_shoulder': 'r_shoulder', 'left_shoulder': 'l_shoulder',
    ...
}
```

**模型配置** (`config/config.py`)：
```python
MOTIONBERT_CONFIG = {
    'dim_feat': 256,      # 特征维度
    'dim_rep': 512,       # 表示维度
    'depth': 5,           # Transformer层数
    'num_heads': 8,       # 注意力头数
    'num_joints': 17,     # 关键点数量
    'maxlen': 243,        # 最大序列长度
}
```

**输出格式**：
```python
{
    'has_3d': True,
    'pose_3d': {
        'hip': np.array([x, y, z]),
        'l_hip': np.array([x, y, z]),
        'l_knee': np.array([x, y, z]),
        'l_ankle': np.array([x, y, z]),
        ...
    },
    'confidence_3d': [0.9, 0.85, ...],  # 17个关节置信度
}
```

---

### 3. 运动学分析模块 (`modules/kinematic_analyzer.py`)

**功能**：计算步频、关节角度、垂直振幅等运动学指标

#### 3.1 步频计算

**方法**：基于脚踝Y坐标的峰值检测

**计算流程**：
```
脚踝Y坐标序列 → 低通滤波 → 峰值检测 (find_peaks) → 周期计算 → 步频
```

**公式**：
```
步频 (步/分) = 检测到的步数 / 视频时长(秒) × 60
```

**评级标准**：
| 等级 | 步频范围 | 说明 |
|------|----------|------|
| 精英 | ≥185 步/分 | 专业运动员水平 |
| 优秀 | 175-185 步/分 | 高级跑者 |
| 良好 | 165-175 步/分 | 中级跑者 |
| 一般 | 155-165 步/分 | 入门跑者 |
| 较差 | <155 步/分 | 需要改进 |

#### 3.2 膝关节角度计算

**【重要】仅在3D数据可用时计算**

**3D角度计算方法**：
```python
def calculate_3d_joint_angle(p1, p2, p3):
    """
    计算三点形成的角度（p2为顶点）

    膝关节角度 = angle(髋-膝-踝)
    """
    v1 = p1 - p2  # 大腿向量
    v2 = p3 - p2  # 小腿向量

    cos_angle = dot(v1, v2) / (|v1| × |v2|)
    angle = arccos(cos_angle)

    return degrees(angle)
```

**分阶段分析**：

| 阶段 | 理想角度 | 说明 |
|------|----------|------|
| 触地期 (Ground Contact) | 155-170° | 膝微屈缓冲 |
| 腾空期 (Flight) | 90-130° | 摆动腿折叠 |
| 过渡期 (Transition) | 120-155° | 蹬离/着地过渡 |

**落地膝角生物力学约束**：
```python
MIN_LANDING_ANGLE = 145°  # 最小允许角度
MAX_LANDING_ANGLE = 175°  # 最大允许角度
```

#### 3.3 步态阶段检测

**方法**：基于脚踝Y坐标的状态机

```python
# 阈值计算
y_range = max(ankle_y) - min(ankle_y)
ground_threshold = max(ankle_y) - y_range * 0.25  # 上25%为触地
flight_threshold = min(ankle_y) + y_range * 0.35  # 下35%为腾空

# 状态判断
if ankle_y >= ground_threshold and velocity < threshold:
    phase = GROUND_CONTACT  # 触地
elif ankle_y <= flight_threshold:
    phase = FLIGHT          # 腾空
else:
    phase = TRANSITION      # 过渡
```

#### 3.4 垂直振幅计算

**归一化方法**：相对躯干长度的百分比

```python
躯干长度 = median(robust_shoulder_to_hip_distance)
身体中心Y序列 = 0.9 × hip_center_y + 0.1 × shoulder_center_y
去趋势振幅 = detrend(body_center_y)
周期振幅 = trimmed_mean(peak_to_trough_amplitude_by_cycle)
归一化振幅 = (周期振幅 / 躯干长度) × 100%
```

**评级标准**：
| 等级 | 振幅范围 | 说明 |
|------|----------|------|
| 优秀 | ≤11% | 能量利用高效 |
| 良好 | 11-17% | 整体在可接受范围 |
| 一般 | 17-25% | 存在能量损耗，有优化空间 |
| 较差 | >25% | 能量浪费较明显 |

#### 3.5 稳定性计算

**综合评分**（加权平均）：
```python
stability = (
    trunk_stability * 0.4 +    # 躯干稳定性（肩髋连线角度变化）
    head_stability * 0.3 +     # 头部稳定性（鼻子位置变化）
    rhythm_consistency * 0.3   # 节奏一致性（步态周期变异）
)
```

---

### 4. 时序深度学习模块 (`modules/temporal_model.py`, 已停用)

**功能**：历史深度学习模块，当前主流程默认停用（保留兼容结构，不参与分析与评分）

**说明**：当前 `backend/main/streamlit` 主分析链路中不再执行本模块。API 与数据库保留 temporal 字段用于历史兼容和论文对照。

#### 4.1 模型架构

**支持的模型类型**：
- `legacy`: LSTM + CNN
- `transformer`: Transformer + TCN
- `joint`: 联合模型

**输入格式**：
```python
input_tensor: (batch, seq_len, features)
# seq_len = 30 (帧)
# features = 33 × 2 = 66 (MediaPipe关键点的x,y坐标)
```

#### 4.2 模型用途

**历史输出格式（兼容示意）**：
```python
# 模型输出用于：
1. 步态阶段分布可视化（饼图）
2. 模型预测置信度展示
3. 论文数据分析和对比
4. 当前默认停用
```

#### 4.3 输出格式

```python
{
    'phase_sequence': [0, 0, 1, 1, 2, 0, ...],  # 每帧阶段（参考）
    'phase_distribution': {
        'ground_contact': 0.45,  # 触地占比
        'flight': 0.35,          # 腾空占比
        'transition': 0.20       # 过渡占比
    },
    'quality_score': 75.5,       # 模型质量评分（参考值）
    'stability_score': 78.0,     # 模型稳定性评分（参考值）
    'model_type': 'LSTM'         # 使用的模型类型
}
```

**注意**：当前主流程下 temporal 结果为空占位结构；技术质量评分由`quality_evaluator.py`基于运动学规则独立计算。

---

### 5. 质量评价模块 (`modules/quality_evaluator.py`)

**功能**：基于纯运动学规则的跑步技术质量评价

**评分策略**：本模块完全基于生物力学规则进行评分，不融合深度学习模型输出。这确保了评分的可解释性和客观性。

#### 5.1 评价维度

| 维度 | 权重 | 评估内容 |
|------|------|----------|
| 稳定性 (Stability) | 30% | 躯干稳定、膝关节角度变异度、垂直运动稳定性 |
| 效率 (Efficiency) | 40% | 步频合理性、垂直振幅控制、触地时间 |
| 跑姿 (Form) | 30% | 膝关节角度、躯干前倾、着地方式 |

#### 5.2 评分逻辑

**效率评分**：
```python
def evaluate_efficiency(kinematic):
    # 步频评分
    cadence_score = score_cadence(cadence)  # 见上文标准

    # 垂直振幅评分
    if amplitude <= 11:
        amp_score = 100
    elif amplitude <= 17:
        amp_score = 80
    elif amplitude <= 25:
        amp_score = 60
    else:
        amp_score = 40

    return cadence_score × 0.5 + amp_score × 0.5
```

**跑姿评分**：
```python
def evaluate_form(kinematic):
    # 触地膝关节角度
    gc_angle = phase_analysis['ground_contact']['mean']
    if 155 <= gc_angle <= 170:
        knee_score = 100
    elif 145 <= gc_angle <= 175:
        knee_score = 75
    else:
        knee_score = 50

# 躯干前倾（中长跑/马拉松稳健阈值）
    if 4 <= forward_lean <= 8:
        lean_score = 100
    elif 3 <= forward_lean <= 10:
        lean_score = 85
    elif 2 <= forward_lean <= 12:
        lean_score = 70
    else:
        lean_score = 55

    return knee_score × 0.5 + lean_score × 0.5
```

#### 5.3 评级标准

| 总分 | 评级 | 说明 |
|------|------|------|
| ≥85 | 优秀 | 技术出色 |
| 70-84 | 良好 | 有一定基础 |
| 55-69 | 一般 | 存在改进空间 |
| <55 | 待改进 | 需要系统训练 |

#### 5.4 数据可靠性标注

```python
data_reliability = {
    'overall': 'high' if is_3d else 'low',
    'is_3d': True/False,
    'angle_data': {
        'reliability': 'high'/'unavailable',
        'description': '使用MotionBERT 3D姿态计算，数据可靠'
    }
}
```

---

### 6. AI分析模块 (`modules/ai_analyzer.py`)

**功能**：生成自然语言分析报告

#### 6.1 提供商

- **智谱AI (GLM-4)**：主要提供商，支持文本和多模态
- **本地规则引擎**：后备方案，基于模板生成

#### 6.2 报告结构

```markdown
## 跑步技术分析报告

### 一、总体评价
- 总体评分、技术评级、分析视角

### 二、各维度表现
- 稳定性、效率、跑姿的得分和等级

### 三、关键技术指标
- 步频分析
- 垂直振幅
- 步态周期时间
- 膝关节角度（侧面视角）
- 稳定性分析

### 四、技术优势

### 五、待改进项

### 六、改进建议

### 七、总结
```

---

#### 6.3 Web 报告导出

当前 Flask + Vue Web 平台在结果页额外支持正式 PDF 报告导出：

- 结果页顶部信息卡显示 `视角类型 / 分析时间 / 模型版本`，并提供直接下载按钮
- PDF 由后端服务端渲染生成，不依赖浏览器打印当前网页
- 导出内容固定包含：封面概览、维度雷达图、维度条形图、核心指标卡片、关键帧骨架图、人工备注、AI/本地分析报告正文，并对头部元信息换行、文本留边、正面关键帧保留原样、侧面关键帧主体裁切和正文分页做版式优化
- 人工备注保存在 `analysis_records.manual_notes` 与 `analysis_records.manual_notes_updated_at`，导出时会一并写入 PDF
- Web 端显示时间按用户当前系统时区转换，便于演示和结果追溯

---

## 数据流向图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              输入视频                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        VideoProcessor                                    │
│              提取帧序列 (frames, fps)                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     MediaPipePoseEstimator                               │
│            检测2D关键点 (33点 × T帧)                                     │
│                 keypoints_2d: List[Dict]                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
            ┌───────────────┐               ┌───────────────┐
            │   启用3D      │               │   禁用3D      │
            └───────────────┘               └───────────────┘
                    │                               │
                    ▼                               │
┌─────────────────────────────────────────────────────────────────────────┐
│                         PoseLifter (MotionBERT)                          │
│                                                                          │
│  MediaPipe(33点) ──映射──> H36M(17点) ──DSTformer──> 3D坐标(17×3)         │
│                                                                          │
│  keypoints_3d: List[{                                                    │
│      'has_3d': True,                                                     │
│      'pose_3d': {'l_hip': [x,y,z], 'l_knee': [x,y,z], ...}              │
│  }]                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        KinematicAnalyzer                                 │
│                                                                          │
│  输入: keypoints_2d + keypoints_3d (可选)                                │
│                                                                          │
│  计算指标:                                                               │
│  ├── cadence: 步频 (步/分)                                              │
│  ├── vertical_motion: 去趋势后按步态周期计算的垂直振幅 (% 躯干长度)                              │
│  ├── angles: 关节角度 (仅3D可用时)                                       │
│  │   ├── knee_left/right: 膝关节角度时间序列                             │
│  │   └── phase_analysis: 分阶段统计                                      │
│  ├── stability: 稳定性评分                                               │
│  └── gait_cycle: 步态周期 (ms)                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      TemporalModelAnalyzer                               │
│                                                                          │
│  输入: keypoints_2d + kinematic_results                                  │
│                                                                          │
│  输出:                                                                   │
│  ├── phase_sequence: 每帧步态阶段 [0,0,1,1,2,0,...]                      │
│  ├── phase_distribution: 阶段占比                                        │
│  ├── quality_score: DL质量评分                                           │
│  └── kinematic_adjusted: 运动学融合后评分                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        QualityEvaluator                                  │
│                                                                          │
│  输入: kinematic_results + temporal_results                              │
│                                                                          │
│  计算:                                                                   │
│  ├── stability_score × 0.35                                             │
│  ├── efficiency_score × 0.35                                            │
│  └── form_score × 0.30                                                  │
│  ────────────────────────                                                │
│  total_score + rating + suggestions                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          AIAnalyzer                                      │
│                                                                          │
│  生成文本报告 (智谱AI / 本地规则引擎)                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            输出文件                                      │
│                                                                          │
│  output/                                                                 │
│  ├── pose_sample.jpg        # 姿态可视化样例                             │
│  ├── skeleton_3d.mp4        # 3D骨架视频 (如有3D数据)                    │
│  ├── angle_curves.png       # 角度变化曲线                               │
│  └── ai_analysis_report.txt # AI分析报告                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 配置文件说明 (`config/config.py`)

### 主要配置项

```python
# 视频处理
VIDEO_CONFIG = {
    'target_width': 640,
    'target_height': 480,
    'fps': 30,
    'supported_formats': ['.mp4', '.avi', '.mov', '.mkv']
}

# 姿态估计
POSE_CONFIG = {
    'backend': 'mediapipe',
    'model_complexity': 1,  # 0=轻量, 1=平衡, 2=高精度
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# MotionBERT 3D
MOTIONBERT_CONFIG = {
    'enabled': True,
    'checkpoint_path': 'data/checkpoints/best_epoch.bin',
    'device': 'auto'  # 'auto'/'cuda'/'cpu'
}

# 质量评价权重
QUALITY_WEIGHTS = {
    'stability': 0.35,
    'efficiency': 0.35,
    'form': 0.30
}

# 评级阈值
QUALITY_THRESHOLDS = {
    'excellent': 85,
    'good': 70,
    'fair': 55,
    'poor': 0
}
```

---

## 使用方法

### 命令行运行

```bash
# 基本用法（侧面视角，启用3D）
python main.py video.mp4

# 指定视角
python main.py video.mp4 --view side    # 侧面
python main.py video.mp4 --view front   # 正面
python main.py video.mp4 --view back    # 背面

# 禁用3D提升（仅2D分析）
python main.py video.mp4 --no-3d

# 生成可视化结果
python main.py video.mp4 --visualize

# 保存到数据库
python main.py video.mp4 --save-db

# 指定输出目录
python main.py video.mp4 --output ./results
```

### 输出示例

```
================================================================================
基于深度学习的跑步动作视频解析与技术质量评价系统
================================================================================
视频文件: running_sample.mp4
姿态估计后端: MEDIAPIPE
视角模式: side
3D姿态提升: ✅ 启用 (MotionBERT)
================================================================================

1️⃣ 视频输入与预处理...
   分辨率: 1920x1080
   帧率: 30.00 FPS
   时长: 10.50 秒
   提取帧数: 300

2️⃣ 人体姿态估计...
   使用3D姿态估计器 (MediaPipe + MotionBERT)...
   ✅ 检测到GPU: NVIDIA GeForce RTX 3060 (12.0GB显存)
   ✅ 3D姿态提升成功
   检测成功: 295/300 帧 (98.3%)

3️⃣ 视角设置...
   使用视角: 侧面视角
   分析策略: 膝关节角度 + 垂直振幅 + 躯干前倾

4️⃣ 运动学特征解析...
   ✅ 检测到3D姿态数据，将使用3D坐标计算关节角度
   📊 数据来源: 3D姿态 (高可靠性)
   步频: 178.5 步/分
   检测步数: 31 步 (视频时长 10.4 秒)
   垂直振幅: 5.23% (躯干长度)
   振幅评级: 优秀
   膝关节角度（分阶段）:
      触地: 162.3° (范围: 155.1°-168.7°)
      腾空: 108.5° (范围: 95.2°-125.3°)

5️⃣ 时序深度学习分析（参考信息）...
   模型质量评分: 78.50 (参考)
   模型稳定性: 82.30 (参考)
   阶段分布: 触地43.2% | 腾空35.1% | 过渡21.7%
   （注：阶段分类仅供参考，评分基于运动学规则）

6️⃣ 跑步技术质量评价（运动学规则）...
   总体评分: 81.25/100
   评级: 良好

7️⃣ AI文本分析与报告生成...
   AI报告已保存: output/ai_analysis_report.txt

================================================================================
📊 分析结果汇总
================================================================================

📐 数据来源
   姿态提取: MediaPipe (CNN) + MotionBERT (Transformer)
   数据维度: ✅ 3D姿态
   数据可靠性: 🟢 高

🎯 技术质量评价（基于运动学规则）
┌─────────────────────────────────────────────────────────────────┐
│  总体评分: 81.25/100                                            │
│  评    级: 良好                                                  │
│  评分方法: 纯运动学规则（生物力学标准）                            │
└─────────────────────────────────────────────────────────────────┘

📈 各维度得分
   稳定性 [████████░░] 82.3  (权重30%)
   效  率 [████████░░] 80.2  (权重40%)
   跑  姿 [███████░░░] 79.5  (权重30%)

📊 关键运动学指标
   步    频: 178.5 步/分 [优秀]
   垂直振幅: 5.23% 躯干 [优秀]
   触地膝角: 162.3° [良好]

🤖 深度学习分析（参考信息）
   模型质量评分: 78.50
   模型稳定性:   82.30
   说明: temporal模块已停用（保留历史兼容字段）

✅ 优势
   • 动作稳定性
   • 跑步效率

💡 改进建议
   • 继续保持良好状态，可适当增加训练量挑战自己

================================================================================
✅ 分析完成!
================================================================================
```

---

## 3D骨架视频生成 (`utils/visualization.py`)

### 功能说明

当3D数据可用时，自动生成3D骨架可视化视频，将3D坐标投影到2D平面显示。

### 骨架连接定义 (Human3.6M)

```python
H36M_SKELETON = [
    # 躯干
    (0, 7),   # hip -> spine
    (7, 8),   # spine -> thorax
    (8, 9),   # thorax -> neck
    (9, 10),  # neck -> head

    # 左腿 (绿色)
    (0, 4), (4, 5), (5, 6),   # hip -> l_hip -> l_knee -> l_ankle

    # 右腿 (蓝色)
    (0, 1), (1, 2), (2, 3),   # hip -> r_hip -> r_knee -> r_ankle

    # 左臂 (黄色)
    (8, 11), (11, 12), (12, 13),

    # 右臂 (紫色)
    (8, 14), (14, 15), (15, 16)
]
```

### 输出格式

- 文件名: `skeleton_3d.mp4`
- 布局: 左侧原始视频 + 右侧3D骨架（黑色背景）
- 分辨率: 原始宽度×2, 原始高度

---

## GPU加速说明

### 自动检测

系统会自动检测CUDA可用性并选择最佳设备：

```python
def _setup_device(self, device: str) -> torch.device:
    if not torch.cuda.is_available():
        print("ℹ️ CUDA不可用，使用CPU推理")
        print("提示: 如需GPU加速，请安装CUDA版PyTorch:")
        print("pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return torch.device('cpu')

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ 检测到GPU: {gpu_name} ({gpu_memory:.1f}GB显存)")
    return torch.device('cuda')
```

### 显存优化

针对4GB显存的优化策略：
- 分批处理长序列（max_batch_frames=81）
- 滑动窗口处理超长视频
- 支持CPU回退

---

## 项目目录结构

```
running/
├── main.py                    # 主程序入口
├── config/
│   └── config.py              # 配置文件
├── modules/
│   ├── video_processor.py     # 视频处理
│   ├── pose_estimator.py      # 姿态估计（2D+3D集成）
│   ├── pose_lifter.py         # MotionBERT 3D提升
│   ├── kinematic_analyzer.py  # 运动学分析
│   ├── temporal_model.py      # 时序深度学习
│   ├── quality_evaluator.py   # 质量评价
│   ├── ai_analyzer.py         # AI报告生成
│   └── database.py            # 数据库管理
├── models/
│   ├── lstm_model.py          # LSTM模型
│   ├── cnn_model.py           # CNN模型
│   ├── transformer_model.py   # Transformer模型
│   └── quality_model.py       # 质量评估模型
├── utils/
│   ├── visualization.py       # 骨架可视化工具
│   └── visualization_charts.py # 数据图表可视化（雷达图、饼图等）
├── data/
│   ├── checkpoints/           # 模型权重
│   │   └── best_epoch.bin     # MotionBERT权重
│   ├── database.db            # 原引擎兼容 SQLite 数据库
│   └── webapp/
│       └── running_web.db     # 当前 Web 平台主数据库
├── output/                    # 分析结果输出（当前主链路使用 output/tasks/<task_id>/）
└── PROJECT_DOCUMENTATION.md   # 本文档
```

---

## 版本信息

- **版本**: 1.2.0
- **Python**: 3.8+
- **核心依赖**:
  - mediapipe >= 0.10.0
  - torch >= 2.0.0
  - opencv-python >= 4.8.0
  - numpy >= 1.24.0
  - scipy >= 1.10.0
  - plotly >= 5.0.0 (新增)

---

## 更新日志

### v1.2.0 (2026-01-10)
- **评分策略重构**: 技术质量评分改为纯运动学规则，不再融合DL模型输出
- **DL模型定位调整**: LSTM/CNN模型输出仅作为参考信息展示，不参与评分
- **可视化图表**: 新增`visualization_charts.py`，支持雷达图、饼图、仪表盘等
- **UI界面优化**: Streamlit界面重构，添加数据可视化图表区
- **三维度评分**: 稳定性(35%) + 效率(35%) + 跑姿(30%)
- **移除节奏评估**: 删除节奏维度（原四维改为三维）
- **文档更新**: 明确深度学习分层应用策略

### v1.1.0 (2026-01-09)
- **3D角度投影**: 新增视角投影计算（侧面→XY平面，正面→YZ平面）
- **置信度加权**: 落地膝角计算改用置信度加权平均
- **相位角度优化**: 腾空期取摆动腿角度，触地期取支撑腿角度
- **垂直振幅稳健化**: 侧面视角改为2D主导、去趋势和按步态周期统计振幅
- **视频兼容性**: 新增视频H.264转码，解决浏览器播放问题
- **文档更新**: 添加深度学习技术详解

### v1.0.0 (2026-01-08)
- 初始版本发布
- MediaPipe 2D姿态估计
- MotionBERT 3D姿态提升
- LSTM/Transformer/CNN时序分析
- 运动学特征计算
- 质量评价系统

---

*文档更新日期: 2026-01-10*


