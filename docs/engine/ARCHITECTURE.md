# 系统架构文档

本文档详细描述跑步动作分析系统的技术架构、设计决策和数据流。

---

## 一、系统概览

### 1.1 技术栈

| 层次 | 技术 | 用途 |
|------|------|------|
| 深度学习框架 | PyTorch 2.0+ | 模型训练与推理 |
| 2D 姿态估计 | MediaPipe Pose | 33 关键点检测 |
| 3D 姿态提升 | MotionBERT/DSTformer | 2D -> 3D 提升 |
| 信号处理 | SciPy | FFT、滤波、峰值检测 |
| Web 界面 | Streamlit | 快速原型 UI |
| REST API | Flask | 后端服务 |
| 数据库 | SQLite + SQLAlchemy | 持久化存储 |
| 可视化 | Plotly, ECharts | 图表生成 |
| AI 分析 | 智谱 AI (glm-4.5-air) | 智能报告生成 |

### 1.2 系统分层

```
+----------------------------------------------------------+
|                     表现层 (Presentation)                  |
|  - Streamlit Web UI (streamlit_app.py)                    |
|  - Flask REST API (backend/app.py + backend/routes/*)    |
+----------------------------------------------------------+
                            |
+----------------------------------------------------------+
|                     业务逻辑层 (Business Logic)            |
|  - 质量评价 (quality_evaluator.py)                        |
|  - AI 分析 (ai_analyzer.py)                               |
+----------------------------------------------------------+
                            |
+----------------------------------------------------------+
|                     特征提取层 (Feature Extraction)        |
|  - 运动学分析 (kinematic_analyzer.py)                     |
|  - 时序模型 (temporal_model.py, 已停用/兼容保留)           |
+----------------------------------------------------------+
                            |
+----------------------------------------------------------+
|                     姿态估计层 (Pose Estimation)           |
|  - 2D 检测 (pose_estimator.py)                            |
|  - 3D 提升 (pose_lifter.py)                               |
+----------------------------------------------------------+
                            |
+----------------------------------------------------------+
|                     数据层 (Data Layer)                    |
|  - 视频处理 (video_processor.py)                          |
|  - 数据库 (database.py)                                   |
+----------------------------------------------------------+
```

---

## 二、核心数据流

### 2.1 完整处理流程

```
视频文件 (MP4/AVI/MOV)
    |
    | VideoProcessor.extract_frames()
    v
帧序列 List[np.ndarray] + fps + video_info
    |
    | PoseEstimator.process_frames()
    v
2D 关键点 List[Dict]  (33 MediaPipe 关键点)
    |
    +------------------------+
    |                        |
    | KeypointMapper         | (2D 直接用于时序分析)
    v                        v
H36M 格式 (T, 17, 3)    运动学分析 (2D 部分)
    |                        - 步频检测
    | DSTformer              - 触地时刻检测
    v                        - 垂直振幅 (2D)
3D 姿态 (T, 17, 3)
    |
    v
运动学分析 (3D 部分)
    - 膝关节角度
    - 髋关节角度
    - 躯干前倾
    |
    v
质量评价 (QualityEvaluator)
    |
    v
AI 报告 (AIAnalyzer)
    |
    v
结果展示 / 存储
```

### 2.2 关键设计决策

#### 决策 1: 2D 时序 + 3D 空间分离

**背景**: 3D 姿态提升存在误差 (MPJPE ~17mm)，但时序关系保留良好。

**方案**:
- **2D 用于时序分析**: 步频检测、触地时刻判断
- **3D 用于空间分析**: 关节角度、躯干姿态

**实现**:
```python
# kinematic_analyzer.py
def analyze_sequence(...):
    # 步频: 使用 2D 膝关节 Y 坐标
    cadence = self._calculate_cadence_improved(keypoints_2d, fps)

    # 膝角: 使用 3D 坐标
    if self.has_3d_data:
        knee_angle = self._calculate_joint_angle_3d(pose_3d, ...)
```

#### 决策 2: 正面/侧面视角分离

**背景**: 不同视角能观察到的指标不同。

| 指标 | 侧面可见 | 正面可见 |
|------|----------|----------|
| 膝关节角度 | O | X |
| 垂直振幅 | O | O (估算) |
| 躯干前倾 | O | X |
| 膝外翻 | X | O |
| 髋部下沉 | X | O |
| 左右对称性 | X | O |

**实现**:
```python
# kinematic_analyzer.py
def analyze_sequence(self, ..., view_angle='side'):
    if view_angle == 'side':
        return self._analyze_side_view(...)
    else:
        return self._analyze_frontal_view(...)
```

#### 决策 3: 步频 FFT 算法

**背景**: 传统峰值检测在低帧率或噪声视频中不稳定。

**方案**: FFT + 自相关融合

```python
# 1. 带通滤波 (1.2-4.0 Hz, 覆盖 90-240 spm)
# 2. FFT 提取主频
# 3. 自相关验证周期
# 4. 置信度加权融合
```

---

## 三、模块依赖关系

```
                    +-------------------+
                    |   config.py       |
                    | (全局配置中心)     |
                    +-------------------+
                           |
    +----------------------+----------------------+
    |                      |                      |
    v                      v                      v
+---------------+  +----------------+  +------------------+
| video_        |  | pose_          |  | kinematic_utils  |
| processor.py  |  | estimator.py   |  | .py              |
+---------------+  +----------------+  +------------------+
        |                  |                      |
        v                  v                      v
        +------------------+          +-----------+
                |                     |
                v                     v
        +----------------+  +-------------------+
        | pose_lifter.py |  | kinematic_        |
        +----------------+  | analyzer.py       |
                |           +-------------------+
                |                     |
                +----------+----------+
                           |
                           v
                +-------------------+
                | quality_          |
                | evaluator.py      |
                +-------------------+
                           |
                           v
                +-------------------+
                | ai_analyzer.py    |
                +-------------------+
                           |
                +----------+----------+
                |                     |
                v                     v
        +---------------+   +----------------+
        | database.py   |   | streamlit_     |
        +---------------+   | app.py         |
                            +----------------+
```

---

## 四、深度学习模型架构

### 4.1 DSTformer (3D 姿态提升)

```
输入: (B, T, 17, 3)  # 2D 关键点序列
              |
    +-------------------+
    |  Spatial Stream   |  <- 空间自注意力
    +-------------------+
              |
    +-------------------+
    |  Temporal Stream  |  <- 时间自注意力
    +-------------------+
              |
    +-------------------+
    |  Fusion Layers    |  <- 特征融合
    +-------------------+
              |
    +-------------------+
    |  Regression Head  |  <- 3D 坐标回归
    +-------------------+
              |
输出: (B, T, 17, 3)  # 3D 关键点序列
```

**关键参数**:
- dim_feat: 256
- dim_rep: 512
- depth: 5 层
- num_heads: 8

### 4.2 时序分类模型（已停用，历史模块）

```
输入: (B, T, 66)  # 33 关键点 * 2 (x, y)
              |
    +-------------------+
    |  Bi-LSTM / CNN    |
    +-------------------+
              |
    +-------------------+
    |  Fully Connected  |
    +-------------------+
              |
输出: (B, 3)  # 阶段分类 (触地/腾空/过渡)
```

> 当前引擎默认停用 `temporal_model.py`，仅保留兼容字段与历史记录读取能力。

---

## 五、评分体系架构

### 5.1 侧面视角评分

```
总分 = 稳定性 * 0.30 + 效率 * 0.40 + 跑姿 * 0.30

稳定性 (30%)
  |- 躯干稳定性 (60%): 角度标准差
  |- 头部稳定性 (40%): 位置变异系数

效率 (40%)
  |- 步频 (40%): 与理想范围的偏差
  |- 垂直振幅 (30%): 躯干长度百分比
  |- 触地时间 (30%): 毫秒

跑姿 (30%)
  |- 膝角 (60%): 触地期角度
  |- 躯干前倾 (40%): 角度范围
```

### 5.2 正面视角评分

```
总分 = 下肢力线 * 0.35 + 横向稳定性 * 0.35 + 效率 * 0.30

下肢力线 (35%)
  |- 膝外翻 (50%): 左右平均偏移角度
  |- 髋部下沉 (30%): 下沉角度
  |- 整体评级 (20%): 综合评价

横向稳定性 (35%)
  |- 髋部横摆 (40%): 相对振幅百分比
  |- 肩部倾斜 (30%): 倾斜角度
  |- 对称性 (30%): 左右差异

效率 (30%)
  |- 步频 (60%): 与理想范围的偏差
  |- 垂直振幅 (40%): 躯干长度百分比
```

---

## 六、扩展架构 (前后端分离)

### 6.1 目标架构

```
+------------------+     +------------------+     +------------------+
|   Vue3 前端      | <-> |   Flask API      | <-> |   分析引擎       |
|   (Element Plus) |     |   (REST/WS)      |     |   (modules/)     |
+------------------+     +------------------+     +------------------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
|   ECharts 图表   |     |   JWT 认证       |     |   SQLite 数据库  |
+------------------+     +------------------+     +------------------+
```

### 6.2 API 设计

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/auth/register` | POST | 用户注册 |
| `/api/auth/login` | POST | 用户登录 |
| `/api/auth/refresh` | POST | 刷新访问令牌 |
| `/api/auth/logout` | POST | 客户端登出 |
| `/api/auth/me` | GET | 获取当前用户 |
| `/api/upload` | POST | 上传视频并创建分析任务 |
| `/api/task/{task_id}` | GET | 获取任务状态 |
| `/api/result/{record_id}` | GET | 获取分析结果详情 |
| `/api/result/{record_id}/notes` | POST | 保存人工备注 |
| `/api/result/{record_id}/pdf-preview` | GET | 预览 PDF 页面 |
| `/api/result/{record_id}/pdf` | GET | 下载 PDF 报告 |
| `/api/result/{record_id}/ai` | POST | 手动触发 AI 报告 |
| `/api/result/{record_id}/local-report` | POST | 生成本地规则报告 |
| `/api/history` | GET | 获取历史记录 |
| `/api/result/{record_id}` | DELETE | 删除记录 |
| `/api/result/{record_id}/rename` | POST | 重命名记录 |
| `/api/statistics` | GET | 获取统计信息 |
| `/api/media/original/{record_id}` | GET | 获取原始视频 |
| `/api/media/output/{record_id}/{filename}` | GET | 获取输出媒体 |


### 6.3 数据库扩展

```sql
-- 用户表
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

-- 分析任务表
CREATE TABLE analysis_tasks (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    stage TEXT,
    error_message TEXT,
    view_angle TEXT NOT NULL DEFAULT 'side',
    enable_3d INTEGER NOT NULL DEFAULT 1,
    input_video_path TEXT,
    result_record_id INTEGER,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

-- 分析结果表
CREATE TABLE analysis_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    task_id TEXT REFERENCES analysis_tasks(id),
    video_filename TEXT NOT NULL,
    video_hash TEXT,
    video_info_json TEXT,
    original_video_path TEXT,
    output_dir TEXT,
    pose_video_filename TEXT,
    keyframes_json TEXT,
    view_angle TEXT NOT NULL,
    enable_3d INTEGER NOT NULL DEFAULT 1,
    model_version TEXT,
    model_checksum TEXT,
    config_json TEXT,
    git_commit TEXT,
    total_score REAL,
    rating TEXT,
    dimension_scores_json TEXT,
    strengths_json TEXT,
    weaknesses_json TEXT,
    suggestions_json TEXT,
    kinematic_json TEXT,
    temporal_json TEXT,
    quality_json TEXT,
    ai_analysis TEXT,
    manual_notes TEXT,
    manual_notes_updated_at DATETIME,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

-- 管理员审计日志
CREATE TABLE admin_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_user_id INTEGER NOT NULL REFERENCES users(id),
    action TEXT NOT NULL,
    target_user_id INTEGER,
    target_record_id INTEGER,
    details_json TEXT,
    created_at DATETIME NOT NULL
);
```

---

## 七、性能考虑

### 7.1 显存优化

- DSTformer 推理: ~2GB 显存
- batch_size <= 8 (4GB GPU)
- 冻结前 3 层减少训练参数

### 7.2 处理时间

| 阶段 | 时间 (1分钟视频) |
|------|------------------|
| 视频抽帧 | ~2s |
| 2D 姿态估计 | ~30s |
| 3D 姿态提升 | ~10s |
| 运动学分析 | ~5s |
| 质量评价 | ~1s |
| **总计** | **~50s** |

### 7.3 优化建议

1. 视频预处理: 降采样到 640x480
2. 跳帧处理: 每 2 帧取 1 帧
3. 模型量化: FP16 推理
4. 结果缓存: 相同视频不重复计算

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE analysis_tasks (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    stage TEXT,
    error_message TEXT,
    view_angle TEXT NOT NULL DEFAULT 'side',
    enable_3d INTEGER NOT NULL DEFAULT 1,
    input_video_path TEXT,
    result_record_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE analysis_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    task_id TEXT,
    video_filename TEXT NOT NULL,
    video_hash TEXT,
    video_info_json TEXT,
    original_video_path TEXT,
    output_dir TEXT,
    pose_video_filename TEXT,
    keyframes_json TEXT,
    view_angle TEXT NOT NULL,
    enable_3d INTEGER NOT NULL DEFAULT 1,
    model_version TEXT,
    model_checksum TEXT,
    config_json TEXT,
    git_commit TEXT,
    total_score REAL,
    rating TEXT,
    dimension_scores_json TEXT,
    strengths_json TEXT,
    weaknesses_json TEXT,
    suggestions_json TEXT,
    kinematic_json TEXT,
    temporal_json TEXT,
    quality_json TEXT,
    ai_analysis TEXT,
    manual_notes TEXT,
    manual_notes_updated_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE admin_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_user_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    target_user_id INTEGER,
    target_record_id INTEGER,
    details_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```
+----------------------------------------------------------+
|                     表现层 (Presentation)                  |
|  - Streamlit Web UI (streamlit_app.py)                    |
|  - Flask REST API (backend/app.py + backend/routes/*)    |
+----------------------------------------------------------+
                            |
+----------------------------------------------------------+
|                     业务逻辑层 (Business Logic)            |
|  - 质量评价 (quality_evaluator.py)                        |
|  - AI 分析 (ai_analyzer.py)                               |
+----------------------------------------------------------+
                            |
+----------------------------------------------------------+
|                     特征提取层 (Feature Extraction)        |
|  - 运动学分析 (kinematic_analyzer.py)                     |
|  - 时序模型 (temporal_model.py, 已停用/兼容保留)           |
+----------------------------------------------------------+
                            |
+----------------------------------------------------------+
|                     姿态估计层 (Pose Estimation)           |
|  - 2D 检测 (pose_estimator.py)                            |
|  - 3D 提升 (pose_lifter.py)                               |
+----------------------------------------------------------+
                            |
+----------------------------------------------------------+
|                     数据层 (Data Layer)                    |
|  - 视频处理 (video_processor.py)                          |
|  - 数据库 (database.py)                                   |
+----------------------------------------------------------+
```

---

## 二、核心数据流

### 2.1 完整处理流程

```
视频文件 (MP4/AVI/MOV)
    |
    | VideoProcessor.extract_frames()
    v
帧序列 List[np.ndarray] + fps + video_info
    |
    | PoseEstimator.process_frames()
    v
2D 关键点 List[Dict]  (33 MediaPipe 关键点)
    |
    +------------------------+
    |                        |
    | KeypointMapper         | (2D 直接用于时序分析)
    v                        v
H36M 格式 (T, 17, 3)    运动学分析 (2D 部分)
    |                        - 步频检测
    | DSTformer              - 触地时刻检测
    v                        - 垂直振幅 (2D)
3D 姿态 (T, 17, 3)
    |
    v
运动学分析 (3D 部分)
    - 膝关节角度
    - 髋关节角度
    - 躯干前倾
    |
    v
质量评价 (QualityEvaluator)
    |
    v
AI 报告 (AIAnalyzer)
    |
    v
结果展示 / 存储
```

### 2.2 关键设计决策

#### 决策 1: 2D 时序 + 3D 空间分离

**背景**: 3D 姿态提升存在误差 (MPJPE ~17mm)，但时序关系保留良好。

**方案**:
- **2D 用于时序分析**: 步频检测、触地时刻判断
- **3D 用于空间分析**: 关节角度、躯干姿态

**实现**:
```python
# kinematic_analyzer.py
def analyze_sequence(...):
    # 步频: 使用 2D 膝关节 Y 坐标
    cadence = self._calculate_cadence_improved(keypoints_2d, fps)

    # 膝角: 使用 3D 坐标
    if self.has_3d_data:
        knee_angle = self._calculate_joint_angle_3d(pose_3d, ...)
```

#### 决策 2: 正面/侧面视角分离

**背景**: 不同视角能观察到的指标不同。

| 指标 | 侧面可见 | 正面可见 |
|------|----------|----------|
| 膝关节角度 | O | X |
| 垂直振幅 | O | O (估算) |
| 躯干前倾 | O | X |
| 膝外翻 | X | O |
| 髋部下沉 | X | O |
| 左右对称性 | X | O |

**实现**:
```python
# kinematic_analyzer.py
def analyze_sequence(self, ..., view_angle='side'):
    if view_angle == 'side':
        return self._analyze_side_view(...)
    else:
        return self._analyze_frontal_view(...)
```

#### 决策 3: 步频 FFT 算法

**背景**: 传统峰值检测在低帧率或噪声视频中不稳定。

**方案**: FFT + 自相关融合

```python
# 1. 带通滤波 (1.2-4.0 Hz, 覆盖 90-240 spm)
# 2. FFT 提取主频
# 3. 自相关验证周期
# 4. 置信度加权融合
```

---

## 三、模块依赖关系

```
                    +-------------------+
                    |   config.py       |
                    | (全局配置中心)     |
                    +-------------------+
                           |
    +----------------------+----------------------+
    |                      |                      |
    v                      v                      v
+---------------+  +----------------+  +------------------+
| video_        |  | pose_          |  | kinematic_utils  |
| processor.py  |  | estimator.py   |  | .py              |
+---------------+  +----------------+  +------------------+
        |                  |                      |
        v                  v                      v
        +------------------+          +-----------+
                |                     |
                v                     v
        +----------------+  +-------------------+
        | pose_lifter.py |  | kinematic_        |
        +----------------+  | analyzer.py       |
                |           +-------------------+
                |                     |
                +----------+----------+
                           |
                           v
                +-------------------+
                | quality_          |
                | evaluator.py      |
                +-------------------+
                           |
                           v
                +-------------------+
                | ai_analyzer.py    |
                +-------------------+
                           |
                +----------+----------+
                |                     |
                v                     v
        +---------------+   +----------------+
        | database.py   |   | streamlit_     |
        +---------------+   | app.py         |
                            +----------------+
```

---

## 四、深度学习模型架构

### 4.1 DSTformer (3D 姿态提升)

```
输入: (B, T, 17, 3)  # 2D 关键点序列
              |
    +-------------------+
    |  Spatial Stream   |  <- 空间自注意力
    +-------------------+
              |
    +-------------------+
    |  Temporal Stream  |  <- 时间自注意力
    +-------------------+
              |
    +-------------------+
    |  Fusion Layers    |  <- 特征融合
    +-------------------+
              |
    +-------------------+
    |  Regression Head  |  <- 3D 坐标回归
    +-------------------+
              |
输出: (B, T, 17, 3)  # 3D 关键点序列
```

**关键参数**:
- dim_feat: 256
- dim_rep: 512
- depth: 5 层
- num_heads: 8

### 4.2 时序分类模型（已停用，历史模块）

```
输入: (B, T, 66)  # 33 关键点 * 2 (x, y)
              |
    +-------------------+
    |  Bi-LSTM / CNN    |
    +-------------------+
              |
    +-------------------+
    |  Fully Connected  |
    +-------------------+
              |
输出: (B, 3)  # 阶段分类 (触地/腾空/过渡)
```

> 当前引擎默认停用 `temporal_model.py`，仅保留兼容字段与历史记录读取能力。

---

## 五、评分体系架构

### 5.1 侧面视角评分

```
总分 = 稳定性 * 0.30 + 效率 * 0.40 + 跑姿 * 0.30

稳定性 (30%)
  |- 躯干稳定性 (60%): 角度标准差
  |- 头部稳定性 (40%): 位置变异系数

效率 (40%)
  |- 步频 (40%): 与理想范围的偏差
  |- 垂直振幅 (30%): 躯干长度百分比
  |- 触地时间 (30%): 毫秒

跑姿 (30%)
  |- 膝角 (60%): 触地期角度
  |- 躯干前倾 (40%): 角度范围
```

### 5.2 正面视角评分

```
总分 = 下肢力线 * 0.35 + 横向稳定性 * 0.35 + 效率 * 0.30

下肢力线 (35%)
  |- 膝外翻 (50%): 左右平均偏移角度
  |- 髋部下沉 (30%): 下沉角度
  |- 整体评级 (20%): 综合评价

横向稳定性 (35%)
  |- 髋部横摆 (40%): 相对振幅百分比
  |- 肩部倾斜 (30%): 倾斜角度
  |- 对称性 (30%): 左右差异

效率 (30%)
  |- 步频 (60%): 与理想范围的偏差
  |- 垂直振幅 (40%): 躯干长度百分比
```

---

## 六、扩展架构 (前后端分离)

### 6.1 目标架构

```
+------------------+     +------------------+     +------------------+
|   Vue3 前端      | <-> |   Flask API      | <-> |   分析引擎       |
|   (Element Plus) |     |   (REST/WS)      |     |   (modules/)     |
+------------------+     +------------------+     +------------------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
|   ECharts 图表   |     |   JWT 认证       |     |   SQLite 数据库  |
+------------------+     +------------------+     +------------------+
```

### 6.2 API 设计

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/auth/register` | POST | 用户注册 |
| `/api/auth/login` | POST | 用户登录 |
| `/api/auth/refresh` | POST | 刷新访问令牌 |
| `/api/auth/logout` | POST | 客户端登出 |
| `/api/auth/me` | GET | 获取当前用户 |
| `/api/upload` | POST | 上传视频并创建分析任务 |
| `/api/task/{task_id}` | GET | 获取任务状态 |
| `/api/result/{record_id}` | GET | 获取分析结果详情 |
| `/api/result/{record_id}/notes` | POST | 保存人工备注 |
| `/api/result/{record_id}/pdf-preview` | GET | 预览 PDF 页面 |
| `/api/result/{record_id}/pdf` | GET | 下载 PDF 报告 |
| `/api/result/{record_id}/ai` | POST | 手动触发 AI 报告 |
| `/api/result/{record_id}/local-report` | POST | 生成本地规则报告 |
| `/api/history` | GET | 获取历史记录 |
| `/api/result/{record_id}` | DELETE | 删除记录 |
| `/api/result/{record_id}/rename` | POST | 重命名记录 |
| `/api/statistics` | GET | 获取统计信息 |
| `/api/media/original/{record_id}` | GET | 获取原始视频 |
| `/api/media/output/{record_id}/{filename}` | GET | 获取输出媒体 |


### 6.3 数据库扩展

```sql
-- 用户表
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

-- 分析任务表
CREATE TABLE analysis_tasks (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    stage TEXT,
    error_message TEXT,
    view_angle TEXT NOT NULL DEFAULT 'side',
    enable_3d INTEGER NOT NULL DEFAULT 1,
    input_video_path TEXT,
    result_record_id INTEGER,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

-- 分析结果表
CREATE TABLE analysis_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    task_id TEXT REFERENCES analysis_tasks(id),
    video_filename TEXT NOT NULL,
    video_hash TEXT,
    video_info_json TEXT,
    original_video_path TEXT,
    output_dir TEXT,
    pose_video_filename TEXT,
    keyframes_json TEXT,
    view_angle TEXT NOT NULL,
    enable_3d INTEGER NOT NULL DEFAULT 1,
    model_version TEXT,
    model_checksum TEXT,
    config_json TEXT,
    git_commit TEXT,
    total_score REAL,
    rating TEXT,
    dimension_scores_json TEXT,
    strengths_json TEXT,
    weaknesses_json TEXT,
    suggestions_json TEXT,
    kinematic_json TEXT,
    temporal_json TEXT,
    quality_json TEXT,
    ai_analysis TEXT,
    manual_notes TEXT,
    manual_notes_updated_at DATETIME,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

-- 管理员审计日志
CREATE TABLE admin_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_user_id INTEGER NOT NULL REFERENCES users(id),
    action TEXT NOT NULL,
    target_user_id INTEGER,
    target_record_id INTEGER,
    details_json TEXT,
    created_at DATETIME NOT NULL
);
```

---

## 七、性能考虑

### 7.1 显存优化

- DSTformer 推理: ~2GB 显存
- batch_size <= 8 (4GB GPU)
- 冻结前 3 层减少训练参数

### 7.2 处理时间

| 阶段 | 时间 (1分钟视频) |
|------|------------------|
| 视频抽帧 | ~2s |
| 2D 姿态估计 | ~30s |
| 3D 姿态提升 | ~10s |
| 运动学分析 | ~5s |
| 质量评价 | ~1s |
| **总计** | **~50s** |

### 7.3 优化建议

1. 视频预处理: 降采样到 640x480
2. 跳帧处理: 每 2 帧取 1 帧
3. 模型量化: FP16 推理
4. 结果缓存: 相同视频不重复计算


