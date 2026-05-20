# 前后端开发完整指南

> 本文档为新 Claude 会话提供完整的开发指令，用于在现有跑步动作分析引擎基础上构建前后端分离的 Web 平台。

---

## 一、项目背景与目标

### 1.1 项目简介

这是一个本科毕业设计项目，已有完整的分析引擎（Python），需要构建 Web 界面实现：
1. 上传视频并分析
2. 查看分析结果（雷达图、曲线、骨架可视化）
3. 保存历史记录并对比
4. 导出 PDF 报告

**核心定位**：学术工程系统，强调可复现性和可追溯性（非商业 demo）。

### 1.2 架构方案

**单仓库单服务架构**：

```
E:\PycharmProjects\running\           # 主工程目录
├── backend/                          # Flask 后端（新建）
│   ├── app.py                        # Flask 应用主入口
│   ├── routes/                       # API 路由
│   │   ├── __init__.py
│   │   ├── analysis.py               # 分析相关接口
│   │   ├── history.py                # 历史记录接口
│   │   └── auth.py                   # 认证接口（可选）
│   ├── services/                     # 业务逻辑层
│   │   ├── __init__.py
│   │   └── analysis_service.py       # 分析服务（调用 modules）
│   ├── models/                       # 数据模型（SQLAlchemy）
│   │   ├── __init__.py
│   │   └── analysis_record.py
│   └── utils/                        # 后端工具函数
│       └── file_utils.py
│
├── frontend/                         # Vue3 前端（新建）
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   │   ├── main.js
│   │   ├── App.vue
│   │   ├── router/                   # 路由配置
│   │   ├── views/                    # 页面组件
│   │   ├── components/               # 通用组件
│   │   ├── stores/                   # Pinia 状态管理
│   │   ├── api/                      # API 调用
│   │   └── utils/                    # 前端工具
│   └── public/
│
├── modules/                          # 分析引擎（保持不变）
├── models/                           # 深度学习模型定义
├── config/                           # 全局配置
├── data/                             # 数据和权重
├── api/                              # 旧 API（删除）
├── web/                              # Streamlit（保留作参考）
└── ...
```

### 1.3 技术栈约束

**后端** (Flask)：
- Flask 2.x + flask-socketio (进度推送，可选)
- SQLite + SQLAlchemy
- 轻量后台任务（ThreadPool / 任务状态表）
- JWT 认证（注册/登录为 MVP 必做）

**前端** (Vue3)：
- Vue 3 + Vite
- Element Plus (UI 组件)
- ECharts (图表)
- Pinia (状态管理)
- vue-router

**关键约束（必须遵守）**：
- 后端**直接调用** modules/ 下的分析引擎，不使用代理或微服务
- 以 `web/streamlit_app.py` 的现有业务顺序作为标准流程（仅做前后端拆分与适配）
- 不改动后处理分析引擎核心逻辑（`modules/kinematic_analyzer.py`、`modules/quality_evaluator.py`、`modules/temporal_model.py` 等）
- 不做过度工程化（本项目为本科毕设，不引入复杂分布式基础设施）
- 旧的 `api/api_server.py` 不再作为当前主链路，前后端开发统一落在 `backend/`

---

## 二、分析引擎接口文档

### 2.1 完整分析流程

以下是后端需要调用的完整分析流程，**必须支持 3D 姿态提升**：

```python
import sys
from pathlib import Path

# 添加项目根目录到路径（backend/ 在 running/ 下）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import POSE_CONFIG, MOTIONBERT_CONFIG
from modules.video_processor import VideoProcessor
from modules.pose_estimator import create_pose_estimator, create_pose_estimator_3d
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.temporal_model import TemporalModelAnalyzer
from modules.quality_evaluator import QualityEvaluator
from modules.ai_analyzer import AIAnalyzer


def run_complete_analysis(video_path: str, view_angle: str = 'side', enable_3d: bool = True) -> dict:
    """
    运行完整分析流程

    Args:
        video_path: 视频文件路径
        view_angle: 'side' (侧面) 或 'front' (正面)
        enable_3d: 是否启用 3D 姿态提升 (默认 True)

    Returns:
        完整分析结果字典
    """
    # ========== 1. 视频预处理 ==========
    processor = VideoProcessor(video_path)
    video_info = processor.get_video_info()

    # 帧提取策略：超过10秒取中间10秒，否则全部
    video_duration = video_info['duration']
    video_fps = video_info['fps']
    target_duration = min(10.0, video_duration)

    if video_duration > 10.0:
        start_time = (video_duration - target_duration) / 2
        start_frame = int(start_time * video_fps)
        max_frames = int(target_duration * video_fps)
    else:
        start_frame = 0
        max_frames = int(video_duration * video_fps)

    frames, fps = processor.extract_frames_from_position(
        start_frame=start_frame,
        target_fps=video_fps,
        max_frames=max_frames
    )

    # ========== 2. 姿态估计 (支持 2D/3D) ==========
    keypoints_sequence = None
    keypoints_3d = None
    poses_3d = None
    has_3d = False

    if enable_3d:
        try:
            # 使用 3D 姿态估计器 (MediaPipe + MotionBERT)
            estimator = create_pose_estimator_3d(
                backend_2d=POSE_CONFIG['backend'],
                enable_3d=True,
                device='auto'
            )
            pose_result = estimator.process_frames(
                frames,
                lift_to_3d=True,
                view_angle=view_angle
            )
            keypoints_sequence = pose_result['keypoints_2d']
            keypoints_3d = pose_result.get('keypoints_3d')
            poses_3d = pose_result.get('poses_3d')
            has_3d = pose_result.get('has_3d', False)
        except Exception as e:
            print(f"3D 估计失败: {e}，回退到 2D")
            estimator = create_pose_estimator(POSE_CONFIG['backend'], POSE_CONFIG)
            keypoints_sequence = estimator.process_frames(frames)
    else:
        estimator = create_pose_estimator(POSE_CONFIG['backend'], POSE_CONFIG)
        keypoints_sequence = estimator.process_frames(frames)

    # ========== 3. 运动学分析 (必须传 3D 数据) ==========
    kinematic_analyzer = KinematicAnalyzer()
    kinematic_results = kinematic_analyzer.analyze_sequence(
        keypoints_sequence, fps,
        view_angle=view_angle,
        poses_3d=poses_3d,           # 重要：传递 3D 姿态数组
        keypoints_3d=keypoints_3d    # 重要：传递 3D 关键点
    )

    # ========== 4. 时序深度学习分析 ==========
    temporal_analyzer = TemporalModelAnalyzer()
    temporal_results = temporal_analyzer.analyze(
        keypoints_sequence,
        view_angle=view_angle,
        kinematic_results=kinematic_results,
        keypoints_3d=keypoints_3d
    )

    # ========== 5. 技术质量评价 ==========
    quality_evaluator = QualityEvaluator()
    quality_results = quality_evaluator.evaluate(
        kinematic_results, temporal_results,
        view_angle=view_angle
    )

    # ========== 6. AI 文本分析 ==========
    ai_analyzer = AIAnalyzer()
    results_for_ai = {
        'quality_evaluation': quality_results,
        'kinematic_analysis': kinematic_results,
        'temporal_analysis': temporal_results,
        'view_angle': view_angle
    }
    ai_text = ai_analyzer.generate_analysis_report(results_for_ai)

    # ========== 7. 整合结果 ==========
    complete_results = {
        'video_info': video_info,
        'view_angle': view_angle,
        'kinematic_analysis': kinematic_results,
        'temporal_analysis': temporal_results,
        'quality_evaluation': quality_results,
        'ai_analysis': ai_text,
        'pose_3d_info': {
            'enabled': enable_3d,
            'has_3d_data': has_3d,
            'description': '使用 MotionBERT 进行 2D→3D 姿态提升' if has_3d else '仅使用 2D 姿态数据'
        }
    }

    # 清理资源
    processor.release()
    estimator.close()

    return complete_results
```

### 2.2 数据结构规格

#### 2.2.1 video_info

```python
{
    'width': int,           # 视频宽度 (像素)
    'height': int,          # 视频高度 (像素)
    'fps': float,           # 帧率
    'frame_count': int,     # 总帧数
    'duration': float,      # 时长 (秒)
    'rotation': int         # 旋转角度 (0/90/180/270)
}
```

#### 2.2.2 kinematic_results (侧面视角)

```python
{
    'angles': {
        'knee_left': List[float],      # 左膝角度序列
        'knee_right': List[float],     # 右膝角度序列
        'knee': {                       # 综合膝角分析
            'phase_analysis': {
                'ground_contact': {     # 触地期
                    'mean': float,      # 平均角度
                    'std': float,       # 标准差
                    'min': float,
                    'max': float,
                    'count': int,
                    'rating': dict
                },
                'flight': {...},        # 腾空期
                'transition': {...},    # 过渡期
                'max_flexion': float,   # 最大屈曲角度
                'range_of_motion': float  # 活动范围
            }
        },
        'data_reliability': {
            'is_3d': bool,              # 是否使用 3D 数据
            'confidence': float
        }
    },
    'vertical_motion': {
        'amplitude_normalized': float,  # 垂直振幅（去趋势后按步态周期统计，躯干长度 %）
        'amplitude_rating': {
            'rating': str,              # 'excellent'/'good'/'fair'/'poor'
            'description': str
        },
        'data_source': str              # '3D' 或 '2D'
    },
    'cadence': {
        'cadence': float,               # 步频 (步/分)
        'step_count': int,              # 检测步数
        'duration': float,              # 分析时长
        'rating': str,                  # 评级
        'confidence': float             # 置信度 (0-1)
    },
    'gait_cycle': {
        'phase_duration_ms': {
            'ground_contact': float,    # 触地时间 (ms)
            'flight': float             # 腾空时间 (ms)
        },
        'duty_factor': float            # 占空比 (触地时间/周期)
    },
    'stability': {
        'overall': float,               # 整体稳定性 (0-100)
        'trunk': float,                 # 躯干稳定性
        'head': float                   # 头部稳定性
    },
    'body_lean': {
        'mean': float,                  # 躯干前倾角 (度)
        'std': float
    },
    'arm_swing': {
        'symmetry': float,              # 摆臂对称性 (0-1)
        'amplitude': float              # 摆臂幅度
    }
}
```

#### 2.2.3 kinematic_results (正面视角)

```python
{
    'lower_limb_alignment': {
        'left_leg': {
            'mean': float,              # 膝外翻角度 (度)
            'issue': str                # 'normal'/'slight_valgus'/'moderate_valgus'
        },
        'right_leg': {...},
        'hip_drop': {
            'mean': float,              # 髋部下沉角度
            'left_drop': float,
            'right_drop': float
        },
        'asymmetry': float,             # 左右不对称性 (%)
        'overall_rating': str
    },
    'lateral_stability': {
        'hip_sway': {
            'relative_range': float,    # 髋部横摆 (%)
            'rating': str
        },
        'shoulder_tilt': float          # 肩部倾斜角度
    },
    'cadence': {
        'cadence': float,               # 步频 (步/分)
        'rating': str,
        'confidence': float
    },
    'vertical_motion': {
        'amplitude_normalized': float,  # 垂直振幅 (%)
        'amplitude_rating': {...}
    },
    'gait_symmetry': {
        'overall': float,               # 整体对称性 (0-1)
        'hip': float,
        'knee': float
    }
}
```

#### 2.2.4 quality_results

```python
{
    'total_score': float,       # 总分 (0-100)
    'rating': str,              # 'excellent'/'good'/'fair'/'poor'
    'dimension_scores': {
        # 侧面视角:
        'stability': float,     # 稳定性得分 (权重 30%)
        'efficiency': float,    # 效率得分 (权重 40%)
        'form': float           # 跑姿得分 (权重 30%)
        # 正面视角:
        # 'lower_limb_alignment': float,  # 下肢力线 (权重 35%)
        # 'lateral_stability': float,     # 横向稳定性 (权重 35%)
        # 'efficiency': float             # 效率 (权重 30%)
    },
    'strengths': List[str],     # 优势项
    'weaknesses': List[str],    # 薄弱项
    'suggestions': List[str],   # 改进建议
    'data_reliability': {
        'overall': str,         # 'high'/'medium'/'low'
        'details': str
    }
}
```

#### 2.2.5 temporal_results

```python
{
    'phase_distribution': {
        'ground_contact': float,    # 触地期占比 (0-1)
        'flight': float,            # 腾空期占比
        'transition': float         # 过渡期占比
    },
    'phases': List[int],            # 每帧阶段 (0=触地, 1=腾空, 2=过渡)
    'gait_events': {
        'touchdown_frames': List[int],   # 触地帧索引
        'toe_off_frames': List[int]      # 离地帧索引
    }
}
```

---

## 三、后端 API 规范

### 3.1 基础配置

**服务端口**: 5000 (可配置)
**CORS**: 允许前端跨域访问
**认证**: JWT Bearer (除登录/注册外其余接口默认鉴权)
**Content-Type**: application/json (响应), multipart/form-data (上传)

### 3.2 API 端点

#### 3.2.0 用户认证

```
POST /api/auth/register
POST /api/auth/login
POST /api/auth/refresh
POST /api/auth/logout
GET  /api/auth/me
```

**说明**：
- 登录成功返回 `access_token`、`refresh_token` 与用户信息。
- 业务接口默认要求 `Authorization: Bearer <access_token>`。
- 登录失败已区分“用户不存在”“密码错误”“用户名为空”“密码为空”等情况。

#### 3.2.1 上传与分析

```
POST /api/upload
GET  /api/task/{task_id}
GET  /api/result/{record_id}
POST /api/result/{record_id}/notes
GET  /api/result/{record_id}/pdf-preview
GET  /api/result/{record_id}/pdf
POST /api/result/{record_id}/ai
POST /api/result/{record_id}/local-report
DELETE /api/result/{record_id}
POST /api/result/{record_id}/rename
```

**上传请求体**：
- `file`: 视频文件
- `view_angle`: `side` 或 `front`
- `enable_3d`: 可选，默认 `true`

**任务状态说明**：
- 当前主链路通过轮询 `GET /api/task/{task_id}` 获取进度。
- 任务状态以 `queued / running / succeeded / failed` 为主。
- AI 报告默认不自动生成，结果页需手动触发 `/api/result/{record_id}/ai`。

#### 3.2.2 历史、统计与媒体

```
GET    /api/history
GET    /api/statistics
GET    /api/media/original/{record_id}
GET    /api/media/output/{record_id}/{filename}
```

#### 3.2.3 管理员接口

```
GET    /api/admin/overview
GET    /api/admin/users
GET    /api/admin/users/{user_id}/records
PATCH  /api/admin/users/{user_id}
POST   /api/admin/users/{user_id}/reset-password
DELETE /api/admin/records/{record_id}/hard-delete
POST   /api/admin/records/hard-delete-batch
DELETE /api/admin/users/{user_id}/hard-delete
POST   /api/admin/users/hard-delete-batch
POST   /api/admin/storage/cleanup-orphans
POST   /api/admin/storage/cleanup-dangling-tasks
POST   /api/admin/storage/cleanup-stale-queued
GET    /api/admin/audit-logs
```

### 3.3 进度推送说明

当前主实现以 `GET /api/task/{task_id}` 轮询为主，不依赖 WebSocket。

如果后续需要进一步优化观感，可在不改变现有接口契约的前提下补充 WebSocket 或 SSE，但不属于当前论文实现主链路。

---

## 四、数据库设计

### 4.1 表结构

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
    enable_3d BOOLEAN DEFAULT TRUE,
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
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (task_id) REFERENCES analysis_tasks(id)
);

-- 管理员审计日志表
CREATE TABLE admin_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_user_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    target_user_id INTEGER,
    target_record_id INTEGER,
    details_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_user_id) REFERENCES users(id)
);

-- 创建索引
CREATE INDEX idx_tasks_user_status ON analysis_tasks(user_id, status);
CREATE INDEX idx_records_user_created ON analysis_records(user_id, created_at DESC);
CREATE INDEX idx_records_view_angle ON analysis_records(view_angle);
CREATE INDEX idx_records_total_score ON analysis_records(total_score);
``` 

### 4.2 追溯元数据要求

每次分析**必须**记录：
1. `video_hash`: 视频文件 SHA256
2. `model_version`: 分析引擎版本
3. `config_json`: 完整配置参数快照
4. `created_at`: 时间戳

---

## 五、前端页面设计

### 5.1 页面结构

| 页面 | 路由 | 功能 |
|------|------|------|
| 登录页 | /login | 用户登录（MVP 必做） |
| 注册页 | /register | 用户注册（MVP 必做） |
| 仪表盘 | `/` | 最近分析、快速入口 |
| 分析页 | `/analyze` | 上传视频、选择视角、查看进度 |
| 结果页 | `/result/:id` | 完整分析结果展示 |
| 历史页 | `/history` | 分页列表、筛选、搜索 |
| 对比页 | `/compare` | 选择多条记录对比 |

### 5.2 分析页 (Analyze)

**布局**：
1. **上传区域**
   - 拖拽上传 / 点击选择
   - 支持格式提示 (MP4/AVI/MOV)
   - 文件大小限制提示 (建议 < 100MB)

2. **视角选择** (必填)
   - 单选：侧面视角 / 正面视角
   - 带图示说明不同视角的适用场景

3. **高级选项** (折叠面板)
   - 3D 分析开关 (默认开启)

4. **进度展示**
   - 进度条 + 百分比
   - 当前阶段文字 ("视频处理中..." / "姿态估计中..." 等)
   - 预计剩余时间 (可选)

5. **完成后**
   - 自动跳转结果页
   - 或显示快速预览 + "查看详情"按钮

### 5.3 结果页 (Result)

**当前结果页补充约束**：
1. 顶部信息卡必须展示 视角类型 / 分析时间 / 模型版本，并在同一卡片空白区提供 下载 PDF 报告 按钮。
2. 人工备注采用右下角浮动控件，独立保存到当前分析记录，并显示备注最近保存时间；刷新页面后不能回退为分析完成时间。
3. PDF 不再通过打印当前网页导出，而是由后端直接生成正式报告文件，至少包含：封面概览、维度雷达图、维度条形图、核心指标、关键帧骨架图、人工备注、AI/本地分析文本；并需要具备头部元信息换行、文本自适应留边、正面关键帧保留原样、侧面关键帧主体裁切和正文分页防孤页。
4. 结果页时间显示以用户当前系统时区为准，前端负责将后端 UTC/naive 时间正确转换后再展示。

**布局**：

```
+------------------------------------------------------------------+
|  [返回]  跑步动作分析报告  #123  [导出PDF] [删除]                    |
+------------------------------------------------------------------+
|                                                                  |
|  +------------------------+  +--------------------------------+  |
|  | 视频播放器             |  | 总体评分                        |  |
|  | (带骨架叠加可选)        |  |  ┌─────────────────────┐      |  |
|  |                        |  |  │   78.4 / 100        │      |  |
|  |                        |  |  │      良好            │      |  |
|  +------------------------+  |  └─────────────────────┘      |  |
|                              |  雷达图 (3/5 维度)              |  |
|                              +--------------------------------+  |
|                                                                  |
|  +-----------------------------------------------------------+  |
|  | 详细指标                                                     |  |
|  | ┌─────────┬─────────┬─────────┬─────────┐                  |  |
|  | │ 步频     │ 振幅     │ 触地时间 │ 膝角     │                  |  |
|  | │ 178步/分 │ 6.2%    │ 245ms   │ 158°    │                  |  |
|  | └─────────┴─────────┴─────────┴─────────┘                  |  |
|  +-----------------------------------------------------------+  |
|                                                                  |
|  +-----------------------------------------------------------+  |
|  | 时间序列曲线 (可切换: 膝角/振幅/步频)                          |  |
|  | [图表区域]                                                   |  |
|  +-----------------------------------------------------------+  |
|                                                                  |
|  +-----------------------------------------------------------+  |
|  | AI 分析报告                                                  |  |
|  | [Markdown 渲染的分析文本]                                     |  |
|  +-----------------------------------------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

**图表需求**：
1. **雷达图**: 显示 3 维度得分 (侧面: 稳定性/效率/跑姿)
2. **仪表盘**: 总分展示
3. **时间序列图**: 膝角曲线、垂直振幅曲线
4. **饼图**: 步态阶段分布
5. **下肢力线图** (正面): 膝外翻角度曲线

### 5.4 历史页 (History)

**功能**：
- 分页列表 (每页 10-20 条)
- 筛选: 日期范围、视角类型、评分范围
- 排序: 时间、评分
- 多选: 勾选后可批量删除或对比
- 卡片 / 列表视图切换

### 5.5 对比页 (Compare)

**功能**：
- 选择 2-3 条记录
- 并排显示雷达图
- 曲线叠加对比
- 差异表格

---

## 六、实施步骤

### 阶段 A: MVP (必须完成)

#### 后端

1. **项目结构搭建**
   - [ ] 创建 `backend/` 目录结构
   - [ ] 配置 Flask + CORS + JWT
   - [ ] 路径导入配置 (sys.path)

2. **认证与权限**
   - [ ] `POST /api/auth/register`
   - [ ] `POST /api/auth/login`
   - [ ] `POST /api/auth/refresh`
   - [ ] 用户数据隔离（仅可访问自己的分析记录）

3. **核心接口**
   - [ ] `POST /api/upload` - 上传分析
   - [ ] `GET /api/task/{task_id}` - 获取任务状态
   - [ ] `GET /api/history` - 历史列表（当前用户）
   - [ ] `GET /api/result/{record_id}` - 记录详情（当前用户）

4. **分析服务**
   - [ ] 封装 `run_complete_analysis()` 适配层
   - [ ] 保持与 `web/streamlit_app.py` 一致的业务顺序
   - [ ] 支持 view_angle 参数
   - [ ] 支持 enable_3d 参数
   - [ ] 保存任务状态与最终记录

5. **数据库**
   - [ ] SQLAlchemy 模型定义（users/tasks/records）
   - [ ] 自动建表
   - [ ] CRUD 操作

#### 前端

1. **项目初始化**
   - [ ] Vue3 + Vite 项目
   - [ ] Element Plus 配置
   - [ ] ECharts 配置
   - [ ] 路由配置

2. **页面开发**
   - [ ] 登录/注册页
   - [ ] 分析页 (上传 + 进度)
   - [ ] 结果页 (图表 + 数据)
   - [ ] 历史页 (列表)

### 阶段 B: 加分项

- [ ] WebSocket 进度推送
- [ ] 历史对比功能
- [ ] PDF 报告导出
- [ ] 暗色主题

### 阶段 C: 亮点功能

- [ ] 逐帧 Inspector
- [ ] 骨架动画播放
- [ ] 实验管理界面

---

## 七、验收标准

1. 能上传视频并在 3 分钟内得到分析结果
2. 注册/登录可用，且用户只能访问自己的记录
3. 数据库记录包含完整追溯元数据（video_hash、model_version、config_json、created_at）
4. 前端能展示雷达图 + 膝角曲线 + 关键指标
5. 支持正面和侧面两种视角
6. 3D 分析功能正常工作
7. 引擎核心评分口径与 Streamlit 结果保持一致

---

## 八、常见问题

**Q: 后端如何导入 modules?**
```python
# backend/services/analysis_service.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# 现在可以导入 modules/
from modules.video_processor import VideoProcessor
```

**Q: 视频文件存储在哪里?**
```
running/
├── data/
│   ├── database.db       # 原引擎兼容 SQLite 数据库
│   └── webapp/
│       └── running_web.db # 当前 Web 平台主数据库
├── output/
│   └── tasks/            # 当前任务媒体输出目录
```

**Q: 如何测试分析流程?**
```bash
# 启动后端
cd backend
python app.py

# 使用 curl 测试
curl -X POST http://localhost:5000/api/upload \
  -F "file=@test_video.mp4" \
  -F "view_angle=side"
```

**Q: 前端如何处理大文件上传?**
- 使用 Element Plus 的 `el-upload` 组件
- 设置 `before-upload` 检查文件大小
- 显示上传进度

---

## 九、参考代码

### 9.1 后端入口示例 (backend/app.py)

```python
from flask import Flask
from flask_cors import CORS
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import API_CONFIG

app = Flask(__name__)
CORS(app)

# 注册蓝图
from routes.analysis import analysis_bp
from routes.history import history_bp

app.register_blueprint(analysis_bp, url_prefix='/api')
app.register_blueprint(history_bp, url_prefix='/api')

@app.route('/')
def index():
    return {'message': '跑步动作分析系统 API', 'version': '2.0.0'}

if __name__ == '__main__':
    app.run(
        host=API_CONFIG.get('host', '0.0.0.0'),
        port=API_CONFIG.get('port', 5000),
        debug=API_CONFIG.get('debug', True)
    )
```

### 9.2 分析服务示例 (backend/services/analysis_service.py)

```python
import sys
from pathlib import Path
import hashlib
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.config import POSE_CONFIG
from modules.video_processor import VideoProcessor
from modules.pose_estimator import create_pose_estimator_3d
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.temporal_model import TemporalModelAnalyzer
from modules.quality_evaluator import QualityEvaluator
from modules.ai_analyzer import AIAnalyzer


class AnalysisService:
    """分析服务"""

    def __init__(self):
        self.tasks = {}  # task_id -> task_info

    def create_task(self, video_path: str, view_angle: str, enable_3d: bool = True) -> str:
        """创建分析任务"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            'status': 'pending',
            'progress': 0,
            'stage': '等待处理',
            'video_path': video_path,
            'view_angle': view_angle,
            'enable_3d': enable_3d,
            'result': None
        }
        return task_id

    def run_analysis(self, task_id: str, progress_callback=None):
        """执行分析（同步）"""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        def update_progress(progress, stage):
            task['progress'] = progress
            task['stage'] = stage
            task['status'] = 'processing'
            if progress_callback:
                progress_callback(progress, stage)

        try:
            video_path = task['video_path']
            view_angle = task['view_angle']
            enable_3d = task['enable_3d']

            # 计算视频哈希
            video_hash = self._calculate_file_hash(video_path)

            # 1. 视频处理
            update_progress(10, '视频预处理中...')
            processor = VideoProcessor(video_path)
            video_info = processor.get_video_info()
            frames, fps = self._extract_frames(processor, video_info)

            # 2. 姿态估计
            update_progress(30, '姿态估计中...')
            keypoints, keypoints_3d, poses_3d, has_3d = self._estimate_pose(
                frames, view_angle, enable_3d
            )

            # 3. 运动学分析
            update_progress(60, '运动学分析中...')
            kinematic_results = self._analyze_kinematics(
                keypoints, fps, view_angle, poses_3d, keypoints_3d
            )

            # 4. 时序分析
            update_progress(70, '时序分析中...')
            temporal_results = self._analyze_temporal(
                keypoints, view_angle, kinematic_results, keypoints_3d
            )

            # 5. 质量评价
            update_progress(80, '质量评价中...')
            quality_results = self._evaluate_quality(
                kinematic_results, temporal_results, view_angle
            )

            # 6. AI 分析
            update_progress(90, 'AI 分析中...')
            ai_text = self._generate_ai_report(
                kinematic_results, temporal_results, quality_results, view_angle
            )

            # 整合结果
            result = {
                'video_info': video_info,
                'video_hash': video_hash,
                'view_angle': view_angle,
                'kinematic_analysis': kinematic_results,
                'temporal_analysis': temporal_results,
                'quality_evaluation': quality_results,
                'ai_analysis': ai_text,
                'pose_3d_info': {
                    'enabled': enable_3d,
                    'has_3d_data': has_3d
                }
            }

            task['result'] = result
            task['status'] = 'completed'
            task['progress'] = 100
            task['stage'] = '分析完成'

            processor.release()
            return result

        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            raise

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件 SHA256"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    # ... 其他辅助方法参考上面的完整流程
```

---

## 十、开发注意事项

1. **不要过度设计**：本科毕设以可演示、可复现为优先
2. **保持接口稳定**：分析引擎核心逻辑与评分口径不要修改
3. **追溯性第一**：每条记录都要可复现
4. **小步提交**：每完成一项功能就测试并记录
5. **3D 支持必须有**：这是核心功能，不能省略
6. **认证与权限是MVP必做**：至少完成注册/登录和用户数据隔离

---

*最后更新: 2026-02-11*



### 4.1 表结构

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
    progress INTEGER DEFAULT 0,
    stage TEXT,
    error_message TEXT,
    view_angle TEXT NOT NULL DEFAULT 'side',
    enable_3d INTEGER NOT NULL DEFAULT 1,
    input_video_path TEXT,
    result_record_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
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
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (task_id) REFERENCES analysis_tasks(id)
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

CREATE INDEX idx_tasks_user_status ON analysis_tasks(user_id, status);
CREATE INDEX idx_records_user_created ON analysis_records(user_id, created_at DESC);
CREATE INDEX idx_records_view_angle ON analysis_records(view_angle);
CREATE INDEX idx_records_total_score ON analysis_records(total_score);
CREATE INDEX idx_admin_audit_created_at ON admin_audit_logs(created_at DESC);
```


