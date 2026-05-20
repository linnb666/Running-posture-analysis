# 跑步动作分析系统

本项目是一个跑步动作分析工程，目标是基于视频完成姿态估计、运动学分析、质量评分与报告生成。仓库同时保留了原始分析引擎，以及重构后的 Flask + Vue 3 Web 平台，适合用于课程设计、论文支撑、功能演示和后续二次开发。

## 项目概览

- 输入：跑步视频，支持侧面视角和正面视角
- 处理链路：2D 姿态估计 -> 3D 姿态提升 -> 运动学分析 -> 评分 -> 报告生成
- 输出：综合评分、关键指标、AI/规则分析文本、关键帧、姿态可视化视频、PDF 报告
- 交互方式：
  - 原始引擎：命令行 + Streamlit
  - Web 平台：Flask 后端 + Vue 3 前端

## 核心能力

- 基于 MediaPipe Pose 的 2D 姿态估计
- 基于 MotionBERT / DSTformer 的 2D 到 3D 姿态提升
- 区分侧面视角与正面视角的差异化分析策略
- 三维度评分体系：稳定性、效率、跑姿
- 分析结果历史记录、结果对比、人工备注与管理员管理
- 关键帧提取、姿态可视化视频生成、PDF 报告导出
- 支持智谱 AI 生成分析文本，并可回退到本地规则报告

## 系统结构

当前代码建议按两层理解：

1. 分析引擎层  
   主要位于 `modules/`、`config/`、`web/streamlit_app.py`、`main.py`，负责姿态估计、3D 提升、运动学分析、评分和规则报告生成。

2. Web 平台层  
   主要位于 `backend/`、`frontend/`、`start_dev.py`、`stop_dev.py`，负责用户登录、任务提交、结果展示、历史管理和 PDF 导出。

## 技术栈

- 后端：Flask、SQLAlchemy、PyJWT
- 前端：Vue 3、Vite、Pinia、Vue Router、Element Plus、ECharts
- 视觉与分析：OpenCV、MediaPipe、NumPy、SciPy、PyTorch
- 原型界面：Streamlit
- 数据存储：SQLite
- AI 文本分析：智谱 AI（可选）

## 仓库结构

```text
running/
├── backend/                    # Flask 后端、数据库、路由、服务层
├── frontend/                   # Vue 3 前端
├── modules/                    # 核心分析引擎
├── config/                     # 引擎配置
├── web/                        # Streamlit 原始界面
├── models/                     # 历史/研究模型定义
├── utils/                      # 可视化与通用工具
├── docs/                       # 项目文档
├── data/                       # 数据目录（数据库、日志、模型权重等）
├── output/                     # 任务输出目录
├── main.py                     # 引擎命令行入口
├── start_dev.py                # 一键启动前后端
├── stop_dev.py                 # 一键停止前后端
├── requirements.txt            # Python 依赖
└── README.md
```

## 模型权重

仓库默认不直接包含大体积模型权重，请从 [GitHub Releases](https://github.com/linnb666/Running-posture-analysis/releases) 下载对应的 `.pth` 文件，并放置到：

```text
data/checkpoints/
```

当前默认配置的权重文件名为：

```text
ap3d_rm_v2_best.pth
```

如果权重文件缺失，3D 姿态提升相关链路将无法按预期工作。

## 快速开始

### 1. 准备环境

- Python 环境：确保可以正常安装 `requirements.txt` 中的依赖
- Node.js 环境：用于运行前端开发服务器

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

如果需要启用智谱 AI 文本分析，还需要额外安装：

```bash
pip install zai
```

安装前端依赖：

```bash
cd frontend
npm install
```

### 2. 下载并放置模型权重

从仓库的 Releases 页面下载权重文件后，放到：

```text
data/checkpoints/ap3d_rm_v2_best.pth
```

### 3. 配置环境变量

建议至少配置以下变量：

```bash
ZHIPU_API_KEY=你的智谱密钥
RUNNING_SECRET_KEY=你自己的后端密钥
RUNNING_ADMIN_USERNAME=admin
RUNNING_ADMIN_PASSWORD=请改成自己的密码
RUNNING_ADMIN_EMAIL=your_email@example.com
```

说明：

- `ZHIPU_API_KEY` 未配置时，系统仍可运行，但 AI 文本分析会回退到本地规则报告
- 后端默认开发管理员账号来自环境变量；若不覆盖，代码中的默认值仅适合本地调试

### 4. 启动项目

一键启动：

```bash
python start_dev.py
```

一键停止：

```bash
python stop_dev.py
```

手动启动后端：

```bash
python backend/app.py
```

手动启动前端：

```bash
cd frontend
npm run dev
```

默认访问地址：

- 后端：`http://127.0.0.1:5000`
- 前端：`http://127.0.0.1:5173`

## 运行入口

- 后端入口：`backend/app.py`
- 前端入口：`frontend/src/main.js`
- 开发启动脚本：`start_dev.py`
- 开发停止脚本：`stop_dev.py`
- 引擎命令行入口：`main.py`
- 原始 Streamlit 界面：`web/streamlit_app.py`

## 数据库与输出

### Web 平台主数据库

- 路径：`data/webapp/running_web.db`
- ORM 定义：`backend/db_models.py`
- 主要表：
  - `users`
  - `analysis_tasks`
  - `analysis_records`
  - `admin_audit_logs`

### 原引擎兼容数据库

- 路径：`data/database.db`
- 代码位置：`modules/database.py`
- 用途：供 Streamlit / CLI 旧链路兼容使用

### 任务输出目录

```text
output/tasks/<task_id>/
```

常见输出包括：

- `input_*.mp4`
- `pose_overlay.mp4`
- `pose_overlay_web.mp4`
- `keyframes/*.jpg`
- 分析结果相关 JSON / 数据库存档

## 当前关键阈值

### 垂直振幅

当前以相对躯干长度百分比表示：

- 优秀：`<= 11%`
- 良好：`11% - 17%`
- 一般：`17% - 25%`
- 待提升：`> 25%`

### 躯干前倾

当前面向中长跑 / 马拉松场景采用稳健阈值：

- 最优：`4° - 8°`
- 良好：`3° - 10°`
- 一般：`2° - 12°`
- 过于直立：`< 2°`
- 过度前倾：`> 12°`

## AI 与报告

- AI 报告默认不自动生成，需在结果页手动触发
- 当前默认文本模型：`glm-4.5-air`
- 外部 AI 调用失败时，可按配置回退到本地规则报告
- PDF 报告由后端直接生成，不是打印网页
- 人工备注保存于 `analysis_records.manual_notes`

## 推荐阅读文档

如果你需要把本项目用于说明文档、系统介绍或后续重构，优先参考以下文档：

1. `docs/engine/ARCHITECTURE.md`
2. `docs/engine/MODULE_REFERENCE.md`
3. `docs/engine/DATA_SOURCE_ANALYSIS.md`
4. `docs/engine/PROJECT_DOCUMENTATION.md`

补充说明：

- `docs/engine/` 下文档适合用于系统说明
- `docs/FRONTEND_PROMPT.md` 与 `docs/rebuild/` 下文档保留了实现过程信息
- 当文档与代码冲突时，应以当前代码实现为准

## 注意事项

- 本项目以研究和课程设计场景为主，依赖本地环境、模型权重和视频输入条件
- 大文件权重未直接纳入仓库历史，请以 Releases 中的版本为准
- 若用于公开演示或部署，建议先替换默认密钥、管理员账号和管理员密码
- 当前代码是项目行为的最终真值，论文或说明材料应尽量贴近代码实现
