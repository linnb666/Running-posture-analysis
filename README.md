# 跑步动作分析系统

本项目核心目标是基于视频完成跑步动作分析、评分与报告生成。当前仓库同时保留了原始分析引擎与重构后的 Flask + Vue3 Web 平台。

## 一、项目结构定位

当前代码建议按两层理解：

1. 分析引擎层：`modules/`、`config/`、`web/streamlit_app.py`、`main.py`
2. Web 平台层：`backend/`、`frontend/`、`start_dev.py`、`stop_dev.py`

系统说明和功能介绍时，应以代码实现为准，并优先参考：

1. `docs/engine/ARCHITECTURE.md`
2. `docs/engine/MODULE_REFERENCE.md`
3. `docs/engine/DATA_SOURCE_ANALYSIS.md`
4. `docs/engine/PROJECT_DOCUMENTATION.md`

## 二、当前主要能力

- 基于 MediaPipe Pose 的 2D 姿态估计
- 基于 MotionBERT / DSTformer 的 2D 到 3D 姿态提升
- 支持侧面视角与正面视角的差异化分析
- 三维度评分：动作稳定性、跑步效率、跑姿标准度
- AI 报告手动触发生成
- 后端直出 PDF 分析报告
- 分析结果人工备注与备注时间记录
- 历史记录、结果对比、管理员管理与审计日志

## 三、运行入口

- 后端入口：`backend/app.py`
- 前端入口：`frontend/src/main.js`
- 开发启动脚本：`start_dev.py`
- 开发停止脚本：`stop_dev.py`
- 引擎命令行入口：`main.py`
- 原引擎 Web 界面：`web/streamlit_app.py`

## 四、目录概览

```text
running/
├── backend/                  # Flask 后端、数据库、路由、服务层
├── frontend/                 # Vue3 前端
├── modules/                  # 核心分析引擎
├── config/                   # 引擎配置
├── web/                      # Streamlit 原引擎界面
├── models/                   # 历史/研究模型定义
├── utils/                    # 可视化与工具函数
├── scripts/                  # 辅助脚本
├── docs/                     # 项目文档
├── data/                     # 数据库、日志、模型权重
├── output/                   # 任务输出目录
├── testVideos/               # 测试视频
├── main.py                   # 引擎命令行入口
├── start_dev.py              # 一键启动前后端
└── stop_dev.py               # 一键停止前后端
```

## 五、数据库与存储

### 1. Web 平台主数据库

- 路径：`data/webapp/running_web.db`
- ORM 定义：`backend/db_models.py`
- 主要表：
  - `users`
  - `analysis_tasks`
  - `analysis_records`
  - `admin_audit_logs`

### 2. 原引擎兼容数据库

- 路径：`data/database.db`
- 代码位置：`modules/database.py`
- 用途：供 Streamlit / CLI 旧链路兼容使用

### 3. 媒体输出目录

- 根目录：`output/tasks/<task_id>/`
- 常见产物：
  - `input_*.mp4`
  - `pose_overlay.mp4`
  - `pose_overlay_web.mp4`
  - `keyframes/*.jpg`

## 六、当前关键阈值

### 1. 垂直振幅

当前以“相对躯干长度百分比”表示：

- 优秀：`<= 11%`
- 良好：`11% - 17%`
- 一般：`17% - 25%`
- 待提升：`> 25%`

### 2. 躯干前倾

当前针对中长跑 / 马拉松场景采用稳健阈值：

- 最优：`4° - 8°`
- 良好：`3° - 10°`
- 一般：`2° - 12°`
- 过于直立：`< 2°`
- 过度前倾：`> 12°`

## 七、AI 与报告

- AI 报告默认不自动生成，需在结果页手动触发
- 当前文本模型：`glm-4.5-air`
- 外部 AI 调用失败时，可按配置回退到本地规则报告
- PDF 由后端直接生成，不是打印当前网页
- 人工备注保存于 `analysis_records.manual_notes`

## 八、开发启动

### 1. 一键启动

```bash
python start_dev.py
```

### 2. 一键停止

```bash
python stop_dev.py
```

### 3. 手动启动

后端：

```bash
cd backend
python app.py
```

前端：

```bash
cd frontend
npm run dev
```

## 九、说明

1. 代码是当前实现的唯一最终真值。
2. `docs/engine/` 下文档用于系统说明。
3. `docs/FRONTEND_PROMPT.md` 与 `docs/rebuild/` 下文档保留实现过程信息，但若与代码冲突，应以代码和 `docs/engine/` 为准。
