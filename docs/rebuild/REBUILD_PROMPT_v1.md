# 本科毕设前后端重构 Prompt（v1）

> 说明：本文档保留重构执行期的提示信息；若与当前代码冲突，应以代码和 `docs/engine/` 下主文档为准。

> 目标：在不改动分析引擎核心算法的前提下，重做 Flask + Vue3 壳层，使系统稳定可演示、可追溯、可维护。

## 一、项目定位与硬约束

1. 项目定位：本科毕设工程，不是商业 Demo。
2. 技术栈不变：Flask + SQLAlchemy + Vue3 + Element Plus + ECharts。
3. 禁止改动 `modules/` 和 `main.py` 的核心分析逻辑（后处理分析引擎保持原样）。
4. 前端视觉风格保持当前项目样式基调，允许在该基调上优化细节。

## 二、数据流真相（以原引擎 web 为标准）

以 `web/streamlit_app.py` 作为业务标准流：

1. 七阶段流程：预处理 -> 姿态估计 -> 视角设置 -> 运动学 -> 时序 -> 质量评价 -> 报告。
2. 媒体产物走文件系统，不走数据库 BLOB：
   - 姿态识别视频：`output/tasks/<task_id>/pose_overlay.mp4` 或 `pose_overlay_web.mp4`
   - 关键帧骨架图：`output/tasks/<task_id>/keyframes/*`
3. 结果页展示应从输出目录读取媒体文件并通过后端媒体接口提供 URL。
4. AI 报告默认不自动生成，用户点击按钮后再触发。

## 三、后端重构目标（Flask）

1. 任务状态机重构（必须避免卡死）：
   - `queued -> running -> persisting -> succeeded/failed`
2. 接口重构：
   - 认证：注册/登录/刷新/登出
   - 分析：上传、任务状态、结果详情、历史、统计、删除
   - AI：单独按钮触发 `/api/result/{record_id}/ai`
   - 媒体：原视频、姿态视频、关键帧文件访问
3. 媒体策略：
   - 每个 task 独立输出目录
   - 数据库保存媒体路径元数据，不存文件体
4. 事务策略：
   - 记录写入与任务完结状态分阶段提交
   - 出错必须 `rollback + failed + error_message`

## 四、前端重构目标（Vue3）

1. Analyze 页面：
   - 上传 + 视角选择 + 开始分析
   - 7 节点进度与后端 stage 同步
2. Result 页面：
   - 原始视频、姿态识别视频、关键帧骨架图
   - 正面/侧面分流展示不同指标图表
   - 顶部信息卡展示视角、分析时间、模型版本，并内置后端直出 PDF 按钮
   - 右下角浮动人工备注控件，备注单独保存并记录最近保存时间
   - PDF 必须由后端直接生成正式报告，内嵌维度雷达图、维度条形图、核心指标、关键帧和人工备注，并处理头部元信息换行、文本留边、正面关键帧保留原样、侧面关键帧主体裁切和正文分页
   - 维度评分卡下提供“详细指标”弹窗
   - AI 分析按钮独立，默认不自动展示
3. 错误与降级：
   - 媒体加载失败不阻断文本与图表
   - 任务失败显示可读错误与重试入口

## 五、字段契约统一（必须）

统一返回字段命名，避免多套别名长期共存：

- 统一主字段：`view_angle`、`created_at`、`dimension_scores`、`quality_results`、`kinematic_results`、`temporal_results`。
- 旧字段（如 `view_type`、`completed_at`）仅保留兼容映射层，不扩散到新代码。

## 六、数据库建议

默认继续 SQLite（WAL 模式）用于毕设阶段。

- 优点：部署简单、调试成本低、单机稳定。
- 若后续答辩演示多端并发再考虑 PostgreSQL 迁移。

## 七、验收标准

使用 `testVideos/` 至少完成正面 + 侧面各 1 条全链路验证：

1. 注册 -> 登录 -> 上传 -> 轮询 -> 结果展示。
2. 验证项：
   - 不再卡 `98%`
   - 进度节点与阶段同步
   - 姿态视频可播放
   - 关键帧为 2D 骨架
   - 正面/侧面结果分流正确
   - AI 默认不自动，按钮触发可用

## 八、回滚与推翻预案

1. 保留旧 Prompt：`docs/rebuild/FRONTEND_PROMPT_legacy.md`
2. 任何大改前，先记录：
   - 当前可用分支状态
   - 接口契约快照
   - 前端页面截图基线
3. 若新方案不稳定，可快速回退到旧 Prompt + 旧接口契约。

## 九、执行纪律

1. 每轮先审查，再修改，再测试。
2. 小步提交，避免一次性大改。
3. 遇到阻塞先给根因和验证路径，不盲改。

## 十、管理员 MVP（新增）

在现有重构基线上，补充管理员最小可用能力：

1. 账号与权限
   - `users` 增加 `is_admin`、`is_active`
   - 启动时确保默认管理员账号存在（支持环境变量覆盖）
   - 受保护接口统一校验 `is_active`
2. 管理接口
   - `GET /api/admin/overview`
   - `GET /api/admin/users`
   - `PATCH /api/admin/users/{user_id}`
   - `POST /api/admin/users/{user_id}/reset-password`
   - `GET /api/admin/users/{user_id}/records`
   - `DELETE /api/admin/records/{record_id}/hard-delete`
   - `POST /api/admin/records/hard-delete-batch`
   - `DELETE /api/admin/users/{user_id}/hard-delete`
   - `POST /api/admin/users/hard-delete-batch`
   - `GET /api/admin/audit-logs`
3. 硬清除要求
   - 删除用户关联数据库数据（用户/任务/记录）
   - 删除媒体与文本产物（原视频、姿态视频、关键帧等文件）
   - 仅允许白名单目录删除，避免路径越权
4. 前端管理页
   - 管理员专属入口与路由守卫
   - 用户状态/角色管理、密码重置
   - 记录级与用户级硬清除，支持批量操作（高风险二次确认）
   - 审计日志可回看，带中文动作名与中文解释



