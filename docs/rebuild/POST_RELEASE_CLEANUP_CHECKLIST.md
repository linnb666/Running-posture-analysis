# 收尾统一清扫清单（暂缓执行）

## 说明
- 本文档仅用于登记遗留清扫项。
- 当前阶段不执行清理，避免影响在研功能与联调。
- 触发时机：项目功能开发与论文材料定稿后，进入最终收尾阶段统一执行。

## 待清扫项（2026-02-15 记录）

1. 非 output/tasks 路径的历史产出物未纳入当前 WebApp 管理清理链路
- 现状：当前后台硬删除与孤儿清理主要绑定 output/tasks + 数据库索引。
- 待清扫目录/文件：
  - output/videos
  - output/keyframes
  - output/visualizations
  - output/ap3d_eval
  - output/pose_accuracy_test
  - output/runtime_logs
  - output/skeleton_3d.mp4
  - output/ai_analysis_report.txt
  - output/full_regression_summary.json
  - output/last_error.txt

2. 旧引擎历史输出目录未接入当前后台清理策略
- 目录：data/outputs
- 说明：该目录为旧流程沉淀产物，当前 Flask + Vue 主链路不依赖该目录索引。

3. 数据索引一致性遗留（需收尾修复）
- 现状：存在少量 analysis_tasks.result_record_id 指向已不存在记录的情况。
- 处理目标：
  - 收尾阶段做一次数据库一致性修复（清理悬挂引用）。
  - 建立 记录删除 -> 任务引用同步置空/修复规则。

## 收尾执行原则
- 不改动后处理分析引擎核心逻辑。
- 先备份数据库与媒体目录，再执行清扫。
- 清扫后执行回归检查：
  - 管理员批量删除
  - 历史记录读取
  - 分析结果媒体访问（原视频、姿态视频、关键帧）
  - 审计日志可追溯性

## 验收标准
- 管理端删除后，数据库索引与媒体目录无孤儿残留。
- 所有媒体路径可追溯到有效记录或任务；无悬挂引用。
- 清扫不影响前台分析、回看、对比、AI报告功能。
