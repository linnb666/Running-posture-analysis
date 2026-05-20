# 文档结构说明

项目文档分为“当前实现文档”“重构过程文档”“历史归档文档”三层。写论文、做系统说明或核对实现时，请按下面顺序阅读。

## 一、推荐阅读顺序

1. **代码为最终真值**
   - `backend/`
   - `frontend/`
   - `modules/`
   - `config/`
2. **当前主文档**
   - `README.md`
   - `docs/engine/ARCHITECTURE.md`
   - `docs/engine/MODULE_REFERENCE.md`
   - `docs/engine/DATA_SOURCE_ANALYSIS.md`
   - `docs/engine/PROJECT_DOCUMENTATION.md`
3. **实现过程文档**
   - `docs/FRONTEND_PROMPT.md`
   - `docs/rebuild/REBUILD_PROMPT_v1.md`
4. **历史参考文档**
   - `docs/rebuild/FRONTEND_PROMPT_legacy.md`
   - `docs/archive/**`

## 二、各层级用途

### 1. 当前主文档

路径：`docs/engine/`

- `ARCHITECTURE.md`：系统分层、数据流、接口与数据库结构
- `MODULE_REFERENCE.md`：核心模块职责与输入输出说明
- `DATA_SOURCE_ANALYSIS.md`：指标来源、计算方式、数据口径说明
- `PROJECT_DOCUMENTATION.md`：项目级整合说明

这些文档应与当前代码保持同步，是论文写作的主要参考材料。

### 2. 实现过程文档

路径：

- `docs/FRONTEND_PROMPT.md`
- `docs/rebuild/REBUILD_PROMPT_v1.md`

这类文档记录的是前后端重构过程中的约束、设计意图和实现提示。它们仍有参考价值，但若字段、路径、接口、数据库结构与当前代码冲突，应以代码和 `docs/engine/` 为准。

### 3. 历史参考文档

路径：

- `docs/rebuild/FRONTEND_PROMPT_legacy.md`
- `docs/archive/**`

此类文档保留历史方案、讨论记录、旧版本实验与回滚参考，不应直接作为当前系统说明或论文正文依据。

## 三、使用原则

1. 论文中引用系统结构、数据库、接口、阈值时，先核对代码，再核对 `docs/engine/`。
2. 若 prompt 文档与代码不一致，应优先修正文档，不直接沿用旧描述。
3. 历史文档保留归档价值，但不承担“当前实现说明”的职责。
