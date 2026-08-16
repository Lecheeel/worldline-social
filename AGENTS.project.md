# Worldline Social Project Rules

本文件是 `worldline-social` 的项目级协作规范。项目公开发布到 GitHub，默认分支固定为 `main`。它依赖 `worldline-engine`，但 Engine 不得反向导入本项目。

## 项目哲学：世界线，而非单一世界

与 `worldline-engine` 保持一致：

- 一条世界线是一个二元组（manifest，录制流）：manifest 钉住定义运行的所有要素（engine/social 版本、人口哈希、模型、Prompt、策略），`seed` 只是随机流参数。同 seed 而 manifest 不同，就是两条不同的世界线；两条世界线只有在 manifest 只差一个字段时才是可比较的。
- 每次实验是一次世界线探索。`runner.build_simulation` 自动组装 worldline manifest，并由引擎作为 `worldline_manifest` 事件记录在事件流最前；人口哈希在运行时对 `PopulationManifest.to_mapping()` 的 canonical JSON 计算，不写入人口文件本身。
- 世界的价值不在"唯一正确的未来"，而在可能性的空间：同一事件存在许多合理走向，微小扰动（LLM 抽样、随机激活、参数微变）经系统放大产生宏观差异——蝴蝶效应。但发散是测量结果，不是结论：研究对象是分布的形状——什么结局在 seed 之间稳定、发散从哪里开始、哪些结局相互吸引。
- 比较的语义是：同一起点，看看世界能走向哪些不同的地方——更重要的是，看看什么不变。
- 方法论纪律：干预实验只允许改动 manifest 中的一个字段，其余全部固定；换 seed 重跑得到的是抽样噪声，不是干预效应。任何干预结论都需要零干预基线对照（同 manifest、不同 seed）。
- 社会模拟的确定性保证单条世界线内部的自洽与可审计（同 manifest、同输入、同 LLM 响应流 → 逐字节可重放）；LLM 是社会世界的主要分叉点，其响应应被记录、可审计。
- 认识论定位：本项目模拟的是"LLM 的民间社会学"，不是真实社会；输出是假设与排练，不是预测，结论永远相对于模型成立。该定位在 README 的"Epistemic status"一节公开声明。
- 一句话：Run one worldline. Compare many. Find what doesn't change.

## 项目边界

- Engine 保留 tick、turn、快照、提交、checkpoint 和事件语义。
- 本仓库拥有 PopulationManifest、SocialState、帖子、评论、关系、分发、动态状态、记忆、模型 Provider 和实验配置。
- Controller 只能返回结构化 `ActionIntent`，不得直接修改 SocialWorld 或其存储。
- 世界内观察只暴露公开 `handle`；`person_id` 仅用于内部状态、事件和动作绑定。
- 不把 API key、完整凭据、私有 Provider 请求或私有人格字段写入公开事件、checkpoint、日志或测试文件。
- 本项目不是 OASIS 兼容层，不迁移其 API、数据库 schema 或 Agent 实现。

## 本地命令

- 本地依赖：`python -m pip install -e ..\worldline-engine`
- 安装：`python -m pip install -e .`
- 向量扩展：`python -m pip install -e ".[vector]"`
- 测试：`python -m unittest discover -s tests -v`
- 静态检查：`python -m compileall -q src tests examples scripts`
- 构建：`python -m pip wheel . --wheel-dir dist --no-deps`

## 测试要求

- 影响人口导入、状态 schema、世界动作、分发、动态状态或恢复语义时，运行完整测试集。
- SocialWorld 必须在不同 Engine 并发度下保持确定性结果。
- 新增模型 Provider 必须添加不需要网络的 fake-provider 测试。
- 生成的 `runs/`、SQLite、缓存、wheel 和日志不得提交。

## Git 约定

- 默认分支为 `main`；首次初始化使用 `git init -b main`。
- 提交使用英文 Conventional Commits，例如 `feat(world): add deterministic reply actions`。
- 提交前检查 `git diff --check`、`.gitignore` 和待提交文件。
- 不自动重写历史，不使用 destructive reset，不在未确认远端设置时推送。

## GitHub 发布

- 仓库保持公开，许可证为 Apache-2.0，与 `worldline-engine` 一致。
- Description 使用：`中文描述 | English description`。
- README 默认使用英文，中文版为 `README.zh-CN.md`；两个版本顶部通过语言切换链接互跳，命令与能力说明保持一致。
- 跨仓库引用使用绝对 GitHub URL，不能依赖同级相对路径。
- 推送前扫描密钥、私有配置、实验数据库、构建产物和 IDE 文件。
