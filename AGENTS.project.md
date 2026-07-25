# Worldline Social Project Rules

本文件是 `worldline-social` 的项目级协作规范。项目公开发布到 GitHub，默认分支固定为 `main`。它依赖 `worldline-engine`，但 Engine 不得反向导入本项目。

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
- README 中文章节在上、英文章节在下，命令和能力说明保持一致。
- 跨仓库引用使用绝对 GitHub URL，不能依赖同级相对路径。
- 推送前扫描密钥、私有配置、实验数据库、构建产物和 IDE 文件。
