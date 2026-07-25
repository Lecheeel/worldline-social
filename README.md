# Worldline Social

Worldline Social 是构建在 [Worldline Engine](https://github.com/Lecheeel/worldline-engine) 之上的规范、可复现社会仿真系统。它是全新设计，不兼容也不迁移 OASIS 的 API、数据库 schema 或 Agent 实现；OASIS 仅用于理解实验问题与社会平台功能范围。

## 当前能力

- 版本化 JSON `PopulationManifest`：校验 handle、关系和来源，并确定性分配内部 `person_id`。
- 版本化 `SocialState`：支持 checkpoint 恢复与 v1 到 v2 状态迁移。
- 社会世界：帖子、评论、回复、feed、thread、广场搜索和帖子/评论点赞。
- 世界内观察只暴露公开 `handle`，内部 `person_id` 保留给状态、事件和动作绑定。
- 可替换的全量帖子与近期帖子分发策略。
- 有界人格与动态状态，并在每个 tick 后确定性恢复。
- Rule、Replay、LLM tool-call Controller，以及可选 memory/vector/provider 扩展。
- JSON 实验配置、SQLite checkpoint/event、CLI、conformance 测试和 GitHub CI。

## 本地开发

要求 Python 3.11 或更高版本。开发时先安装本地 Engine，再安装 Social：

```powershell
python -m pip install -e ..\worldline-engine
python -m pip install -e .
python -m unittest discover -s tests -v
python -m compileall -q src tests examples scripts
python -m pip wheel . --wheel-dir dist --no-deps
```

可选记忆和向量扩展：

```powershell
python -m pip install -e ".[vector]"
python -m pip install -e ".[embedding]"
```

## 运行示例实验

```powershell
worldline-social validate-population examples\population.json
worldline-social run examples\experiment.json
```

运行器会按实验配置文件所在目录解析路径，组装人口、Controller、分发策略、调度器、SQLite checkpoint/event 存储，并输出 JSON 摘要。只有数据库中已经存在相同 `simulation_id` 的 checkpoint 时，才使用 `--resume`。

## 许可证

Apache License 2.0，与 [Worldline Engine](https://github.com/Lecheeel/worldline-engine) 保持一致。详见 [LICENSE](LICENSE)。

---

## Overview

Worldline Social is a clear, replayable social simulation system built on [Worldline Engine](https://github.com/Lecheeel/worldline-engine). It is a clean design rather than an OASIS compatibility or migration effort. OASIS is used only as a reference for research questions and social-platform scope.

## Current Capabilities

- Versioned JSON `PopulationManifest` validation and deterministic internal `person_id` assignment.
- Versioned `SocialState` with checkpoint recovery and v1-to-v2 migration.
- Posts, comments, replies, feeds, threads, square search, and post/comment likes.
- Public observations expose `handle` values only; internal IDs remain in state, events, and bound actions.
- Replaceable all-posts and recent-posts distribution policies.
- Bounded trait and dynamic-state models with deterministic per-tick recovery.
- Rule, replay, and LLM tool-call controllers with optional memory/vector/provider extensions.
- JSON experiment configuration, SQLite checkpoint/events, CLI, conformance tests, and GitHub CI.

## Local Development

Python 3.11 or newer is required. Install the local Engine before Social during development:

```powershell
python -m pip install -e ..\worldline-engine
python -m pip install -e .
python -m unittest discover -s tests -v
python -m compileall -q src tests examples scripts
python -m pip wheel . --wheel-dir dist --no-deps
```

Optional memory and vector integrations:

```powershell
python -m pip install -e ".[vector]"
python -m pip install -e ".[embedding]"
```

## Run the Example

```powershell
worldline-social validate-population examples\population.json
worldline-social run examples\experiment.json
```

The runner resolves paths relative to its configuration file, composes the population, controllers, distribution policy, scheduler, and SQLite checkpoint/event stores, then prints a JSON summary. Use `--resume` only when a checkpoint for the same `simulation_id` already exists.

## License

Apache License 2.0, matching [Worldline Engine](https://github.com/Lecheeel/worldline-engine). See [LICENSE](LICENSE).
