# Worldline Social

[![CI](https://github.com/Lecheeel/worldline-social/actions/workflows/ci.yml/badge.svg)](https://github.com/Lecheeel/worldline-social/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

> **如果一群人拥有记忆、关系和选择，他们会把世界带向哪里？**

Worldline Social 是一个可复现的社会仿真实验场。你可以定义一群人，让他们在同一个数字社会中发帖、评论、建立关系、接收信息，并观察情绪、观点和行为如何随着时间彼此影响。

它不是一个静态的数据集，也不是一段只能运行一次的演示。每次实验都有明确的人口、规则、时间线和事件记录：你可以从同一个起点重新开始，替换一种分发策略，换一组 Agent，或者回到某个 checkpoint，看看世界会不会走向另一条路。

## 这里可以探索什么

- **观点如何扩散**：同一条信息，在不同的 feed 和关系网络中会走多远？
- **关系如何改变行为**：关注、回应和互动会怎样影响一个人的下一步选择？
- **情绪如何进入公共空间**：压力、愤怒、疲劳与恢复，如何改变表达和参与？
- **Agent 如何形成连续性**：有记忆的角色，和只看当前信息的角色，会做出怎样不同的决定？
- **规则如何塑造结果**：改变平台规则、人口构成或行动预算，世界会发生什么变化？

## 一个小世界，也足够有趣

```text
导入人物 → 让他们行动 → 推进时间 → 保存世界
    ↑                         ↓
    └──────── 回放 / 比较 ────────┘
```

项目提供从人口清单到实验配置的完整入口，并支持规则控制、回放控制和可选的 LLM、记忆与向量扩展。你可以先从一个两三个人的小实验开始，再逐步增加关系、动态状态和更复杂的决策者。

你也可以直接阅读 [示例人口](examples/population.json)、[实验配置](examples/experiment.json) 和 [最小运行脚本](examples/social_simulation.py)，从一个只有 Alice 和 Bob 的小世界开始。

## 把好奇心变成实验

从一个小问题开始：一条关于公共议题的帖子，会被谁看见？谁会回应？当一个人的情绪、关系和记忆都发生变化时，下一条选择还会一样吗？Worldline Social 让这些变化有迹可循，也让不同答案可以在同一个起点上被比较。

## 快速开始

需要 Python 3.11 或更高版本。开发时先安装 Engine，再安装 Social：

```powershell
python -m pip install -e ..\worldline-engine
python -m pip install -e .
worldline-social validate-population examples\population.json
worldline-social run examples\experiment.json
```

实验完成后会输出摘要，并将 checkpoint 与事件写入配置指定的位置。使用 `--resume` 可以从已有 checkpoint 继续。

首次运行会创建实验数据库；重复开始同一个实验前，请先移除 `runs\example.sqlite`，或使用 `--resume` 从已有 checkpoint 继续。

## 项目关系

Worldline Social 建立在 [Worldline Engine](https://github.com/Lecheeel/worldline-engine) 之上。Engine 提供稳定的时间线和执行语义，Social 在此基础上定义人物、社会动作、信息分发和动态状态。两者保持独立，方便把同一套运行内核用于其他类型的世界。

## 项目状态

这是一个持续构建中的研究型开源项目。当前重点是让社会实验可复现、可恢复、可比较，并为更丰富的人口、记忆和 Agent 行为留下清晰的扩展空间。

## License

Apache License 2.0，与 [Worldline Engine](https://github.com/Lecheeel/worldline-engine) 保持一致。详见 [LICENSE](LICENSE)。

---

## Overview

> **Give people memory, relationships, and choices. Then see where the world goes.**

Worldline Social is a reproducible laboratory for social simulation. Populate a shared digital society, let agents post, reply, form relationships, receive information, and observe how behavior, mood, and opinions influence one another over time.

This is more than a one-off demo or a static dataset. Every experiment has an explicit population, rule set, timeline, and event trail. Start from the same point, change the feed policy or the agents, resume from a checkpoint, and see whether the world takes a different path.

## Questions Worth Exploring

- How far does an idea travel through different feeds and relationship networks?
- How do follows, replies, and interactions change a person’s next decision?
- How do stress, anger, fatigue, and recovery shape public expression?
- What changes when agents remember the past instead of seeing only the present?
- How much of an outcome comes from the people, and how much from the rules?

## Quick Start

Python 3.11 or newer is required. During development, install the Engine first:

```powershell
python -m pip install -e ..\worldline-engine
python -m pip install -e .
worldline-social validate-population examples\population.json
worldline-social run examples\experiment.json
```

The example prints a summary and writes checkpoints and events to the configured location. Use `--resume` to continue from an existing checkpoint.

The first run creates the experiment database. To start the same experiment from scratch again, remove `runs\example.sqlite`; use `--resume` when you want to continue an existing run.

## Turn Curiosity Into an Experiment

Start with a small question: who sees a post about a public issue, who responds, and does the next choice change when mood, relationships, and memory change? Worldline Social keeps those changes traceable and makes alternative answers comparable from the same starting point.

## Project Relationship

Worldline Social is built on [Worldline Engine](https://github.com/Lecheeel/worldline-engine). Engine provides the timeline and execution semantics; Social defines people, social actions, information flow, and evolving state. The two projects remain independent so the same execution core can power many kinds of worlds.

## Project Status

Worldline Social is an evolving research-oriented open-source project. Current work focuses on reproducible, resumable, and comparable social experiments, with room for richer populations, memory, and agent behavior.

## License

Apache License 2.0, matching [Worldline Engine](https://github.com/Lecheeel/worldline-engine). See [LICENSE](LICENSE).
