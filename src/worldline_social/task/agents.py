"""LLM agents for the task workflow: goal decomposition and world design.

Both agents speak JSON through the provider-neutral chat-completions
contract, so they work with any ``ModelProvider`` (and are unit-testable
with scripted fakes).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..generation.json_utils import extract_json_object
from ..providers.base import CompletionRequest, ModelMessage, ModelProvider
from ..stats import UsageRecord, UsageRecorder
from .models import Decomposition, ResearchQuestion, WorldDesign

DECOMPOSE_SYSTEM_PROMPT = """你是一个社会模拟实验设计师。用户给你一个研究目标，你需要把它拆解成一个可执行的模拟实验方案。

你的拆解必须包含三部分：

1. **research_questions**：为了构建这个模拟世界，需要搜索哪些资料？每个问题是一条 web 搜索查询（可以直接提交给搜索引擎），例如"某公司 2025 年 财务 丑闻 时间线"。列出 3-8 个问题，按重要性排序。每个问题给出一句话 rationale（为什么需要这条资料）。

2. **architecture_requirements**：世界架构要求。描述这个模拟世界需要怎样的背景设定、事件脚本和世界规则（2-4 句话）。

3. **cast_requirements**：主体清单要求。这个模拟世界需要哪些类型的"能发声的主体"（个人/机构/媒体/政府等），以及他们之间应有的关系结构（2-4 句话）。

## 输出格式

只输出一个 JSON 对象：

{
  "research_questions": [
    {"id": "q1", "question": "搜索查询语句", "rationale": "为什么需要"}
  ],
  "architecture_requirements": "世界架构要求描述",
  "cast_requirements": "主体清单要求描述"
}

## 要求

- 搜索查询要具体、可直接搜索，不要空泛。
- 如果用户提供了种子材料，查询应围绕材料中的实体与争议展开。
"""

DECOMPOSE_USER_TEMPLATE = """## 用户目标

{goal}

## 种子材料（用户提供的背景资料，可能为空）

{seed_material}

请按系统提示的要求输出拆解 JSON。"""

DESIGN_SYSTEM_PROMPT = """你是一个社会模拟世界架构师。基于用户目标、种子材料和研究资料，设计模拟世界的架构。

你的设计必须包含：

1. **title**：世界的简短标题（一句话）。
2. **background**：世界背景设定（3-5 句话）：这个世界的舞台、当前局势、关键实体。
3. **event_script**：事件脚本。世界开始时已经发生的事件/正在发酵的争议，以及初始引爆点（80 字以内，将成为模拟世界的第一条帖子）。
4. **world_rules**：世界规则（2-4 条，每条一句话）：平台机制、互动模式、叙事约束等。
5. **required_actors**：必须出现在模拟中的主体名单（3-10 个），每个用"名称：角色"格式描述，例如 "某某公司：涉事企业"。

## 输出格式

只输出一个 JSON 对象：

{
  "title": "世界标题",
  "background": "背景设定",
  "event_script": "事件脚本与引爆点",
  "world_rules": "世界规则",
  "required_actors": ["主体1：角色", "主体2：角色"]
}
"""

DESIGN_USER_TEMPLATE = """## 用户目标

{goal}

## 种子材料

{seed_material}

## 研究资料（web 搜索结果摘要）

{research_notes}

## 拆解要求（来自分解阶段）

架构要求：{architecture_requirements}
主体要求：{cast_requirements}

请按系统提示的要求输出世界架构 JSON。"""


class DecomposeAgent:
    """Turn the user goal into research questions and design requirements."""

    def __init__(
        self,
        provider: ModelProvider,
        model: str,
        *,
        max_attempts: int = 3,
        usage_recorder: UsageRecorder | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_attempts = max(1, max_attempts)
        self._usage_recorder = usage_recorder

    async def decompose(self, goal: str, seed_material: str = "") -> Decomposition:
        for attempt in range(self._max_attempts):
            response = await self._provider.complete(
                CompletionRequest(
                    model=self._model,
                    messages=(
                        ModelMessage("system", DECOMPOSE_SYSTEM_PROMPT),
                        ModelMessage(
                            "user",
                            DECOMPOSE_USER_TEMPLATE.format(
                                goal=goal, seed_material=seed_material or "（无）"
                            ),
                        ),
                    ),
                    temperature=0.2,
                    max_tokens=2048,
                    thinking="disabled",
                )
            )
            self._record(response)
            parsed = extract_json_object(response.content)
            if parsed is not None:
                decomposition = Decomposition.from_mapping(parsed)
                if decomposition.research_questions:
                    return decomposition
        raise ValueError("decomposition agent failed to produce a valid plan")

    def _record(self, response: Any) -> None:
        if self._usage_recorder is not None and getattr(response, "usage", None):
            self._usage_recorder(
                UsageRecord(
                    simulation_id="",
                    tick_id=-1,
                    entity_id="task:decompose",
                    model=self._model,
                    phase="generation",
                    provider_request_id=response.provider_request_id,
                    usage=dict(response.usage),
                )
            )


class WorldDesignAgent:
    """Synthesize the world architecture from research material."""

    def __init__(
        self,
        provider: ModelProvider,
        model: str,
        *,
        max_attempts: int = 3,
        usage_recorder: UsageRecorder | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_attempts = max(1, max_attempts)
        self._usage_recorder = usage_recorder

    async def design(
        self,
        goal: str,
        seed_material: str,
        research_notes: Mapping[str, Any],
        decomposition: Decomposition,
    ) -> WorldDesign:
        notes_text = _render_research_notes(research_notes)
        for attempt in range(self._max_attempts):
            response = await self._provider.complete(
                CompletionRequest(
                    model=self._model,
                    messages=(
                        ModelMessage("system", DESIGN_SYSTEM_PROMPT),
                        ModelMessage(
                            "user",
                            DESIGN_USER_TEMPLATE.format(
                                goal=goal,
                                seed_material=seed_material or "（无）",
                                research_notes=notes_text or "（无）",
                                architecture_requirements=decomposition.architecture_requirements or "（无）",
                                cast_requirements=decomposition.cast_requirements or "（无）",
                            ),
                        ),
                    ),
                    temperature=0.3,
                    max_tokens=2048,
                    thinking="disabled",
                )
            )
            self._record(response)
            parsed = extract_json_object(response.content)
            if parsed is not None:
                design = WorldDesign.from_mapping(parsed)
                if design.title and design.event_script:
                    return design
        raise ValueError("world design agent failed to produce a valid design")

    def _record(self, response: Any) -> None:
        if self._usage_recorder is not None and getattr(response, "usage", None):
            self._usage_recorder(
                UsageRecord(
                    simulation_id="",
                    tick_id=-1,
                    entity_id="task:architect",
                    model=self._model,
                    phase="generation",
                    provider_request_id=response.provider_request_id,
                    usage=dict(response.usage),
                )
            )


def render_event_text(design: WorldDesign, research_notes: Mapping[str, Any], seed_material: str = "") -> str:
    """Render world design + research into the event text the generation
    pipeline consumes (extract actors -> profiles -> manifest)."""
    notes_text = _render_research_notes(research_notes)
    actors = "\n".join(f"- {actor}" for actor in design.required_actors)
    parts = [
        f"【世界标题】{design.title}",
        f"【背景设定】\n{design.background}",
        f"【事件脚本】\n{design.event_script}",
        f"【世界规则】\n{design.world_rules}",
        f"【必须出现的主体】\n{actors or '（由抽取器自行判断）'}",
        f"【研究资料】\n{notes_text or '（无）'}",
    ]
    if seed_material.strip():
        parts.append(f"【用户提供的种子材料】\n{seed_material}")
    return "\n\n".join(part for part in parts if part)


def _render_research_notes(research_notes: Mapping[str, Any], max_chars: int = 4000) -> str:
    """Render research notes compactly so extraction prompts stay focused.

    Search summaries can be very long; truncate each note and cap the total
    to keep the downstream extraction call stable.
    """
    notes = research_notes.get("notes", ()) if isinstance(research_notes, dict) else ()
    parts: list[str] = []
    budget = max_chars
    for note in notes:
        if not isinstance(note, dict):
            continue
        query = str(note.get("query", ""))
        summary = str(note.get("summary", "")).strip()
        error = note.get("error")
        if error:
            block = f"- 查询「{query}」失败：{error}"
        elif summary:
            block = f"- 查询「{query}」：{_truncate(summary, 700)}"
        else:
            continue
        if len(block) > budget:
            break
        parts.append(block)
        budget -= len(block)
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…（已截断）"
