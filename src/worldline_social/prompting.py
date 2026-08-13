"""Runtime prompt builders that carry persona, traits and state into LLM calls.

The generation pipeline writes rich persona fields into ``model_policy``;
this module renders them (plus the live ``dynamic_state``) into the system
message of every LLM turn, so an agent behaves consistently with its persona.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from worldline_engine.protocols import TurnContext

from .providers.base import ModelMessage

_TRAIT_LABELS: dict[str, str] = {
    "openness": "开放性",
    "conscientiousness": "尽责性",
    "extraversion": "外向性",
    "agreeableness": "宜人性",
    "neuroticism": "神经质",
    "honesty_humility": "诚实-谦逊",
    "machiavellianism": "马基雅维利主义",
    "narcissism": "自恋",
    "psychopathy": "精神病态",
}

_STANCE_LABELS: dict[str, str] = {
    "supportive": "支持",
    "opposing": "反对",
    "neutral": "中立",
    "observer": "旁观",
}

_STATE_LABELS: dict[str, str] = {
    "mood": "情绪（-1 负面 ~ +1 正面）",
    "anger": "愤怒",
    "stress": "压力",
    "fatigue": "疲劳",
    "threat": "威胁感",
}


def render_person_system_message(person: Mapping[str, Any]) -> str:
    """Render a person's identity, persona, traits and live state as a system message."""
    policy = person.get("model_policy", {}) or {}
    lines = [
        f"你是 {person.get('display_name') or person.get('handle')}"
        f"（@{person.get('handle')}）。",
        person.get("bio") or "",
    ]
    persona = str(policy.get("persona", "")).strip()
    if persona:
        lines.append(f"\n## 人设\n{persona}")
    stance = str(policy.get("stance", "")).strip()
    if stance:
        label = _STANCE_LABELS.get(stance, stance)
        lines.append(f"\n## 立场\n你对事件核心争议持{label}立场。")
    topics = policy.get("interested_topics") or ()
    if topics:
        lines.append("\n## 感兴趣的话题\n" + "\n".join(f"- {topic}" for topic in topics))
    memory = str(policy.get("personal_memory", "")).strip()
    if memory:
        lines.append(f"\n## 与事件的关联\n{memory}")

    dynamic = person.get("dynamic_state", {}) or {}
    state_parts = [f"{label}: {dynamic.get(key, 0.0)}" for key, label in _STATE_LABELS.items()]
    lines.append("\n## 当前状态\n" + "\n".join(state_parts))

    traits = person.get("private_traits", {}) or {}
    trait_parts = [
        f"{_TRAIT_LABELS.get(key, key)}: {traits.get(key, 0.5)}"
        for key in _TRAIT_LABELS
        if key in traits
    ]
    if trait_parts:
        lines.append("\n## 性格（0-1 强度）\n" + "\n".join(trait_parts))

    lines.append(
        "\n## 行为要求\n"
        "你是一个活跃的社交媒体用户，请通过以下方式参与舆论：\n"
        "- 发帖表达你的立场与观点（create_post）\n"
        "- 评论或回复他人的帖子（create_comment / reply_comment）\n"
        "- 点赞你认同的内容（like_post）\n"
        "- 需要更多信息时搜索或查看帖子详情（search_square / view_thread）\n"
        "不要重复执行刚刚做过的动作；每次行动前参考上一次行动的结果。\n"
        "你的每一次发言、点赞、评论和互动都必须符合以上人设、立场、性格与当前状态。\n"
        "没有想做的事时结束回合。"
    )
    return "\n".join(line for line in lines if line)


def render_observation_message(context: TurnContext) -> str:
    """Render the observation, previous result and remaining budget."""
    previous = None
    if context.previous_result is not None:
        previous = {
            "status": context.previous_result.status.value,
            "data": dict(context.previous_result.data),
            "error_code": context.previous_result.error_code,
        }
    payload = {
        "observation": context.observation,
        "previous_result": previous,
        "remaining_actions": context.remaining_actions,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


class SocialPromptBuilder:
    """Prompt builder bound to a live world, so dynamic state is always fresh."""

    def __init__(self, world: Any) -> None:
        self._world = world

    def __call__(self, context: TurnContext) -> Sequence[ModelMessage]:
        person = self._person(context.entity_id)
        return (
            ModelMessage("system", render_person_system_message(person)),
            ModelMessage("user", render_observation_message(context)),
        )

    def _person(self, entity_id: str) -> Mapping[str, Any]:
        people = self._world.state.get("people", {})
        person = people.get(entity_id)
        if person is None:
            return {"handle": entity_id, "model_policy": {}}
        return person
