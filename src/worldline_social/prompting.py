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


def render_person_system_message(
    person: Mapping[str, Any], include_dynamic: bool = True
) -> str:
    """Render a person's identity, persona, traits and (optionally) live
    state as a system message.

    ``include_dynamic=False`` yields a fully static message: identical
    across ticks, so DeepSeek's automatic prefix caching can hit on it
    (dynamic state then lives in the user message instead).
    """
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

    if include_dynamic:
        dynamic = person.get("dynamic_state", {}) or {}
        state_parts = [
            f"{label}: {dynamic.get(key, 0.0)}"
            for key, label in _STATE_LABELS.items()
        ]
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
        "- 优先互动：点赞认同的帖子（like_post）、评论他人的帖子（create_comment）、"
        "回复评论（reply_comment）\n"
        "- 保持在场感：当你对话题有新的立场、新的信息或回应质疑时发帖（create_post），"
        "保持你的存在感，但不要重复表达已经说过的观点\n"
        "- 观察中已包含广场动态全文（observation.feed），每条帖子附带最新评论（含 comment_id）"
        "与评论总数（comment_count），通常无需再执行 view_feed\n"
        "- 回复评论时，comment_id 必须直接引用 feed 或 view_thread 返回的评论，"
        "绝不能用 post_id 代替\n"
        "- 评论数多的帖子（comment_count 较高）讨论激烈，可先 view_thread 查看全部评论再参与；"
        "已经查看过的帖子不要重复查看\n"
        "- 需要查找历史内容或特定话题时用 search_square\n"
        "- 不要重复执行刚刚做过的动作；每次行动前参考上一次行动的结果。\n"
        "你的每一次发言、点赞、评论和互动都必须符合以上人设、立场、性格与当前状态。\n"
        "没有想做的事时结束回合。"
    )
    return "\n".join(line for line in lines if line)


def _summarize_data(value: Any, depth: int = 0) -> Any:
    """Shrink previous-result payloads so a full feed is not echoed twice.

    The observation already carries the feed; echoing the same feed again
    through ``previous_result.data`` doubles prompt size every action.
    Strings are capped at 160 chars, lists at 3 items (with a count note),
    and nesting is limited to 3 levels.
    """
    if depth > 3:
        return "…"
    if isinstance(value, str):
        return value if len(value) <= 160 else value[:160] + "…"
    if isinstance(value, list):
        if len(value) <= 3:
            return [_summarize_data(item, depth + 1) for item in value]
        return [_summarize_data(item, depth + 1) for item in value[:3]] + [
            f"…（另有 {len(value) - 3} 项）"
        ]
    if isinstance(value, dict):
        return {key: _summarize_data(item, depth + 1) for key, item in value.items()}
    return value


def render_observation_message(context: TurnContext) -> str:
    """Render the observation, previous result and remaining budget."""
    previous = None
    if context.previous_result is not None:
        previous = {
            "status": context.previous_result.status.value,
            "data": _summarize_data(dict(context.previous_result.data)),
            "error_code": context.previous_result.error_code,
        }
    payload = {
        "observation": context.observation,
        "previous_result": previous,
        "remaining_actions": context.remaining_actions,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def render_dynamic_state_section(person: Mapping[str, Any]) -> str:
    """Render only the live emotional state (goes at the top of the user
    message so the system message can stay cache-stable)."""
    dynamic = person.get("dynamic_state", {}) or {}
    state_parts = [
        f"{label}: {dynamic.get(key, 0.0)}"
        for key, label in _STATE_LABELS.items()
    ]
    return "## 当前状态\n" + "\n".join(state_parts)


class SocialPromptBuilder:
    """Prompt builder bound to a live world, so dynamic state is always fresh."""

    def __init__(self, world: Any) -> None:
        self._world = world

    def __call__(self, context: TurnContext) -> Sequence[ModelMessage]:
        person = self._person(context.entity_id)
        return (
            ModelMessage(
                "system", render_person_system_message(person, include_dynamic=False)
            ),
            ModelMessage(
                "user",
                render_dynamic_state_section(person)
                + "\n\n"
                + render_observation_message(context),
            ),
        )

    def _person(self, entity_id: str) -> Mapping[str, Any]:
        people = self._world.state.get("people", {})
        person = people.get(entity_id)
        if person is None:
            return {"handle": entity_id, "model_policy": {}}
        return person
