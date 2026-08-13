"""Turn extracted participants into PersonProfile values."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Mapping

from ..population import PersonProfile
from ..providers.base import CompletionRequest, ModelMessage, ModelProvider
from .extract import Participant
from .json_utils import extract_json_object
from .prompts import PROFILE_SYSTEM_PROMPT, PROFILE_USER_TEMPLATE

TRAIT_KEYS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "honesty_humility",
    "machiavellianism",
    "narcissism",
    "psychopathy",
)

GROUP_TYPES = ("organization", "media", "government", "company", "platform", "group")

_SLUG_RE = re.compile(r"[^a-z0-9]+")

_STANCE_INITIAL_STATE: dict[str, Mapping[str, float]] = {
    "supportive": {"mood": 0.2, "anger": 0.0, "stress": 0.1, "fatigue": 0.0, "threat": 0.0},
    "opposing": {"mood": -0.2, "anger": 0.3, "stress": 0.2, "fatigue": 0.0, "threat": 0.0},
    "neutral": {"mood": 0.0, "anger": 0.0, "stress": 0.1, "fatigue": 0.0, "threat": 0.0},
    "observer": {"mood": 0.0, "anger": 0.0, "stress": 0.1, "fatigue": 0.0, "threat": 0.0},
}


class ProfileGenerator:
    """Generate a detailed PersonProfile for one extracted participant."""

    def __init__(
        self,
        provider: ModelProvider,
        model: str,
        temperature: float | None = 0.4,
        max_attempts: int = 3,
        thinking: str | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_attempts = max_attempts
        self._thinking = thinking

    async def generate(
        self,
        participant: Participant,
        external_id: str,
        event_excerpt: str = "",
    ) -> PersonProfile:
        """Return a PersonProfile, falling back to rule-based output on failure."""
        for attempt in range(self._max_attempts):
            response = await self._provider.complete(
                CompletionRequest(
                    model=self._model,
                    messages=(
                        ModelMessage("system", PROFILE_SYSTEM_PROMPT),
                        ModelMessage(
                            "user",
                            PROFILE_USER_TEMPLATE.format(
                                name=participant.name,
                                entity_type=participant.entity_type,
                                role=participant.role,
                                summary=participant.summary,
                                stance=participant.stance,
                                event_excerpt=_excerpt(event_excerpt),
                            ),
                        ),
                    ),
                    temperature=self._temperature,
                    max_tokens=2048,
                    thinking=self._thinking,
                )
            )
            parsed = extract_json_object(response.content)
            if parsed is not None:
                return self._build_profile(participant, external_id, parsed)
        return self._rule_based_profile(participant, external_id)

    def _build_profile(
        self, participant: Participant, external_id: str, parsed: Mapping[str, Any]
    ) -> PersonProfile:
        display_name = str(parsed.get("display_name", "")).strip() or participant.name
        bio = str(parsed.get("bio", "")).strip() or _default_bio(participant)
        persona = str(parsed.get("persona", "")).strip()
        raw_traits = parsed.get("traits")
        traits = {
            key: _coerce_trait(raw_traits.get(key) if isinstance(raw_traits, dict) else None)
            for key in TRAIT_KEYS
        }
        stance = str(parsed.get("stance", participant.stance)).strip().lower()
        if stance not in _STANCE_INITIAL_STATE:
            stance = participant.stance
        model_policy = {
            "persona": persona,
            "stance": stance,
            "interested_topics": _coerce_str_list(parsed.get("interested_topics")),
            "personal_memory": str(parsed.get("personal_memory", "")).strip(),
            "age": parsed.get("age"),
            "gender": str(parsed.get("gender", "other")).strip(),
            "country": str(parsed.get("country", "")).strip(),
            "profession": str(parsed.get("profession", "")).strip(),
            "entity_type": participant.entity_type,
            "source_role": participant.role,
        }
        return PersonProfile(
            external_id=external_id,
            handle=_handle_for(participant.name, external_id),
            display_name=display_name,
            bio=bio,
            private_traits=traits,
            initial_state=dict(_STANCE_INITIAL_STATE[stance]),
            model_policy=model_policy,
        )

    def _rule_based_profile(
        self, participant: Participant, external_id: str
    ) -> PersonProfile:
        is_group = participant.entity_type in GROUP_TYPES
        if is_group:
            traits = _group_traits(participant.entity_type)
            bio = f"官方账号：{participant.name}。{participant.role}"
            persona = (
                f"{participant.name} 是{participant.role}相关的官方账号，"
                f"在事件中持{_stance_zh(participant.stance)}立场。{participant.summary}"
            )
        else:
            traits = {key: 0.5 for key in TRAIT_KEYS}
            bio = f"{participant.role}。{participant.summary}"
            persona = (
                f"{participant.name} 在事件中扮演{participant.role}，"
                f"持{_stance_zh(participant.stance)}立场。{participant.summary}"
            )
        return PersonProfile(
            external_id=external_id,
            handle=_handle_for(participant.name, external_id),
            display_name=participant.name,
            bio=bio,
            private_traits=traits,
            initial_state=dict(_STANCE_INITIAL_STATE[participant.stance]),
            model_policy={
                "persona": persona,
                "stance": participant.stance,
                "entity_type": participant.entity_type,
                "source_role": participant.role,
            },
        )


def _coerce_trait(value: Any) -> float:
    """Normalize a trait value from LLM output to [0.0, 1.0]."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        text = value.strip()
        try:
            return max(0.0, min(1.0, float(text)))
        except ValueError:
            pass
        lowered = text.lower()
        if "高" in lowered or "high" in lowered or "strong" in lowered:
            return 0.8
        if "中" in lowered or "medium" in lowered or "moderate" in lowered:
            return 0.5
        if "低" in lowered or "low" in lowered or "weak" in lowered:
            return 0.2
    return 0.5


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _group_traits(entity_type: str) -> dict[str, float]:
    if entity_type == "media":
        return {**{key: 0.5 for key in TRAIT_KEYS}, "openness": 0.7, "conscientiousness": 0.7}
    if entity_type == "government":
        return {**{key: 0.5 for key in TRAIT_KEYS}, "conscientiousness": 0.8, "agreeableness": 0.5}
    if entity_type == "company":
        return {**{key: 0.5 for key in TRAIT_KEYS}, "conscientiousness": 0.7, "machiavellianism": 0.4}
    return {key: 0.5 for key in TRAIT_KEYS}


def _handle_for(name: str, external_id: str) -> str:
    slug = _SLUG_RE.sub("_", name.lower()).strip("_")[:24]
    digest = hashlib.sha256(external_id.encode("utf-8")).hexdigest()[:6]
    base = slug or "user"
    return f"{base}_{digest}"


def _default_bio(participant: Participant) -> str:
    return f"{participant.role or participant.entity_type}。{participant.summary}"


def _excerpt(text: str, limit: int = 800) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "……"


def _stance_zh(stance: str) -> str:
    return {
        "supportive": "支持",
        "opposing": "反对",
        "neutral": "中立",
        "observer": "观察",
    }.get(stance, "中立")
