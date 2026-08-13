"""Bounded, replaceable social-dynamics policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class TraitProfile:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    honesty_humility: float = 0.5
    machiavellianism: float = 0.0
    narcissism: float = 0.0
    psychopathy: float = 0.0

    def __post_init__(self) -> None:
        values = self.__dict__
        if any(not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("trait values must be in [0, 1]")

    def to_mapping(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DynamicState:
    mood: float = 0.0
    anger: float = 0.0
    stress: float = 0.0
    fatigue: float = 0.0
    threat: float = 0.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.mood <= 1.0:
            raise ValueError("mood must be in [-1, 1]")
        for name in ("anger", "stress", "fatigue", "threat"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_mapping(self) -> dict[str, float]:
        return dict(self.__dict__)


class DynamicsPolicy(Protocol):
    def advance(
        self,
        person_id: str,
        traits: Mapping[str, Any],
        dynamic_state: Mapping[str, Any],
        tick_id: int,
    ) -> Mapping[str, Any]: ...

    def apply_feedback(
        self,
        person_id: str,
        traits: Mapping[str, Any],
        dynamic_state: Mapping[str, Any],
        feedback: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...


class RecoveryDynamics:
    """Conservative baseline: internal states recover toward neutral values."""

    def advance(
        self,
        person_id: str,
        traits: Mapping[str, Any],
        dynamic_state: Mapping[str, Any],
        tick_id: int,
    ) -> Mapping[str, Any]:
        del person_id, traits, tick_id
        state = DynamicState(
            mood=float(dynamic_state.get("mood", 0.0)),
            anger=float(dynamic_state.get("anger", 0.0)),
            stress=float(dynamic_state.get("stress", 0.0)),
            fatigue=float(dynamic_state.get("fatigue", 0.0)),
            threat=float(dynamic_state.get("threat", 0.0)),
        )
        return DynamicState(
            mood=_toward_zero(state.mood, 0.08),
            anger=_toward_zero(state.anger, 0.05),
            stress=_toward_zero(state.stress, 0.04),
            fatigue=_toward_zero(state.fatigue, 0.02),
            threat=_toward_zero(state.threat, 0.05),
        ).to_mapping()

    def apply_feedback(
        self,
        person_id: str,
        traits: Mapping[str, Any],
        dynamic_state: Mapping[str, Any],
        feedback: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Baseline ignores feedback; moods only recover over time."""
        del person_id, traits, feedback
        return dict(dynamic_state)


class AffectiveDynamics:
    """Feedback-driven dynamics modulated by personality traits.

    Social actions (posting, receiving likes or dislikes, receiving comments)
    shift mood, anger, stress and threat. Trait values shape the response:
    neurotic people react harder to negative feedback, agreeable people are
    more affected by approval, psychopathic profiles are insulated from it.
    """

    def advance(
        self,
        person_id: str,
        traits: Mapping[str, Any],
        dynamic_state: Mapping[str, Any],
        tick_id: int,
    ) -> Mapping[str, Any]:
        del person_id, traits, tick_id
        state = DynamicState(
            mood=float(dynamic_state.get("mood", 0.0)),
            anger=float(dynamic_state.get("anger", 0.0)),
            stress=float(dynamic_state.get("stress", 0.0)),
            fatigue=float(dynamic_state.get("fatigue", 0.0)),
            threat=float(dynamic_state.get("threat", 0.0)),
        )
        return DynamicState(
            mood=_toward_zero(state.mood, 0.04),
            anger=_toward_zero(state.anger, 0.03),
            stress=_toward_zero(state.stress, 0.02),
            fatigue=_toward_zero(state.fatigue, 0.02),
            threat=_toward_zero(state.threat, 0.02),
        ).to_mapping()

    def apply_feedback(
        self,
        person_id: str,
        traits: Mapping[str, Any],
        dynamic_state: Mapping[str, Any],
        feedback: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        del person_id
        trait = TraitProfile(**{
            key: float(traits.get(key, default))
            for key, default in {
                "openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
                "agreeableness": 0.5, "neuroticism": 0.5, "honesty_humility": 0.5,
                "machiavellianism": 0.0, "narcissism": 0.0, "psychopathy": 0.0,
            }.items()
        })
        state = DynamicState(
            mood=float(dynamic_state.get("mood", 0.0)),
            anger=float(dynamic_state.get("anger", 0.0)),
            stress=float(dynamic_state.get("stress", 0.0)),
            fatigue=float(dynamic_state.get("fatigue", 0.0)),
            threat=float(dynamic_state.get("threat", 0.0)),
        )
        mood = state.mood
        anger = state.anger
        stress = state.stress
        threat = state.threat
        negative_insulation = 1.0 - trait.psychopathy * 0.6
        for item in feedback:
            kind = str(item.get("kind", "")).strip()
            count = max(1, int(item.get("count", 1) or 1))
            if kind == "post_created":
                mood += 0.05 * count * (0.5 + trait.extraversion)
            elif kind == "received_like":
                mood += 0.03 * count * (0.5 + trait.agreeableness * 0.5)
            elif kind == "received_unlike":
                neuro_weight = 0.5 + trait.neuroticism
                mood -= 0.08 * count * neuro_weight * negative_insulation
                anger += 0.06 * count * neuro_weight * negative_insulation
            elif kind == "received_comment":
                stress += 0.04 * count * (0.5 + trait.neuroticism * 0.5)
                mood += 0.02 * count * trait.extraversion
            elif kind == "read_negative":
                neuro_weight = 0.5 + trait.neuroticism
                stress += 0.07 * neuro_weight * negative_insulation
                threat += 0.05 * neuro_weight * negative_insulation
                mood -= 0.04 * negative_insulation
                anger += 0.05 * neuro_weight * negative_insulation
            elif kind == "read_positive":
                mood += 0.03
        return DynamicState(
            mood=_clamp(mood, -1.0, 1.0),
            anger=_clamp(anger, 0.0, 1.0),
            stress=_clamp(stress, 0.0, 1.0),
            fatigue=state.fatigue,
            threat=_clamp(threat, 0.0, 1.0),
        ).to_mapping()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _toward_zero(value: float, step: float) -> float:
    if value > 0:
        return max(0.0, value - step)
    if value < 0:
        return min(0.0, value + step)
    return 0.0
