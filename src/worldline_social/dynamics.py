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


def _toward_zero(value: float, step: float) -> float:
    if value > 0:
        return max(0.0, value - step)
    if value < 0:
        return min(0.0, value + step)
    return 0.0
