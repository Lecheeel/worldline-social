"""Extract social-media participants and relationships from event text."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..providers.base import CompletionRequest, ModelMessage, ModelProvider
from .json_utils import extract_json_object
from .prompts import (
    EVENT_EXTRACTION_SYSTEM_PROMPT,
    EVENT_EXTRACTION_USER_TEMPLATE,
)

ENTITY_TYPES = (
    "person",
    "organization",
    "media",
    "government",
    "company",
    "platform",
    "group",
)
STANCES = ("supportive", "opposing", "neutral", "observer")
RELATIONSHIP_TYPES = (
    "follow",
    "employer_of",
    "member_of",
    "official_of",
    "ally",
    "opponent",
    "media_covers",
    "family_of",
    "colleague_of",
    "other",
)


@dataclass(frozen=True)
class Participant:
    """One extracted social actor from the source event."""

    name: str
    entity_type: str
    role: str = ""
    summary: str = ""
    stance: str = "neutral"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "role": self.role,
            "summary": self.summary,
            "stance": self.stance,
        }


@dataclass(frozen=True)
class Relationship:
    """One directed relationship between two participants."""

    source: str
    target: str
    relationship_type: str = "other"
    description: str = ""


@dataclass(frozen=True)
class ExtractionResult:
    participants: tuple[Participant, ...] = ()
    relationships: tuple[Relationship, ...] = ()
    initial_spark: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "participants": [item.to_mapping() for item in self.participants],
            "relationships": [
                {
                    "source": item.source,
                    "target": item.target,
                    "relationship_type": item.relationship_type,
                    "description": item.description,
                }
                for item in self.relationships
            ],
            "initial_spark": self.initial_spark,
        }


class EventExtractor:
    """Call a model provider once to turn event text into structured actors."""

    def __init__(
        self,
        provider: ModelProvider,
        model: str,
        temperature: float | None = 0.2,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_attempts = max_attempts

    async def extract(self, text: str) -> ExtractionResult:
        if not text.strip():
            raise ValueError("event text must not be empty")
        payload = {
            "system_prompt": EVENT_EXTRACTION_SYSTEM_PROMPT,
            "user_prompt": EVENT_EXTRACTION_USER_TEMPLATE.format(text=text),
            "attempts": self._max_attempts,
        }
        for attempt in range(self._max_attempts):
            response = await self._provider.complete(
                CompletionRequest(
                    model=self._model,
                    messages=(
                        ModelMessage("system", EVENT_EXTRACTION_SYSTEM_PROMPT),
                        ModelMessage(
                            "user", EVENT_EXTRACTION_USER_TEMPLATE.format(text=text)
                        ),
                    ),
                    temperature=self._temperature,
                    max_tokens=2048,
                )
            )
            parsed = extract_json_object(response.content)
            if parsed is not None:
                result = self._parse(parsed)
                payload["attempts_used"] = attempt + 1
                payload["raw_parseable"] = True
                return ExtractionResult(
                    participants=result[0],
                    relationships=result[1],
                    initial_spark=str(parsed.get("initial_spark", "")).strip(),
                    diagnostics=payload,
                )
        payload["attempts_used"] = self._max_attempts
        payload["raw_parseable"] = False
        return ExtractionResult(diagnostics=payload)

    @staticmethod
    def _parse(parsed: Mapping[str, Any]) -> tuple[tuple[Participant, ...], tuple[Relationship, ...]]:
        participants: list[Participant] = []
        for raw in parsed.get("participants", ()) or ():
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            participants.append(
                Participant(
                    name=name,
                    entity_type=_normalize_enum(
                        raw.get("entity_type"), ENTITY_TYPES, "person"
                    ),
                    role=str(raw.get("role", "")).strip(),
                    summary=str(raw.get("summary", "")).strip(),
                    stance=_normalize_enum(raw.get("stance"), STANCES, "neutral"),
                )
            )
        names = {item.name for item in participants}
        relationships: list[Relationship] = []
        for raw in parsed.get("relationships", ()) or ():
            if not isinstance(raw, dict):
                continue
            source = str(raw.get("source", "")).strip()
            target = str(raw.get("target", "")).strip()
            if source not in names or target not in names:
                continue
            relationships.append(
                Relationship(
                    source=source,
                    target=target,
                    relationship_type=_normalize_enum(
                        raw.get("relationship_type"), RELATIONSHIP_TYPES, "other"
                    ),
                    description=str(raw.get("description", "")).strip(),
                )
            )
        return tuple(participants), tuple(relationships)


def _normalize_enum(value: Any, allowed: Sequence[str], default: str) -> str:
    text = str(value or "").strip().lower()
    if text in allowed:
        return text
    # Accept common spellings such as "government agency" -> "government".
    for candidate in allowed:
        if candidate in text or text in candidate:
            return candidate
    return default


def is_async_provider(provider: ModelProvider) -> bool:
    """Return True when ``provider.complete`` is a coroutine function."""
    return inspect.iscoroutinefunction(getattr(provider, "complete", None))
