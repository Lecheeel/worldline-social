"""Provider-neutral contracts for model generation and native tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from worldline_engine.protocols import ActionSpec, JsonValue


@dataclass(frozen=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ModelToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, JsonValue]


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    messages: Sequence[ModelMessage]
    tools: Sequence[ActionSpec] = field(default_factory=tuple)
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class CompletionResponse:
    content: str | None
    tool_calls: Sequence[ModelToolCall] = field(default_factory=tuple)
    provider_request_id: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A provider error stripped of secrets and provider client objects."""


class ModelProvider(Protocol):
    provider_id: str

    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
