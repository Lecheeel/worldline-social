"""Controllers that combine the engine protocol with model providers."""

from __future__ import annotations

import inspect
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from worldline_engine import ActionIntent, FinishTurn
from worldline_engine.protocols import Controller, TurnContext

from .providers.base import CompletionRequest, ModelMessage, ModelProvider

PromptBuilder = Callable[[TurnContext], Sequence[ModelMessage] | Awaitable[Sequence[ModelMessage]]]


class LLMToolController(Controller):
    """Translate normalized provider tool calls into engine intents."""

    def __init__(self, provider: ModelProvider, model: str, prompt_builder: PromptBuilder | None = None, temperature: float | None = 0.0, max_tokens: int | None = 512) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self._provider = provider
        self._model = model
        self._prompt_builder = prompt_builder or self._default_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._request_count = 0
        self._usage: dict[str, int] = defaultdict(int)

    async def next_action(self, context: TurnContext) -> ActionIntent | FinishTurn:
        messages = self._prompt_builder(context)
        if inspect.isawaitable(messages):
            messages = await messages
        response = await self._provider.complete(CompletionRequest(
            model=self._model, messages=tuple(messages),
            tools=tuple(context.available_actions), temperature=self._temperature,
            max_tokens=self._max_tokens,
        ))
        self._request_count += 1
        for metric, value in response.usage.items():
            self._usage[metric] += value
        if not response.tool_calls:
            return FinishTurn("model_finished_without_tool_call")
        call = response.tool_calls[0]
        return ActionIntent(call.name, call.arguments, client_ref=call.call_id)

    def dump_state(self) -> dict[str, Any]:
        return {"request_count": self._request_count, "usage": dict(self._usage)}

    def load_state(self, state: Any) -> None:
        if not isinstance(state, dict) or not isinstance(state.get("request_count", 0), int) or not isinstance(state.get("usage", {}), dict):
            raise ValueError("LLMToolController state is invalid")
        if not all(isinstance(key, str) and isinstance(value, int) for key, value in state.get("usage", {}).items()):
            raise ValueError("LLMToolController usage state is invalid")
        self._request_count = state.get("request_count", 0)
        self._usage = defaultdict(int, state.get("usage", {}))

    @staticmethod
    def _default_prompt(context: TurnContext) -> Sequence[ModelMessage]:
        previous = None if context.previous_result is None else {
            "status": context.previous_result.status.value,
            "data": dict(context.previous_result.data),
            "error_code": context.previous_result.error_code,
        }
        return (ModelMessage("system", "Choose one available world action or finish."), ModelMessage("user", json.dumps({"observation": context.observation, "previous_result": previous, "remaining_actions": context.remaining_actions}, default=str)))
