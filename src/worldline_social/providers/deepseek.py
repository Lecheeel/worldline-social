"""DeepSeek's OpenAI-compatible chat-completions provider."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import (
    CompletionRequest,
    CompletionResponse,
    ModelProvider,
    ModelToolCall,
    ProviderError,
)


class DeepSeekProvider(ModelProvider):
    """Minimal stdlib client for DeepSeek's OpenAI-compatible endpoint.

    The API key stays in this object and is never included in requests' audit
    payloads, raised error messages, or state serialization.
    """

    provider_id = "deepseek"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("DeepSeek API key must not be empty")
        if not base_url.startswith("https://"):
            raise ValueError("DeepSeek base_url must use HTTPS")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> "DeepSeekProvider":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ProviderError("DEEPSEEK_API_KEY is not set")
        return cls(api_key=api_key)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return await asyncio.to_thread(self._complete_sync, request)

    def _complete_sync(self, request: CompletionRequest) -> CompletionResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
        }
        if request.tools:
            payload["tools"] = [self._tool_schema(tool) for tool in request.tools]
            payload["tool_choice"] = "auto"
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        request_bytes = json.dumps(payload).encode("utf-8")
        http_request = Request(
            f"{self._base_url}/chat/completions",
            data=request_bytes,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(http_request, timeout=self._timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
                request_id = response.headers.get("x-request-id")
        except HTTPError as error:
            raise ProviderError(f"DeepSeek request failed with HTTP {error.code}") from error
        except URLError as error:
            raise ProviderError(f"DeepSeek network request failed: {error.reason}") from error
        except TimeoutError as error:
            raise ProviderError("DeepSeek request timed out") from error

        try:
            choice = response_payload["choices"][0]
            message = choice["message"]
        except (IndexError, KeyError, TypeError) as error:
            raise ProviderError("DeepSeek returned an invalid completion envelope") from error

        tool_calls: list[ModelToolCall] = []
        for index, tool_call in enumerate(message.get("tool_calls") or ()):
            try:
                function = tool_call["function"]
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise TypeError("arguments must decode to an object")
                tool_calls.append(
                    ModelToolCall(
                        call_id=str(tool_call.get("id", f"call-{index}")),
                        name=str(function["name"]),
                        arguments=arguments,
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ProviderError("DeepSeek returned an invalid tool call") from error

        usage = response_payload.get("usage") or {}
        return CompletionResponse(
            content=message.get("content"),
            tool_calls=tuple(tool_calls),
            provider_request_id=request_id,
            usage={
                key: value for key, value in usage.items() if isinstance(value, int)
            },
        )

    @staticmethod
    def _tool_schema(tool: Any) -> dict[str, Any]:
        parameters = dict(tool.parameters_schema)
        # Worlds may use compact constraints such as {"required": ["amount"]}.
        # OpenAI-compatible APIs require a complete JSON Schema object.
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }


def create_deepseek_provider(config: Mapping[str, Any]) -> DeepSeekProvider:
    """Factory usable by the explicit provider registry."""

    api_key = config.get("api_key")
    if not isinstance(api_key, str):
        raise ValueError("DeepSeek provider requires string api_key")
    base_url = config.get("base_url", "https://api.deepseek.com")
    timeout_seconds = config.get("timeout_seconds", 60.0)
    if not isinstance(base_url, str) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("DeepSeek base_url and timeout_seconds are invalid")
    return DeepSeekProvider(api_key, base_url, float(timeout_seconds))
