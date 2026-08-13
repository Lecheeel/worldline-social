"""Web search for the research stage via DeepSeek's Responses API.

This is the same capability exposed by ``tools/deepseek-search-mcp/server.py``
(server-side ``web_search`` tool, no extra search API key), implemented here
as an importable function so the task runner can call it in-process.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEFAULT_MAX_OUTPUT_TOKENS = 1500
DEFAULT_TIMEOUT_SECONDS = 120


class SearchError(RuntimeError):
    """A web-search failure (network, API, or empty result)."""


@dataclass(frozen=True)
class SearchResult:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)


def _extract_output_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in data.get("output", ()) or ():
        if item.get("type") != "message":
            continue
        for content in item.get("content", ()) or ():
            if content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    return "\n".join(part for part in parts if part)


def web_search_sync(
    query: str,
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> SearchResult:
    """Run one server-side web search; raises :class:`SearchError` on failure."""
    if not query.strip():
        raise SearchError("query must not be empty")
    key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise SearchError("DEEPSEEK_API_KEY is not set")
    payload = {
        "model": model,
        "tools": [{"type": "web_search"}],
        "input": query,
        "max_output_tokens": max(1, min(int(max_output_tokens), 8192)),
    }
    request = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:500]
        raise SearchError(f"DeepSeek API HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise SearchError(f"DeepSeek API network error: {error.reason}") from error
    except TimeoutError as error:
        raise SearchError("DeepSeek API request timed out") from error

    text = _extract_output_text(data)
    if not text and data.get("error"):
        raise SearchError(f"DeepSeek API error: {data['error']}")
    if not text:
        raise SearchError("web search returned no content")
    return SearchResult(text=text, usage=dict(data.get("usage") or {}))


async def web_search(
    query: str,
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> SearchResult:
    """Async wrapper around :func:`web_search_sync` (runs in a thread)."""
    return await asyncio.to_thread(
        web_search_sync,
        query,
        max_output_tokens=max_output_tokens,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
