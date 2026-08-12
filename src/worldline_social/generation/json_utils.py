"""Robust JSON extraction helpers for LLM structured outputs.

LLM responses frequently wrap JSON in fences, add prose, or get truncated.
These helpers tolerate those cases and return ``None`` when no usable object
can be recovered, letting callers fall back to retries or rule-based output.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(content: str | None) -> dict[str, Any] | None:
    """Return the first parseable JSON object inside ``content``.

    Handles markdown fences, surrounding prose, stray control characters and
    truncated objects (unbalanced braces are closed defensively).
    """
    if not content:
        return None
    text = content.strip()
    # Strip markdown code fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Prefer the outermost parseable object so nested values (e.g. a
    # ``traits`` sub-object) are not mistaken for the whole payload.
    for candidate in _candidate_objects(text):
        parsed = _try_load(candidate)
        if parsed is not None:
            return parsed

    # Defensive last attempt: close unbalanced braces and retry.
    repaired = _repair_truncated(text)
    if repaired is not text:
        for candidate in _candidate_objects(repaired):
            parsed = _try_load(candidate)
            if parsed is not None:
                return parsed
    return None


def _candidate_objects(text: str) -> list[str]:
    """Yield substrings that look like JSON objects, outermost first."""
    starts = [match.start() for match in re.finditer(r"\{", text)]
    candidates: list[str] = []
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    return candidates


def _try_load(candidate: str) -> dict[str, Any] | None:
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _repair_truncated(text: str) -> str:
    """Close unbalanced brackets and stray quotes defensively."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    result = text.rstrip()
    opens = text.count("{") + text.count("[")
    closes = text.count("}") + text.count("]")
    if closes > opens:
        return text
    # Only close a string when an unescaped quote is still open (odd count).
    if result and result[-1] not in '",}]' and result.count('"') % 2 == 1:
        result += '"'
    result += "]" * (text.count("[") - text.count("]"))
    result += "}" * (text.count("{") - text.count("}"))
    return result
