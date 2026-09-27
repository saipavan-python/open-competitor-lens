"""Tolerant JSON extraction from model output.

Models wrap JSON in prose and markdown fences even when told not to. A parser
that raises on the first stray character would fail the whole run over a
cosmetic problem, so extraction is best-effort and returns ``None`` instead.
"""

from __future__ import annotations

import json
from typing import Any

_FENCE_PREFIXES = ("```json", "```JSON", "```")


def strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if present."""
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].strip() in _FENCE_PREFIXES:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _first_balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` span, respecting string literals."""
    start = text.find("{")
    if start == -1:
        return None
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
                return text[start: index + 1]
    return None


def safe_json_loads(text: str | None) -> Any | None:
    """Parse JSON from model output, or return ``None``.

    Tries, in order: the whole string, the string with fences stripped, and the
    first balanced object found inside surrounding prose.
    """
    if not text or not text.strip():
        return None
    candidates = [text.strip(), strip_code_fences(text)]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    for candidate in candidates:
        if (span := _first_balanced_object(candidate)) is not None:
            try:
                return json.loads(span)
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def dumps(value: Any, *, indent: int | None = 2) -> str:
    """Serialise to JSON, falling back to ``str`` for unserialisable values."""
    try:
        return json.dumps(value, indent=indent, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)
