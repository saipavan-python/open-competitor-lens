"""Shared plumbing for the LLM adapters.

The parts that are not vendor-specific live here so that "a response that does not
match the requested schema is an error" is written down once. Both vendors honour
it, and a third would too, which is the point: that rule is the contract, not a
detail of OpenAI's response format.

The schema-coercion rule itself lives in :mod:`app.utils.structured`, because
``ExtractionService`` calls an injected LLM and must read the same answer without
reaching into a provider's private module.
"""

from __future__ import annotations

import time

from app.schemas.provider import LLMResponse
from app.utils.structured import parse_structured

__all__ = ["parse_structured", "response"]


def response(
    content: str,
    *,
    model: str,
    provider: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    started: float,
    finish_reason: str | None = None,
) -> LLMResponse:
    """Build a response, timing it from a ``time.monotonic()`` reading.

    Duration is computed here rather than in each adapter so that every provider
    reports it the same way, and so a caller summing durations across providers
    is summing the same quantity.
    """
    return LLMResponse(
        content=content,
        model=model,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_ms=int((time.monotonic() - started) * 1000),
        finish_reason=finish_reason,
    )
