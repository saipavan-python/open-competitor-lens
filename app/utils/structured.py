"""Reading a structured answer out of a model completion.

Lives in ``utils`` rather than next to one adapter because three callers need it:
the OpenAI adapter, the Gemini adapter, and :class:`~app.services.extraction_service.
ExtractionService`, which calls an injected LLM and must therefore be able to read
that LLM's answer without importing a provider's private internals. A vendor
adapter owns *how* a request is made; this owns *what a compliant answer looks
like*, which is not vendor-specific.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from app.core.exceptions import ProviderError

#: A fenced code block, which models emit even when told not to.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_structured(content: str, schema: type[Any] | dict[str, Any] | None) -> Any:
    """Coerce a completion into ``schema``, or raise.

    Three shapes are tried in order, because all three occur in practice: the
    response parses as JSON directly; it is wrapped in a markdown fence; or it has
    prose around a JSON object. Stopping at the first two would make the third a
    hard failure for a difference in punctuation.

    When ``schema`` is ``None`` the content is returned untouched. The model
    router asks for text for some prompts, and a prompt that wanted prose must
    not be failed for not producing JSON.
    """
    if schema is None:
        return content

    payload = _as_json(content)
    if payload is None:
        # The offending content is *not* put in the message. ProviderError is
        # surfaced to API clients through ``to_dict``, and a model answer can
        # contain anything a page contained. The schema name is enough to debug
        # with, and the raw text is already in the run's own logs.
        raise ProviderError(
            f"the model did not return JSON for a structured request "
            f"(wanted {_schema_name(schema)})",
            retryable=True,
        )

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            fields = ", ".join(
                ".".join(str(part) for part in err["loc"]) or "<root>"
                for err in exc.errors()[:5]
            )
            raise ProviderError(
                f"the model's JSON did not match {_schema_name(schema)}: {fields}",
                retryable=True,
            ) from exc
    return payload


def _as_json(content: str) -> Any:
    """The JSON in a completion, from any of the shapes models actually emit."""
    text = (content or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = _FENCE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = min(
        (index for index in (text.find("{"), text.find("[")) if index != -1),
        default=-1,
    )
    if start == -1:
        return None
    tail = text[start:]
    # Trim to the matching brace so trailing prose is dropped rather than
    # becoming a parse error.
    for end in range(len(tail), 0, -1):
        if tail[end - 1] not in "}]":
            continue
        try:
            return json.loads(tail[:end])
        except json.JSONDecodeError:
            continue
    return None


def _schema_name(schema: type[Any] | dict[str, Any] | None) -> str:
    """A readable name for the schema, for the error message."""
    if isinstance(schema, type):
        return schema.__name__
    if isinstance(schema, Mapping):
        return str(schema.get("title") or schema.get("name") or "inline schema")
    return str(schema)
