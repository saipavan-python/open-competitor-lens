"""The Google Gemini backend.

Gemini differs from OpenAI in two ways that matter here. The system instruction is
a separate parameter rather than a message, and structured output is requested
with a Pydantic-aware ``response_schema`` rather than a JSON mode. The response
is still validated locally by :mod:`app.providers.llm._shared`, because a model
can return a well-formed JSON object that does not satisfy the schema, and
"well-formed but wrong shape" is the failure the evidence gate downstream cannot
see for itself.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.providers.llm._shared import parse_structured, response
from app.schemas.provider import LLMResponse, ModelRole

log = get_logger(__name__)


class GeminiLLMProvider:
    """Text generation through ``google-genai``."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        *,
        default_model: str = "gemini-2.0-flash",
        timeout: int = 120,
        client: Any = None,
    ) -> None:
        if not api_key.strip():
            raise ProviderError(
                "the Gemini provider needs an API key; set GEMINI_API_KEY",
            )
        self._default_model = default_model
        self._timeout = timeout
        self._client = client or self._build_client(api_key, timeout)

    @staticmethod
    def _build_client(api_key: str, timeout: int) -> Any:
        """The SDK client, imported here so the dependency is paid only on use."""
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderError(
                "the google-genai package is required; pip install google-genai",
            ) from exc
        return genai.Client(api_key=api_key)

    def model_for(self, role: ModelRole) -> str:
        """The model id for a role, falling back to the default."""
        return getattr(self, f"_{role}_model", None) or self._default_model

    def set_profiles(self, profiles: dict[str, str]) -> None:
        """Record the configured model per role."""
        for role, model in profiles.items():
            setattr(self, f"_{role}_model", model)

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: type[Any] | dict[str, Any] | None = None,
        model_role: ModelRole = "default",
        temperature: float | None = None,
    ) -> LLMResponse:
        """One completion, validated against ``schema`` before it is returned."""
        model = self.model_for(model_role)
        config: dict[str, Any] = {
            "temperature": 0.0 if temperature is None else temperature,
        }
        if system:
            config["system_instruction"] = system
        # A Pydantic model is handed to the API as a response schema. Gemini takes
        # the schema *inside* the config, not as a separate request argument; the
        # response MIME type has to be set alongside it or the schema is ignored.
        if isinstance(schema, type):
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema
        # An inline dict schema cannot be sent, so the requirement becomes words.
        # Appended to the prompt rather than left as an unused helper.
        instruction = self.schema_instruction(schema)
        contents = f"{prompt}\n\n{instruction}" if instruction else prompt

        started = time.monotonic()
        try:
            result = self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            if hasattr(result, "__await__"):
                result = await result
        except Exception as exc:  # noqa: BLE001 - re-raised as ProviderError below
            raise ProviderError(
                f"the Gemini request failed: {exc}", retryable=True
            ) from exc

        content = (getattr(result, "text", None) or "").strip()
        usage = getattr(result, "usage_metadata", None)
        out = response(
            content,
            model=model,
            provider=self.name,
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            started=started,
        )
        parse_structured(content, schema)
        log.debug(
            "gemini completion", extra={"model": model, "duration_ms": out.duration_ms}
        )
        return out

    @staticmethod
    def schema_instruction(schema: type[Any] | dict[str, Any] | None) -> str:
        """The JSON requirement as text, for a schema Gemini cannot be given.

        Only an inline dict schema needs this. A Pydantic model is sent to the
        API as a real response schema, so restating it in the prompt would be
        redundant — and for a large schema, a second copy to get out of sync.
        """
        if schema is None or isinstance(schema, type):
            return ""
        return (
            "Reply with a single JSON object matching this schema and nothing "
            f"else: {json.dumps(schema, default=str)}"
        )
