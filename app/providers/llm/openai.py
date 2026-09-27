"""The OpenAI (and OpenAI-compatible) chat-completions backend.

``OPENAI_BASE_URL`` is honoured, so this also serves any endpoint that speaks the
chat-completions protocol — a local model server, a gateway, a proxy. The
structured-output path uses ``response_format`` with a JSON schema where the
endpoint supports it, and falls back to instructing the model in the prompt where
it does not, then validating locally either way. The fallback matters because a
gateway that silently ignores ``response_format`` would otherwise produce prose
where JSON was requested, and the failure would surface as a confusing
validation error in a schema far from the call.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.providers.llm._shared import parse_structured, response
from app.schemas.provider import LLMResponse, ModelRole

log = get_logger(__name__)


class OpenAILLMProvider:
    """Chat completions through the OpenAI SDK."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        default_model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout: int = 120,
        client: Any = None,
    ) -> None:
        if not api_key.strip():
            raise ProviderError(
                "the OpenAI provider needs an API key; set OPENAI_API_KEY",
            )
        self._default_model = default_model
        self._timeout = timeout
        self._client = client or self._build_client(api_key, base_url, timeout)

    @staticmethod
    def _build_client(api_key: str, base_url: str | None, timeout: int) -> Any:
        """The SDK client, imported here so the dependency is paid only on use."""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderError(
                "the openai package is required for this provider; pip install openai",
            ) from exc
        return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def model_for(self, role: ModelRole) -> str:
        """The model id for a role, falling back to the default."""
        return getattr(self, f"_{role}_model", None) or self._default_model

    def set_profiles(self, profiles: dict[str, str]) -> None:
        """Record the configured model per role.

        Set by the registry from ``Settings.model_profiles`` so that role-to-model
        mapping stays in configuration, and a prompt never hardcodes a model id.
        """
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
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": self._prompt_with_schema(prompt, schema)})

        started = time.monotonic()
        try:
            completion = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0 if temperature is None else temperature,
                **self._format_kwargs(schema),
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as ProviderError below
            raise ProviderError(
                f"the OpenAI request failed: {exc}", retryable=True
            ) from exc

        choice = completion.choices[0] if completion.choices else None
        content = (choice.message.content if choice and choice.message else "") or ""
        result = response(
            content,
            model=model,
            provider=self.name,
            prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
            started=started,
            finish_reason=getattr(choice, "finish_reason", None) if choice else None,
        )
        # Validate here rather than trusting the endpoint honoured response_format.
        parse_structured(content, schema)
        log.debug(
            "openai completion",
            extra={"model": model, "duration_ms": result.duration_ms},
        )
        return result

    @staticmethod
    def _prompt_with_schema(prompt: str, schema: type[Any] | dict[str, Any] | None) -> str:
        """The prompt, with the JSON requirement stated in words.

        Stated even when ``response_format`` is also sent. A gateway that drops
        the parameter then still gets the instruction, and one that honours it
        gets a redundant line rather than a contradiction.
        """
        if schema is None:
            return prompt
        name = getattr(schema, "__name__", "the requested schema")
        return (
            f"{prompt}\n\n"
            f"Reply with a single JSON object matching {name} and nothing else. "
            f"No prose before or after the JSON."
        )

    @staticmethod
    def _format_kwargs(schema: type[Any] | dict[str, Any] | None) -> dict[str, Any]:
        """Structured-output arguments, or none.

        ``response_format`` is only sent when a schema was asked for. A gateway
        that does not understand the parameter is handled by the prompt
        instruction in :meth:`_prompt_with_schema` and by the local validation
        in :meth:`generate`, not by trying the call first — a refusal here would
        be worse than the prose the parameter exists to prevent.
        """
        if schema is None:
            return {}
        return {"response_format": {"type": "json_object"}}
