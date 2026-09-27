"""Provider resolution: name plus configuration to a live adapter.

The rest of the system asks for ``resolve("search", "tavily")`` and receives an
object satisfying :class:`~app.providers.base.SearchProvider`. It never imports a
vendor SDK and never inspects an environment variable, which is what keeps the
orchestrator testable with offline substitutes.

Two behaviours here are load-bearing:

- A missing credential raises :class:`ProviderNotConfiguredError` *naming the
  environment variable*. "Search is unavailable" costs an hour of debugging;
  "set TAVILY_API_KEY" does not.
- ``fallbacks`` lets a degraded run proceed. Research that silently returns
  nothing is worse than research that says it ran degraded.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import ProviderNotConfiguredError
from app.core.logging import get_logger
from app.providers.base import ProviderHealth

log = get_logger(__name__)

ProviderKind = str  # "llm" | "search" | "web" | "browser" | "embedding" | "vector" | "graph"


@dataclass(frozen=True)
class ProviderSpec:
    """How to build one provider, and what must be configured first."""

    kind: ProviderKind
    name: str
    factory: Callable[[Settings], Any]
    required_env: tuple[str, ...] = ()
    #: Read from ``Settings`` instead of the process environment, so tests can
    #: configure a provider without mutating the real environment.
    settings_fields: tuple[str, ...] = ()
    health: ProviderHealth = field(default=None)  # type: ignore[assignment]

    def missing_settings(self, settings: Settings) -> list[str]:
        """Environment variables that are absent or blank.

        A blank value counts as missing: ``OPENAI_API_KEY=`` is a misconfigured
        deploy, and treating it as present produces an authentication error
        twenty seconds later with a much less obvious message.
        """
        missing: list[str] = []
        for name in self.required_env:
            if not (os.environ.get(name) or "").strip():
                missing.append(name)
        for field_name in self.settings_fields:
            if not (getattr(settings, field_name, None) or ""):
                missing.append(field_name.upper())
        return missing


def _unavailable(kind: ProviderKind, name: str) -> Callable[[Settings], Any]:
    """A factory for an adapter that does not exist in this deployment.

    Provided so a misconfigured browser or graph store fails at the point of use
    with an explanation, rather than at import time taking down the whole run.
    """

    def build(_settings: Settings) -> Any:
        raise ProviderNotConfiguredError(
            f"{name}", f"{kind.upper()}_PROVIDER_{name.upper()}"
        )

    return build


class ProviderRegistry:
    """Resolves provider names to instances, and tracks their health."""

    def __init__(self, settings: Settings | None = None, *, specs: list[ProviderSpec] | None = None
                 ) -> None:
        self._settings = settings or get_settings()
        self._specs: dict[tuple[str, str], ProviderSpec] = {}
        self._health: dict[tuple[str, str], ProviderHealth] = {}
        for spec in specs if specs is not None else default_specs():
            self.register(spec)

    def register(self, spec: ProviderSpec) -> None:
        self._specs[(spec.kind, spec.name)] = spec
        self._health[(spec.kind, spec.name)] = spec.health or ProviderHealth(spec.name)

    def health(self, kind: str, name: str) -> ProviderHealth:
        key = (kind, name)
        if key not in self._health:
            self._health[key] = ProviderHealth(name)
        return self._health[key]

    def health_snapshot(self) -> list[dict[str, Any]]:
        return [health.snapshot() for health in self._health.values()]

    def available(self, kind: str, names: list[str] | None = None) -> list[str]:
        """Names of every provider of ``kind`` that is configured right now."""
        candidates = names or [n for (k, n) in self._specs if k == kind]
        return [
            name
            for name in candidates
            if (spec := self._specs.get((kind, name))) is not None
            and not spec.missing_settings(self._settings)
        ]

    def resolve(
        self, kind: str, name: str, *, fallbacks: list[str] | None = None
    ) -> Any:
        """Build a provider, falling back when the requested one is unusable.

        Raises :class:`ProviderNotConfiguredError` if the provider is unknown or
        every candidate in the chain is unconfigured. The error names the
        credential of the *originally requested* provider, because that is the
        one the operator asked for and the one they need to set.
        """
        primary = self._specs.get((kind, name))
        if primary is None:
            raise ProviderNotConfiguredError(
                f"{name}",
                f"{kind.upper()}_PROVIDER",
                message=(
                    f"Unknown {kind} provider '{name}'. "
                    f"Registered: {sorted(n for (k, n) in self._specs if k == kind)}"
                ),
            )

        missing = primary.missing_settings(self._settings)
        if not missing:
            return self._build(primary)

        tried = [name]
        for fallback in fallbacks or []:
            spec = self._specs.get((kind, fallback))
            tried.append(fallback)
            if spec is None:
                log.warning(
                    "fallback provider is not registered",
                    extra={"kind": kind, "fallback": fallback},
                )
                continue
            if spec.missing_settings(self._settings):
                log.warning(
                    "fallback provider is unconfigured",
                    extra={"kind": kind, "fallback": fallback,
                           "missing": spec.missing_settings(self._settings)},
                )
                continue
            log.warning(
                "provider unconfigured; using fallback",
                extra={"kind": kind, "requested": name, "fallback": fallback,
                       "missing": missing},
            )
            return self._build(spec)

        raise ProviderNotConfiguredError(
            f"{name}",
            missing[0],
            message=(
                f"{kind} provider '{name}' is not configured: "
                f"{', '.join(missing)}. Tried: {', '.join(tried)}."
            ),
        )

    def _build(self, spec: ProviderSpec) -> Any:
        instance = spec.factory(self._settings)
        if getattr(instance, "name", None) is None:
            try:
                instance.name = spec.name  # type: ignore[attr-defined]
            except AttributeError:
                log.debug("provider instance is immutable and has no name",
                          extra={"provider": spec.name})
        return instance


def _offline(factory: Callable[[], Any]) -> Callable[[Settings], Any]:
    """Adapt a no-argument offline constructor to the registry's factory shape.

    The offline providers take a name and no settings, so they cannot be handed
    to :meth:`ProviderRegistry._build` directly.
    """
    return lambda _settings: factory()


def default_specs() -> list[ProviderSpec]:
    """The provider catalogue.

    Declared here rather than imported from the adapter packages so that the
    registry can be constructed — and tested — without pulling in ``playwright``,
    ``tavily`` or ``chromadb``. Task 6 replaces the real factories; the offline
    entries are permanent, because deterministic fixtures are how the test suite
    runs without credentials and without network.
    """
    from app.providers.offline import (
        OfflineBrowserProvider,
        OfflineEmbeddingProvider,
        OfflineGraphStore,
        OfflineLLMProvider,
        OfflineSearchProvider,
        OfflineVectorStore,
        OfflineWebProvider,
    )

    return [
        ProviderSpec("llm", "openai", _unavailable("llm", "openai"),
                     settings_fields=("openai_api_key",)),
        ProviderSpec("llm", "gemini", _unavailable("llm", "gemini"),
                     settings_fields=("gemini_api_key",)),
        ProviderSpec("llm", "offline", _offline(OfflineLLMProvider)),
        ProviderSpec("search", "tavily", _unavailable("search", "tavily"),
                     settings_fields=("tavily_api_key",)),
        ProviderSpec("search", "offline", _offline(OfflineSearchProvider)),
        ProviderSpec("web", "http", _offline(OfflineWebProvider)),
        ProviderSpec("web", "firecrawl", _unavailable("web", "firecrawl"),
                     settings_fields=("firecrawl_api_key",)),
        ProviderSpec("web", "offline", _offline(OfflineWebProvider)),
        ProviderSpec("browser", "browser_use", _unavailable("browser", "browser_use"),
                     settings_fields=("langchain_api_key",)),
        ProviderSpec("browser", "offline", _offline(OfflineBrowserProvider)),
        ProviderSpec("embedding", "openai", _unavailable("embedding", "openai"),
                     settings_fields=("openai_api_key",)),
        ProviderSpec("embedding", "offline", _offline(OfflineEmbeddingProvider)),
        ProviderSpec("vector", "chroma", _unavailable("vector", "chroma")),
        ProviderSpec("vector", "memory", _offline(OfflineVectorStore)),
        ProviderSpec("vector", "offline", _offline(OfflineVectorStore)),
        ProviderSpec("graph", "memory", _offline(OfflineGraphStore)),
        ProviderSpec("graph", "offline", _offline(OfflineGraphStore)),
    ]


_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """The process-wide registry."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry(specs=default_specs())
    return _registry
