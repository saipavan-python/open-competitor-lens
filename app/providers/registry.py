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

    Provided so a misconfigured graph store fails at the point of use with an
    explanation, rather than at import time taking down the whole run.
    """

    def build(_settings: Settings) -> Any:
        raise ProviderNotConfiguredError(
            f"{name}", f"{kind.upper()}_PROVIDER_{name.upper()}"
        )

    return build


# --- real factories --------------------------------------------------------
#
# Each is a *function returning a factory*, so that the module-level catalogue in
# ``default_specs()`` stays importable without any vendor SDK. The SDK import
# happens inside the returned factory, which is only called by ``resolve()``. A
# deployment that never resolves OpenAI never imports ``openai``, and the test
# suite can build and inspect the whole catalogue with no vendor package
# installed at all.


def openai_llm_factory() -> Callable[[Settings], Any]:
    """Build the OpenAI chat-completions provider."""

    def build(settings: Settings) -> Any:
        from app.providers.llm.openai import OpenAILLMProvider

        provider = OpenAILLMProvider(
            api_key=settings.openai_api_key or "",
            default_model=settings.openai_model,
            base_url=settings.openai_base_url or None,
            timeout=settings.llm_timeout,
        )
        provider.set_profiles(_model_profiles(settings))
        return provider

    return build


def gemini_llm_factory() -> Callable[[Settings], Any]:
    """Build the Gemini provider."""

    def build(settings: Settings) -> Any:
        from app.providers.llm.gemini import GeminiLLMProvider

        provider = GeminiLLMProvider(
            api_key=settings.gemini_api_key or "",
            default_model=settings.gemini_model,
            timeout=settings.llm_timeout,
        )
        provider.set_profiles(_model_profiles(settings))
        return provider

    return build


def _model_profiles(settings: Settings) -> dict[str, str]:
    """The role-to-model mapping, read once so both LLM providers agree.

    Both adapters need it, and duplicating the three-line mapping in each factory
    is how the two providers end up disagreeing about which model answers a
    ``reasoning`` role.
    """
    roles = ("fast", "reasoning", "final", "default")
    return {role: settings.model_for_role(role) for role in roles}


def tavily_search_factory() -> Callable[[Settings], Any]:
    """Build the Tavily search provider."""

    def build(settings: Settings) -> Any:
        from app.providers.search.tavily import TavilySearchProvider

        return TavilySearchProvider(
            api_key=settings.tavily_api_key or "",
            timeout=settings.search_timeout,
        )

    return build


def http_web_factory() -> Callable[[Settings], Any]:
    """Build the plain-HTTP web provider.

    Takes no credential, so it is always available. The safe fetcher inside is
    what enforces SSRF rules and the private-address block, so this is a real
    provider rather than a stand-in.
    """

    def build(settings: Settings) -> Any:
        from app.providers.web.http import HttpWebProvider

        return HttpWebProvider(timeout=settings.http_timeout)

    return build


def firecrawl_web_factory() -> Callable[[Settings], Any]:
    """Build the Firecrawl provider, for pages that need rendering."""

    def build(settings: Settings) -> Any:
        from app.providers.web.firecrawl import FirecrawlWebProvider

        return FirecrawlWebProvider(
            api_key=settings.firecrawl_api_key, timeout=settings.browser_timeout
        )

    return build


def browser_use_factory() -> Callable[[Settings], Any]:
    """Build the Playwright browser provider.

    The constructor does not launch a browser, so resolving this provider stays
    cheap; the launch happens on first navigation and raises
    ``BrowserUnavailableError`` with the install command if the binaries are
    missing.
    """

    def build(settings: Settings) -> Any:
        from app.providers.browser.browser_use import BrowserUseProvider

        return BrowserUseProvider(timeout=settings.browser_timeout, headless=True)

    return build


def openai_embedding_factory() -> Callable[[Settings], Any]:
    """Build the OpenAI embedding provider."""

    def build(settings: Settings) -> Any:
        from app.providers.embedding.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key or "",
            model=settings.openai_embedding_model,
        )

    return build


def chroma_vector_factory() -> Callable[[Settings], Any]:
    """Build the Chroma vector store.

    No credential, so it is always registered as available. The client is built
    on first use, so this resolves cheaply and a deployment with no write access
    to the data directory fails at the first upsert rather than at startup.
    """

    def build(settings: Settings) -> Any:
        from app.providers.vector.chroma import ChromaVectorStore

        return ChromaVectorStore(path=settings.chroma_persist_directory)

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

    Declared as factories rather than imported instances so that the registry can
    be constructed — and tested — without pulling in ``playwright``, ``tavily`` or
    ``chromadb`` at import time. Each real factory imports its own SDK lazily,
    inside the function, so a deployment that never resolves that provider never
    pays for the dependency.

    The offline entries are permanent, not placeholders: deterministic fixtures
    are how the test suite runs without credentials and without network, and an
    offline provider that was genuinely unconfigured would make every test that
    touches the evidence gate a network test.
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
        ProviderSpec("llm", "openai", openai_llm_factory(),
                     settings_fields=("openai_api_key",)),
        ProviderSpec("llm", "gemini", gemini_llm_factory(),
                     settings_fields=("gemini_api_key",)),
        ProviderSpec("llm", "offline", _offline(OfflineLLMProvider)),
        ProviderSpec("search", "tavily", tavily_search_factory(),
                     settings_fields=("tavily_api_key",)),
        ProviderSpec("search", "offline", _offline(OfflineSearchProvider)),
        ProviderSpec("web", "http", http_web_factory()),
        ProviderSpec("web", "firecrawl", firecrawl_web_factory(),
                     settings_fields=("firecrawl_api_key",)),
        ProviderSpec("web", "offline", _offline(OfflineWebProvider)),
        ProviderSpec("browser", "browser_use", browser_use_factory(),
                     settings_fields=("langchain_api_key",)),
        ProviderSpec("browser", "offline", _offline(OfflineBrowserProvider)),
        ProviderSpec("embedding", "openai", openai_embedding_factory(),
                     settings_fields=("openai_api_key",)),
        ProviderSpec("embedding", "offline", _offline(OfflineEmbeddingProvider)),
        ProviderSpec("vector", "chroma", chroma_vector_factory()),
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
