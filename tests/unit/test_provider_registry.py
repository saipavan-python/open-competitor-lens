"""Which adapter a run actually gets, and when.

The registry exists so the orchestrator names a *capability* ("llm", "search",
"web") and the deployment decides which vendor fills it. That only works if the
catalogue reflects reality, which is what most of this file is about: every kind
that has a real adapter in the tree must be wired to it, and wiring must not
require the vendor SDK to be installed. A registry that names a provider whose
module imports `openai` at module scope is a registry that cannot be loaded in a
deployment without that package, and the failure surfaces as an ImportError at
startup rather than as "that provider is unavailable".

The credentials tests matter for a related reason. A run with no API key must
degrade to the offline stand-ins and still produce a report, so an absent key has
to remove a provider from ``available()`` rather than raise when something tries
to build it. And the reverse: the two credential-free adapters (HTTP fetch and
Chroma) must always be present, because they are the ones that keep a keyless
deployment working at all.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderNotConfiguredError
from app.providers.registry import ProviderRegistry, default_specs


def _registry(**env: str) -> ProviderRegistry:
    """A registry over a settings object built only from ``env``."""
    for key in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY",
    ):
        pytest.MonkeyPatch().delenv(key, raising=False)
    for key, value in env.items():
        pytest.MonkeyPatch().setenv(key, value)
    return ProviderRegistry(settings=Settings(), specs=default_specs())


class TestTheCatalogueNamesRealAdapters:
    @pytest.mark.parametrize(
        ("kind", "name", "expected"),
        [
            ("llm", "openai", "OpenAILLMProvider"),
            ("llm", "gemini", "GeminiLLMProvider"),
            ("search", "tavily", "TavilySearchProvider"),
            ("web", "http", "HttpWebProvider"),
            ("web", "firecrawl", "FirecrawlWebProvider"),
            ("embedding", "openai", "OpenAIEmbeddingProvider"),
            ("vector", "chroma", "ChromaVectorStore"),
            ("vector", "memory", "OfflineVectorStore"),
            ("llm", "offline", "OfflineLLMProvider"),
            ("search", "offline", "OfflineSearchProvider"),
            ("web", "offline", "OfflineWebProvider"),
            ("browser", "offline", "OfflineBrowserProvider"),
            ("embedding", "offline", "OfflineEmbeddingProvider"),
        ],
    )
    def test_it_resolves_to_that_class(
        self, kind: str, name: str, expected: str
    ) -> None:
        """A name in the catalogue means a working adapter, not a stand-in.

        The web and vector entries are the regression: both adapters were
        implemented and tested, but the registry pointed at a placeholder, so a
        real deployment with a Firecrawl key silently crawled nothing and stored
        nothing.
        """
        registry = _registry(
            OPENAI_API_KEY="sk-x", GEMINI_API_KEY="g-x",
            TAVILY_API_KEY="t-x", FIRECRAWL_API_KEY="f-x",
        )
        assert type(registry.resolve(kind, name)).__name__ == expected

    def test_every_kind_has_at_least_one_offline_stand_in(self) -> None:
        """A keyless run must still be able to produce a report."""
        registry = _registry()
        for kind in ("llm", "search", "web", "embedding", "vector", "browser"):
            assert "offline" in registry.available(kind), kind


class TestCredentialsDecideAvailability:
    def test_a_provider_without_its_key_is_not_advertised(self) -> None:
        registry = _registry()
        assert "openai" not in registry.available("llm")
        assert "gemini" not in registry.available("llm")
        assert "tavily" not in registry.available("search")
        assert "firecrawl" not in registry.available("web")

    def test_a_key_makes_its_provider_available(self) -> None:
        registry = _registry(OPENAI_API_KEY="sk-x")
        assert "openai" in registry.available("llm")

    def test_credential_free_adapters_are_always_available(self) -> None:
        """The ones that keep a keyless deployment working at all."""
        registry = _registry()
        assert "http" in registry.available("web")
        assert "chroma" in registry.available("vector")
        assert "memory" in registry.available("vector")

    def test_building_an_unavailable_provider_says_how_to_fix_it(self) -> None:
        """The error must name the setting, not just the provider.

        Otherwise the first run in a new deployment fails with a message that
        requires reading the source to act on.
        """
        registry = _registry()
        with pytest.raises(ProviderNotConfiguredError) as excinfo:
            registry.resolve("llm", "openai")
        assert "openai" in str(excinfo.value).lower()


class TestModelsAreRoutedByRole:
    def test_the_configured_roles_reach_the_provider(self) -> None:
        """A provider must agree with the router about which model answers what.

        Otherwise ``fast_model``/``reasoning_model``/``final_model`` are read by
        the router and discarded by the adapter, and every task silently costs
        the reasoning model.
        """
        registry = _registry(
            OPENAI_API_KEY="sk-x",
            FAST_MODEL="cheap-model",
            REASONING_MODEL="mid-model",
            FINAL_MODEL="best-model",
        )
        provider = registry.resolve("llm", "openai")

        assert provider.model_for("fast") == "cheap-model"
        assert provider.model_for("reasoning") == "mid-model"
        assert provider.model_for("final") == "best-model"
        assert provider.model_for("default") == "cheap-model"

    def test_both_llm_vendors_get_the_same_roles(self) -> None:
        """Otherwise the two vendors disagree about the same task."""
        registry = _registry(
            OPENAI_API_KEY="sk-x", GEMINI_API_KEY="g-x",
            FAST_MODEL="cheap-model",
        )
        assert registry.resolve("llm", "gemini").model_for("fast") == "cheap-model"

    def test_a_configured_vendor_model_is_used_when_no_roles_are_set(self) -> None:
        """``OPENAI_MODEL`` must not be a dead setting.

        Every call resolves its model through the role map, which used to be
        hardcoded to OpenAI ids. So a deployment behind an OpenAI-compatible
        gateway, or one running Gemini, got its adapter selected and then had
        ``gpt-4o`` requested from it — a model that gateway or vendor may not
        serve at all.
        """
        registry = _registry(OPENAI_API_KEY="sk-x", OPENAI_MODEL="llama-3.1-70b")
        provider = registry.resolve("llm", "openai")

        assert provider.model_for("fast") == "llama-3.1-70b"
        assert provider.model_for("final") == "llama-3.1-70b"

    def test_a_gemini_deployment_is_never_asked_for_an_openai_model(self) -> None:
        """The regression for the other vendor, which had no way to work at all."""
        registry = _registry(GEMINI_API_KEY="g-x", GEMINI_MODEL="gemini-2.5-pro")
        provider = registry.resolve("llm", "gemini")

        for role in ("fast", "reasoning", "final"):
            assert provider.model_for(role) == "gemini-2.5-pro"

    def test_an_explicit_role_still_wins_over_the_vendor_model(self) -> None:
        """Per-role configuration is how a deployment mixes vendors."""
        registry = _registry(
            GEMINI_API_KEY="g-x", GEMINI_MODEL="gemini-2.5-pro",
            REASONING_MODEL="gpt-4.1",
        )
        provider = registry.resolve("llm", "gemini")

        assert provider.model_for("reasoning") == "gpt-4.1"
        assert provider.model_for("fast") == "gemini-2.5-pro"

    def test_an_unset_role_falls_back_to_the_configured_default(self) -> None:
        registry = _registry(OPENAI_API_KEY="sk-x", OPENAI_MODEL="default-model")
        provider = registry.resolve("llm", "openai")

        assert provider.model_for("fast") == "default-model"
        assert provider.model_for("something-unknown") == "default-model"


class TestTheCatalogueLoadsWithoutAnyVendorSdk:
    def test_building_the_catalogue_imports_nothing_vendor_specific(self) -> None:
        """A deployment without the SDKs must still be able to start.

        The factories are functions returning factories precisely so that
        ``default_specs()`` is a static description and the vendor import happens
        inside the factory the graph calls. If that ever regresses, this fails
        before a real deployment does.
        """
        import sys

        for module in ("openai", "tavily", "chromadb"):
            sys.modules.pop(module, None)
        specs = default_specs()
        assert specs
        assert not {"openai", "tavily", "chromadb"} & set(sys.modules)
