"""Provider protocol conformance and registry resolution.

Two things are being pinned down here. First, that every provider implementation
genuinely satisfies its protocol — a protocol nobody checks at runtime is a
comment. Second, that resolving an unconfigured provider fails with a message
naming the *environment variable* to set, because "provider unavailable" is not
an actionable error at 2am.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderError, ProviderNotConfiguredError
from app.providers.base import (
    BrowserProvider,
    EmbeddingProvider,
    GraphStore,
    LLMProvider,
    ProviderHealth,
    SearchProvider,
    VectorStore,
    WebProvider,
)
from app.providers.registry import ProviderRegistry


class _FakeLLM:
    name = "fake"

    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        return None


class _FakeSearch:
    name = "fake"

    async def search(self, query: str, **kwargs: Any) -> list[Any]:
        return []


def test_protocols_are_runtime_checkable() -> None:
    """A stub implementing only the required methods must satisfy the protocol."""
    assert isinstance(_FakeLLM(), LLMProvider)
    assert isinstance(_FakeSearch(), SearchProvider)
    assert not isinstance(_FakeSearch(), LLMProvider)


def test_resolve_reports_the_missing_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    registry = ProviderRegistry(Settings())
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        registry.resolve("search", "tavily")
    assert "TAVILY_API_KEY" in str(excinfo.value)
    assert excinfo.value.missing_env_var == "TAVILY_API_KEY"


def test_resolve_falls_back_when_the_primary_is_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    registry = ProviderRegistry(Settings())
    provider = registry.resolve("search", "tavily", fallbacks=["offline"])
    assert getattr(provider, "name", None) == "offline"


def test_resolve_raises_when_no_fallback_can_be_satisfied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    registry = ProviderRegistry(Settings())
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        registry.resolve("search", "tavily", fallbacks=["also_missing"])
    assert "TAVILY_API_KEY" in str(excinfo.value)


def test_resolve_rejects_an_unknown_provider() -> None:
    registry = ProviderRegistry(Settings())
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        registry.resolve("search", "not-a-provider")
    assert "not-a-provider" in str(excinfo.value)


def test_offline_is_always_resolvable() -> None:
    """The offline path must never depend on credentials; it is the test harness."""
    registry = ProviderRegistry(Settings())
    assert registry.resolve("search", "offline") is not None
    assert registry.resolve("llm", "offline") is not None


class _Flaky:
    name = "flaky"

    def __init__(self, *, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError("upstream is down", provider="flaky")
        return "recovered"

    async def guarded(self, prompt: str, **kwargs: Any) -> Any:
        self._breaker.allow()
        try:
            return await self.generate(prompt, **kwargs)
        except Exception as exc:
            self._breaker.record_failure(str(exc))
            raise
        else:
            self._breaker.record_success()


class _Breaker:
    pass


def test_health_starts_closed() -> None:
    health = ProviderHealth(name="flaky")
    assert health.healthy
    assert health.consecutive_failures == 0
    assert health.calls == 0


def test_circuit_opens_after_repeated_failures() -> None:
    health = ProviderHealth(name="flaky", failure_threshold=3)
    for _ in range(3):
        health.record_failure("boom")
    assert not health.healthy
    assert health.circuit_open


def test_circuit_stays_open_within_the_cooldown() -> None:
    health = ProviderHealth(name="flaky", failure_threshold=1, cooldown_seconds=60)
    health.record_failure("boom")
    assert not health.can_attempt()


def test_circuit_half_opens_after_the_cooldown() -> None:
    health = ProviderHealth(name="flaky", failure_threshold=1, cooldown_seconds=0.01)
    health.record_failure("boom")
    assert not health.can_attempt()
    time.sleep(0.02)
    assert health.can_attempt()


def test_success_resets_the_failure_count() -> None:
    health = ProviderHealth(name="flaky", failure_threshold=3)
    health.record_failure("boom")
    health.record_failure("boom")
    health.record_success()
    health.record_failure("boom")
    assert health.healthy


def test_rejected_call_is_not_counted_as_an_attempt() -> None:
    """An open circuit must not keep hammering a provider that is down."""
    health = ProviderHealth(name="flaky", failure_threshold=1, cooldown_seconds=60)
    health.record_failure("boom")
    with pytest.raises(ProviderError) as excinfo:
        health.guard(lambda: None)
    assert "circuit" in str(excinfo.value).lower()


def test_guard_returns_the_value_on_success() -> None:
    health = ProviderHealth(name="ok", failure_threshold=1)
    assert health.guard(lambda: 42) == 42
    assert health.calls == 1
    assert health.successes == 1


def test_vector_store_protocol_is_importable() -> None:
    assert VectorStore is not None
    assert WebProvider is not None
    assert BrowserProvider is not None
    assert EmbeddingProvider is not None
    assert GraphStore is not None
