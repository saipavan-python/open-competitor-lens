"""Provider protocols and the circuit breaker that guards them.

Everything that leaves the process goes through one of these protocols. The
project depends on several external services, and the point of the protocols is
that a run can substitute any of them — a real API in production, deterministic
fixtures in tests — without the orchestrator noticing.

They are declared as :class:`typing.Protocol` rather than abstract base classes
so an adapter does not have to import a base it will never use, and so third
party implementations satisfy them structurally.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

from app.core.exceptions import ProviderError
from app.schemas.provider import (
    FetchResult,
    GraphEntity,
    GraphRelationship,
    LLMResponse,
    ModelRole,
    RetrievedChunk,
)
from app.schemas.search import SearchHit

T = TypeVar("T")


@runtime_checkable
class LLMProvider(Protocol):
    """A text generation backend."""

    name: str

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: type[Any] | dict[str, Any] | None = None,
        model_role: ModelRole = "default",
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return a completion.

        ``model_role`` selects a profile rather than a concrete model id, so the
        model choice stays in configuration and a prompt never hardcodes one.
        ``schema`` is a hint for structured output; implementations that cannot
        enforce it may validate the response themselves, but they must not
        silently return prose where JSON was requested.
        """
        ...


@runtime_checkable
class SearchProvider(Protocol):
    """A web search backend. Discovery only — a hit is not a retrieved source."""

    name: str

    async def search(
        self, query: str, *, limit: int = 10, **kwargs: Any
    ) -> list[SearchHit]:
        ...


@runtime_checkable
class WebProvider(Protocol):
    """A page fetcher and content extractor."""

    name: str

    async def fetch(self, url: str) -> FetchResult:
        """Retrieve one page.

        Implementations must run the result through
        :func:`app.utils.urls.validate_url`, because this is the boundary where
        a user-supplied URL becomes an outbound request.
        """
        ...

    async def extract(self, url: str, schema: type[Any]) -> dict[str, Any]:
        """Retrieve a page and return its content shaped to ``schema``."""
        ...


@runtime_checkable
class BrowserProvider(Protocol):
    """A real browser, for pages that server-side retrieval cannot reach."""

    name: str

    async def open(self, url: str) -> None: ...

    async def navigate(self, url: str) -> FetchResult: ...

    async def extract(self, schema: type[Any]) -> dict[str, Any]: ...

    async def close(self) -> None: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A text embedding backend."""

    name: str

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    async def embed_query(self, text: str) -> list[float]:
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Similarity search over embedded chunks."""

    name: str

    async def upsert(
        self, collection: str, chunks: list[RetrievedChunk], embeddings: list[list[float]]
    ) -> None: ...

    async def query(
        self, collection: str, embedding: list[float], *, limit: int = 8
    ) -> list[RetrievedChunk]: ...

    async def delete_collection(self, collection: str) -> None: ...

    async def count(self, collection: str) -> int: ...


@runtime_checkable
class GraphStore(Protocol):
    """Entity/relationship storage backing GraphRAG."""

    name: str

    async def upsert_entities(self, entities: list[GraphEntity]) -> None: ...

    async def upsert_relationships(
        self, relationships: list[GraphRelationship]
    ) -> None: ...

    async def query(self, text: str, *, limit: int = 8) -> list[GraphEntity]:
        ...


class ProviderHealth:
    """Per-provider call accounting and a circuit breaker.

    A provider that is down must not be called once per competitor for the rest
    of the run: that converts one outage into a quota exhaustion, and the
    resulting errors bury the real cause. After ``failure_threshold``
    consecutive failures the circuit opens and calls are refused for
    ``cooldown_seconds``, after which one probe is allowed through.

    Not thread-safe. Provider calls are awaited on one event loop, and a lock
    would add ceremony without adding correctness at that concurrency level.
    """

    __slots__ = (
        "name", "failure_threshold", "cooldown_seconds", "calls", "successes",
        "failures", "consecutive_failures", "last_error", "last_success_at",
        "_opened_at", "_was_refused",
    )

    def __init__(
        self, name: str, *, failure_threshold: int = 5, cooldown_seconds: float = 60.0
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = cooldown_seconds
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.last_error: str | None = None
        self.last_success_at: float | None = None
        self._opened_at: float | None = None
        self._was_refused = False

    @property
    def circuit_open(self) -> bool:
        """True while the circuit is refusing calls."""
        return not self.can_attempt()

    @property
    def healthy(self) -> bool:
        return self.consecutive_failures < self.failure_threshold

    @property
    def was_refused(self) -> bool:
        """True when the last :meth:`can_attempt` check refused the call."""
        return self._was_refused

    def can_attempt(self) -> bool:
        """Whether a call is currently permitted."""
        self._was_refused = False
        if self._opened_at is None:
            return True
        if time.monotonic() - self._opened_at >= self.cooldown_seconds:
            # Half-open: let exactly one probe through. If it fails, the next
            # can_attempt re-opens the circuit for another cooldown period.
            self._opened_at = None
            return True
        self._was_refused = True
        return False

    def record_success(self) -> None:
        self.calls += 1
        self.successes += 1
        self.consecutive_failures = 0
        self.last_error = None
        self.last_success_at = time.monotonic()
        self._opened_at = None

    def record_failure(self, error: str = "") -> None:
        self.calls += 1
        self.failures += 1
        self.consecutive_failures += 1
        self.last_error = error
        if self.consecutive_failures >= self.failure_threshold:
            self._opened_at = time.monotonic()

    def guard[T](self, call: Callable[[], T]) -> T:
        """Run ``call``, recording the outcome and enforcing the breaker.

        A refused call raises without reaching the provider, and is deliberately
        *not* counted as an attempt: incrementing the counters while the circuit
        is open would keep it open forever.
        """
        if not self.can_attempt():
            raise ProviderError(
                f"Circuit breaker for provider '{self.name}' is open after "
                f"{self.consecutive_failures} consecutive failures; "
                f"retry in {self.cooldown_seconds:.0f}s",
                provider=self.name,
                retryable=True,
            )
        try:
            result = call()
        except Exception as exc:
            self.record_failure(str(exc))
            raise
        else:
            self.record_success()
            return result

    async def aguard(self, call: Callable[[], Any]) -> Any:
        """Await an async ``call`` under the same breaker rules."""
        if not self.can_attempt():
            raise ProviderError(
                f"Circuit breaker for provider '{self.name}' is open after "
                f"{self.consecutive_failures} consecutive failures; "
                f"retry in {self.cooldown_seconds:.0f}s",
                provider=self.name,
                retryable=True,
            )
        try:
            result = await call()
        except Exception as exc:
            self.record_failure(str(exc))
            raise
        else:
            self.record_success()
            return result

    def reset(self) -> None:
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.last_error = None
        self._opened_at = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "consecutive_failures": self.consecutive_failures,
            "circuit_open": self.circuit_open,
            "last_error": self.last_error,
        }
