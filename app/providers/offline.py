"""Deterministic in-process provider implementations.

These are not mocks that assert a call was made; they are working substitutes
that satisfy the same protocols with reproducible output. They exist because a
test suite that needs three API keys is a test suite that gets skipped, and a
pipeline that has never been run end to end is a pipeline that breaks in
production.

Every output is a pure function of its input, so a run is reproducible: the same
query returns the same hits, the same URL the same page, the same text the same
embedding. Nothing here reaches the network, and nothing here invents a fact —
the fixtures are declared in the fixture module, and a URL with no fixture
returns ``not_found`` rather than plausible-looking filler.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from app.core.logging import get_logger
from app.providers.base import ProviderHealth
from app.schemas.provider import (
    FetchResult,
    GraphEntity,
    GraphRelationship,
    LLMResponse,
    ModelRole,
    RetrievedChunk,
)
from app.schemas.search import SearchHit

log = get_logger(__name__)

#: Domain vocabulary hashed into deterministic vectors. Not a language model:
#: enough shared structure that semantically similar strings land near each
#: other, which is all the retrieval tests need to be meaningful.
_VOCABULARY: tuple[str, ...] = (
    "pricing", "price", "cost", "plan", "tier", "free", "trial", "seat", "billing",
    "product", "feature", "platform", "api", "integration", "workflow", "automation",
    "support", "service", "uptime", "sla", "guarantee", "refund",
    "competitor", "alternative", "compare", "versus", "vs", "switch", "migration",
    "security", "soc2", "gdpr", "compliance", "encryption", "sso",
    "customer", "user", "team", "company", "review", "testimonial", "case", "study",
    "market", "leader", "leading", "best", "fastest", "easiest",
)


def _digest(*parts: str) -> bytes:
    return hashlib.sha256("|".join(parts).encode("utf-8")).digest()


def _deterministic_int(*parts: str) -> int:
    return int.from_bytes(_digest(*parts)[:8], "big")


class _OfflineBase:
    """Shared bookkeeping: a name and a circuit breaker."""

    def __init__(self, name: str, health: ProviderHealth | None = None) -> None:
        self.name = name
        self.health = health or ProviderHealth(name)

    def snapshot(self) -> dict[str, Any]:
        return self.health.snapshot()


class OfflineLLMProvider(_OfflineBase):
    """Returns schema-shaped JSON derived from the prompt.

    Callers get something structurally valid to parse, so the JSON-handling and
    validation paths are exercised offline. The content is obviously synthetic
    and prefixed as such, which matters: a fixture that looked like real
    evidence would be far more dangerous than one that cannot.
    """

    def __init__(self, name: str = "offline", *, content: str | None = None) -> None:
        super().__init__(name)
        self._fixed_content = content
        self.prompts: list[str] = []

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: type[Any] | dict[str, Any] | None = None,
        model_role: ModelRole = "default",
        temperature: float | None = None,
    ) -> LLMResponse:
        del system, temperature
        self.prompts.append(prompt)
        if self._fixed_content is not None:
            body = self._fixed_content
        elif schema is not None:
            body = json.dumps(_synthesize(schema, prompt))
        else:
            body = f"[offline-fixture] no model available; prompt was {len(prompt)} chars"
        return LLMResponse(
            content=body,
            model=f"offline-{model_role}",
            provider=self.name,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(body) // 4),
            finish_reason="stop",
        )


def _field_defaults(schema: Any) -> dict[str, Any] | None:
    """Best-effort JSON-schema default fields for a Pydantic model or dict."""
    if isinstance(schema, dict):
        return schema
    model_json_schema = getattr(schema, "model_json_schema", None)
    if model_json_schema is None:
        return None
    try:
        json_schema = model_json_schema()
    except Exception:  # noqa: BLE001 - a schema we cannot introspect yields no fields
        return None
    fields: dict[str, Any] = {}
    for name, spec in (json_schema.get("properties") or {}).items():
        if "default" in spec:
            fields[name] = spec["default"]
        elif spec.get("type") == "array":
            fields[name] = []
        elif spec.get("type") == "object":
            fields[name] = {}
        elif spec.get("type") == "boolean":
            fields[name] = False
        elif spec.get("type") == "integer":
            fields[name] = 0
        elif spec.get("type") == "number":
            fields[name] = 0.0
        else:
            fields[name] = "[offline-fixture] unknown"
    return fields


def _synthesize(schema: Any, prompt: str) -> dict[str, Any]:
    fields = _field_defaults(schema) or {}
    fields.setdefault("_offline", True)
    fields.setdefault("_source", "offline-fixture")
    del prompt
    return fields


class OfflineSearchProvider(_OfflineBase):
    """Deterministic search results over a declared result set.

    Only URLs registered through :meth:`add` are ever returned. An unregistered
    query yields an empty list, because the whole point of the evidence rules is
    that the system must never manufacture a source.
    """

    def __init__(self, name: str = "offline", results: list[SearchHit] | None = None
                 ) -> None:
        super().__init__(name)
        self._results: list[SearchHit] = list(results or [])

    def add(
        self, url: str, *, title: str | None = None, snippet: str | None = None,
        terms: list[str] | None = None,
    ) -> SearchHit:
        hit = SearchHit(
            url=url,
            title=title,
            snippet=snippet,
            domain=url.split("/")[2] if "://" in url else None,
            provider=self.name,
        )
        hit.metadata = " ".join(terms or [])  # type: ignore[attr-defined]
        self._results.append(hit)
        return hit

    async def search(self, query: str, *, limit: int = 10, **kwargs: Any) -> list[SearchHit]:
        del kwargs
        terms = {word for word in query.lower().split() if len(word) > 2}
        scored: list[tuple[int, int, SearchHit]] = []
        for index, hit in enumerate(self._results):
            haystack = " ".join(
                part for part in (hit.title, hit.snippet, getattr(hit, "metadata", ""))
                if part
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((-score, index, hit))
        scored.sort()
        return [hit.model_copy(update={"rank": rank})
                for rank, (_, _, hit) in enumerate(scored[:limit], start=1)]


class OfflineWebProvider(_OfflineBase):
    """Serves declared fixtures by URL; anything else is ``not_found``."""

    def __init__(self, name: str = "offline", pages: dict[str, FetchResult] | None = None
                 ) -> None:
        super().__init__(name)
        self._pages: dict[str, FetchResult] = dict(pages or {})

    def add(
        self, url: str, *, text: str = "", html: str | None = None,
        status: int = 200, title: str | None = None, content_type: str = "text/html",
    ) -> FetchResult:
        result = FetchResult(
            url=url,
            final_url=url,
            status=status,
            content_type=content_type,
            text=text,
            html=html if html is not None else f"<html><body>{text}</body></html>",
            title=title or url,
            provider=self.name,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            content_length=len(text),
        )
        self._pages[url] = result
        return result

    async def fetch(self, url: str) -> FetchResult:
        return self._pages.get(url, FetchResult(
            url=url, final_url=url, status=404, retrieval_status="not_found",
            error="no offline fixture registered for this URL", provider=self.name,
        ))

    async def extract(self, url: str, schema: type[Any]) -> dict[str, Any]:
        result = await self.fetch(url)
        if not result.ok:
            raise LookupError(result.error)
        return _field_defaults(schema) or {"text": result.text}


class OfflineBrowserProvider(_OfflineBase):
    """A browser that renders whatever the web provider returned.

    It cannot do anything a real browser can, and does not pretend to: the
    escalation path in Task 6 is what decides a page needs a browser, and this
    makes that path testable without one installed.
    """

    def __init__(self, web: OfflineWebProvider | None = None, name: str = "offline"
                 ) -> None:
        super().__init__(name)
        self.web = web or OfflineWebProvider()
        self._current: str | None = None

    async def open(self, url: str) -> None:
        self._current = url

    async def navigate(self, url: str) -> FetchResult:
        self._current = url
        return await self.web.fetch(url)

    async def extract(self, schema: type[Any]) -> dict[str, Any]:
        if self._current is None:
            raise RuntimeError("no page is open")
        return await self.web.extract(self._current, schema)

    async def close(self) -> None:
        self._current = None


class OfflineEmbeddingProvider(_OfflineBase):
    """Hashed bag-of-words embeddings over a fixed vocabulary.

    Similar strings get similar vectors, which is the property retrieval depends
    on. It captures no semantics beyond shared words, and must not be used
    outside tests.
    """

    def __init__(self, name: str = "offline", *, dimensions: int = 64) -> None:
        super().__init__(name)
        self.dimensions = dimensions
        self._cache: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        lowered = text.lower()
        vector = [0.0] * self.dimensions
        for index, word in enumerate(_VOCABULARY):
            if word in lowered:
                vector[index % self.dimensions] += 1.0
        for position, char in enumerate(lowered):
            if not char.isalnum() and not char.isspace():
                vector[(position * 7) % self.dimensions] += 0.1
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        self._cache[text] = vector
        return vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


class OfflineVectorStore(_OfflineBase):
    """In-memory vector store with exact cosine search."""

    def __init__(self, name: str = "memory") -> None:
        super().__init__(name)
        self._collections: dict[str, dict[str, tuple[RetrievedChunk, list[float]]]] = {}

    async def upsert(
        self, collection: str, chunks: list[RetrievedChunk], embeddings: list[list[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        store = self._collections.setdefault(collection, {})
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            store[chunk.id] = (chunk, list(embedding))

    async def query(
        self, collection: str, embedding: list[float], *, limit: int = 8
    ) -> list[RetrievedChunk]:
        store = self._collections.get(collection, {})
        scored = [
            (cosine_similarity(embedding, vector), chunk)
            for chunk, vector in store.values()
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk.model_copy(update={"score": round(score, 6)})
                for score, chunk in scored[:limit] if score > 0]

    async def delete_collection(self, collection: str) -> None:
        self._collections.pop(collection, None)

    async def count(self, collection: str) -> int:
        return len(self._collections.get(collection, {}))


class OfflineGraphStore(_OfflineBase):
    """In-memory entity and relationship store for GraphRAG tests."""

    def __init__(self, name: str = "memory") -> None:
        super().__init__(name)
        self.entities: dict[str, GraphEntity] = {}
        self.relationships: dict[str, GraphRelationship] = {}

    async def upsert_entities(self, entities: list[GraphEntity]) -> None:
        for entity in entities:
            existing = self.entities.get(entity.id)
            if existing is not None:
                entity = entity.model_copy(update={
                    "source_ids": sorted(set(existing.source_ids) | set(entity.source_ids))
                })
            self.entities[entity.id] = entity

    async def upsert_relationships(self, relationships: list[GraphRelationship]) -> None:
        for relationship in relationships:
            self.relationships[relationship.id] = relationship

    async def query(self, text: str, *, limit: int = 8) -> list[GraphEntity]:
        terms = {word for word in text.lower().split() if len(word) > 2}
        scored: list[tuple[int, str, GraphEntity]] = []
        for entity in self.entities.values():
            haystack = f"{entity.name} {entity.type}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((-score, entity.id, entity))
        scored.sort()
        return [entity for _, _, entity in scored[:limit]]
