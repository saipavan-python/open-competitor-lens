"""OpenAI embeddings.

Shared by the LLM provider's own client where possible, but constructed
separately so a deployment can embed with one key and generate with another.

``embed_query`` and ``embed_documents`` issue one call each rather than batching.
That is a deliberate trade: the retrieval corpus for one company is a few dozen
chunks, and a batch that partially fails leaves the caller unable to tell which
embeddings exist. Batching would be the right call at a scale where the round-trip
dominates; the comment is here so the change is made when that is true rather than
rediscovered.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import ProviderError
from app.core.logging import get_logger

log = get_logger(__name__)


class OpenAIEmbeddingProvider:
    """Embeddings through the OpenAI SDK."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        timeout: int = 60,
        client: Any = None,
    ) -> None:
        if not api_key.strip():
            raise ProviderError(
                "the OpenAI embedding provider needs an API key; set OPENAI_API_KEY",
            )
        self._model = model
        self._dimensions = dimensions
        self._timeout = timeout
        self._client = client or self._build_client(api_key, timeout)

    @staticmethod
    def _build_client(api_key: str, timeout: int) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderError(
                "the openai package is required for openai embeddings",
            ) from exc
        return AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a corpus, preserving order.

        An empty corpus returns an empty list rather than making a call: a caller
        with nothing to index should not be charged for, or blocked by, a request
        that cannot succeed.
        """
        if not texts:
            return []
        vectors = await self._embed(texts)
        if len(vectors) != len(texts):
            raise ProviderError(
                "the embedding response did not match the number of inputs",
            )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query string."""
        vectors = await self._embed([text])
        if not vectors:
            raise ProviderError("the embedding response was empty for a query")
        return vectors[0]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """The one call both public methods share."""
        kwargs: dict[str, Any] = {"model": self._model}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        try:
            result = await self._client.embeddings.create(input=texts, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised as ProviderError below
            raise ProviderError(
                f"the OpenAI embedding request failed: {exc}",
            ) from exc
        # Sorted by index because the API does not promise response order, and a
        # misordered corpus would silently attach the wrong text to each vector.
        ordered = sorted(result.data, key=lambda item: getattr(item, "index", 0))
        return [list(item.embedding) for item in ordered]
