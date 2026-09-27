"""Chroma-backed vector store.

Chroma is imported inside the methods rather than at module scope, for the same
reason the registry imports its factories lazily: the test suite must be able to
import this module, and to test the collection-name and score-mapping logic, on a
machine where the native extension did not build. An import-time dependency would
turn an optional backend into a hard one.

Two behaviours are worth stating because they are the ones a caller gets wrong:

- Collections are per-company (``company_slug``), and ids are namespaced with the
  company too. Sharing one collection across companies is how a run ends up
  retrieving a competitor's pricing as if it were the target's, and the namespace
  is what makes that impossible rather than merely unlikely.
- ``query`` returns a score, not a similarity. Chroma's distance is
  ``1 - cosine_similarity``, and lower is better, so the score is
  ``1 - distance``. A caller that forgot to invert it would rank the *least*
  relevant chunk first and, because retrieval would still return results, would
  never notice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.schemas.provider import RetrievedChunk

log = get_logger(__name__)


class ChromaVectorStore:
    """A persistent Chroma collection set, one per company."""

    name = "chroma"

    def __init__(
        self,
        *,
        path: str = "./data/chroma",
        host: str | None = None,
        port: int = 8000,
        client: Any = None,
    ) -> None:
        self._path = path
        self._host = host
        self._port = port
        self._client = client
        self._ready = client is not None

    @property
    def client(self) -> Any:
        """The Chroma client, built on first use."""
        if not self._ready:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ProviderError(
                    "the chromadb package is required for the chroma provider",
                ) from exc
            if self._host:
                self._client = chromadb.HttpClient(host=self._host, port=self._port)
            else:
                self._client = chromadb.PersistentClient(path=self._path)
            self._ready = True
        return self._client

    def _collection(self, collection: str) -> Any:
        """Get or create the collection for a company."""
        try:
            return self.client.get_or_create_collection(
                name=collection, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as ProviderError below
            raise ProviderError(
                f"could not open the chroma collection {collection!r}: {exc}"
            ) from exc

    async def upsert(
        self,
        collection: str,
        chunks: Sequence[RetrievedChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store chunks and their vectors, keyed by company and chunk id.

        Chroma rejects an empty upsert, and a caller with nothing to index should
        be a no-op rather than an error.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ProviderError(
                "the number of embeddings does not match the number of chunks",
            )
        handle = self._collection(collection)
        try:
            handle.upsert(
                ids=[self._id(collection, chunk.id) for chunk in chunks],
                embeddings=[list(vector) for vector in embeddings],
                documents=[chunk.text for chunk in chunks],
                metadatas=[self._metadata(collection, chunk) for chunk in chunks],
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as ProviderError below
            raise ProviderError(f"chroma upsert failed: {exc}") from exc
        log.info(
            "chroma upsert", extra={"collection": collection, "chunks": len(chunks)}
        )

    async def query(
        self, collection: str, embedding: Sequence[float], *, limit: int = 8
    ) -> list[RetrievedChunk]:
        """Nearest chunks, scored, nearest first.

        Returns an empty list for a collection that has never been written, which
        is the normal state for the first company analysed in a fresh deployment —
        not an error.
        """
        handle = self._collection(collection)
        try:
            result = handle.query(
                query_embeddings=[list(embedding)], n_results=max(1, limit)
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as ProviderError below
            log.warning("chroma query failed", extra={"collection": collection, "error": str(exc)})
            return []

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for position, stored_id in enumerate(ids):
            metadata = dict(metadatas[position] or {}) if position < len(metadatas) else {}
            text = documents[position] if position < len(documents) else ""
            chunks.append(
                RetrievedChunk(
                    id=self._strip(collection, stored_id),
                    text=text or "",
                    # Inverted: chroma returns a cosine distance, so 1 - distance
                    # is the similarity a higher-is-better score implies.
                    score=self._score(distances[position] if position < len(distances) else 1.0),
                    source_id=metadata.get("source_id"),
                    url=metadata.get("url"),
                    company=metadata.get("company"),
                    competitor=metadata.get("competitor"),
                    page_type=metadata.get("page_type"),
                    category=metadata.get("category"),
                    metadata={k: v for k, v in metadata.items() if isinstance(v, str)},
                )
            )
        return chunks

    async def count(self, collection: str) -> int:
        """How many chunks a company's collection holds."""
        try:
            return int(self._collection(collection).count())
        except Exception as exc:  # noqa: BLE001 - a count must not break a run
            log.warning("chroma count failed", extra={"collection": collection, "error": str(exc)})
            return 0

    async def delete_collection(self, collection: str) -> None:
        """Remove a company's collection entirely, for a re-run from scratch."""
        try:
            self.client.delete_collection(name=collection)
        except Exception as exc:  # noqa: BLE001 - deleting an absent collection is fine
            log.info(
                "chroma collection not deleted",
                extra={"collection": collection, "error": str(exc)},
            )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _id(collection: str, chunk_id: str) -> str:
        """A collection-scoped id, so two companies cannot collide."""
        return f"{collection}::{chunk_id}"

    @classmethod
    def _strip(cls, collection: str, stored_id: str) -> str:
        """The chunk id as it was before namespacing."""
        prefix = f"{collection}::"
        return stored_id[len(prefix) :] if stored_id.startswith(prefix) else stored_id

    @staticmethod
    def _metadata(collection: str, chunk: RetrievedChunk) -> dict[str, Any]:
        """The chunk's fields, as Chroma metadata.

        Chroma accepts only scalars, so a list-valued field is dropped rather than
        stringified: a JSON blob in metadata would be readable but unqueryable,
        which is worse than absent.
        """
        data: dict[str, Any] = {"collection": collection}
        for name in ("source_id", "url", "company", "competitor", "page_type", "category"):
            value = getattr(chunk, name, None)
            if isinstance(value, str) and value:
                data[name] = value
        for name, value in (chunk.metadata or {}).items():
            if isinstance(value, str):
                data[name] = value
        return data

    @staticmethod
    def _score(distance: Any) -> float:
        """A cosine distance as a higher-is-better score."""
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(value):
            return 0.0
        return round(1.0 - value, 6)
