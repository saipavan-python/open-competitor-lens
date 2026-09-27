"""The run's retrieval facade, and the only thing analysts ask for chunks.

Everything above this line — analysts, graph nodes, the report builder — calls
:meth:`RagService.retrieve` and never touches an index, a token or a filter.
That is a deliberate narrowing: the prefilter-before-rank rule is the kind of
thing that gets quietly broken by a caller who "just wants the top chunks", and
the way to make it unbreakable is to make the index unreachable.

The service also holds *one* index per run. Analysts for nine categories run
against the same documents, and giving each its own index would mean nine
chunker runs, nine vocabularies, and nine chances for two analysts to disagree
about which chunk says what.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.core.logging import get_logger
from app.rag.chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, chunk_documents
from app.rag.indexing import InvertedIndex, build_index
from app.rag.metadata import validate_filters
from app.rag.reranking import RerankWeights, rerank
from app.rag.retrieval import MIN_SCORE, retrieve
from app.schemas.document import Chunk, Document, RetrievedChunk

log = get_logger(__name__)

#: Enough to cover a pricing page, a product page and a few reviews without
#: carrying the whole corpus into a prompt.
DEFAULT_K = 8


class RagService:
    """Indexes a run's documents and answers filtered, reranked queries."""

    def __init__(
        self,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap: int = DEFAULT_OVERLAP,
        min_score: float = MIN_SCORE,
        default_k: int = DEFAULT_K,
        weights: RerankWeights | None = None,
    ) -> None:
        self._index: InvertedIndex | None = None
        self._max_chars = max_chars
        self._overlap = overlap
        self._min_score = min_score
        self._default_k = default_k
        self._weights = weights or RerankWeights()

    # -- indexing ---------------------------------------------------------

    def index(self, documents: Iterable[Document]) -> str:
        """(Re)build the index over ``documents`` and return its id.

        Replaces rather than merges. A run has one view of its documents, and a
        second index over a superset would let a stale page from an earlier
        phase answer a query about a re-crawled one.
        """
        batch = list(documents)
        chunked: list[Chunk] = chunk_documents(
            batch, max_chars=self._max_chars, overlap=self._overlap
        )
        self._index = build_index(chunked)
        log.info(
            "rag index built",
            extra={
                "index_id": self._index.index_id,
                "documents": len(batch),
                "chunks": self._index.size,
            },
        )
        return self._index.index_id

    def index_id(self) -> str | None:
        return self._index.index_id if self._index else None

    @property
    def size(self) -> int:
        return self._index.size if self._index else 0

    @property
    def is_empty(self) -> bool:
        return not self._index or not self._index.size

    def chunks(self) -> list[Chunk]:
        return list(self._index.documents) if self._index else []

    # -- retrieval --------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, str] | None = None,
        k: int | None = None,
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        """Up to ``k`` chunks for ``query``, filtered before ranking.

        Returns ``[]`` rather than raising when nothing matches or nothing is
        indexed. An analyst's job is then to report the category as not publicly
        available, which is a finding; an exception would take down the run.
        """
        if self._index is None:
            log.info("retrieve on an empty index", extra={"query": query})
            return []

        checked = validate_filters(filters)
        found = retrieve(
            self._index,
            query,
            filters=checked,
            k=k or self._default_k,
            min_score=self._min_score,
        )
        if rerank and found:
            found = self._rerank(query, found)
        log.info(
            "retrieved",
            extra={"query": query, "filters": checked, "chunks": len(found)},
        )
        return found

    def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        filters: dict[str, str] | None = None,
        k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Several queries, merged and de-duplicated by chunk id.

        An analyst usually asks the same category several ways. Merging by id
        keeps a chunk that answered two questions out of the prompt twice, which
        would otherwise crowd out the chunks that only answered one.
        """
        merged: dict[str, RetrievedChunk] = {}
        for query in queries:
            for chunk in self.retrieve(query, filters=filters, k=k):
                existing = merged.get(chunk.id)
                if existing is None or chunk.score > existing.score:
                    merged[chunk.id] = chunk
        return sorted(merged.values(), key=lambda item: (-item.score, item.id))

    def sources_for(self, chunks: Sequence[RetrievedChunk]) -> list[str]:
        """The source ids a chunk set may legitimately cite."""
        return sorted({chunk.source_id for chunk in chunks if chunk.source_id})

    # -- internals --------------------------------------------------------

    def _rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        return rerank(query, chunks, weights=self._weights)
