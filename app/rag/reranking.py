"""Reranking: a second pass over an already-filtered result set.

The first-pass ranker answers "does this chunk contain the query's words". It
cannot answer "is this the *best* one of the near-misses", which is a different
question and matters here for a specific reason: a pricing question retrieves
the pricing page, the pricing page's footer, and a blog post that mentions
pricing once. Lexically they are close. Only one of them is a source of prices.

So the reranker applies the small number of signals that reliably track
usefulness and are computable without a model:

* **Page type.** A pricing question wants ``PageType.PRICING``. This is the
  strongest signal available, because page type comes from the URL and title
  rather than from the prose, so it is not fooled by a chunk that says the word
  "pricing" in a blog post.
* **Heading match.** A chunk whose heading contains the query's key noun is
  about the thing asked about.
* **Term coverage.** How much of the query a chunk actually covers, which
  penalises a chunk matching one common term out of three.
* **Position.** Earlier results are more likely to be the intended one, so a
  small prior keeps reranking from promoting a marginal result over a clear one.

Deliberately *not* here: any similarity to a model embedding. A reranker that can
only be inspected by trusting the model is a reranker whose mistakes are
unexplainable in a report that has to justify every citation.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.constants import PageType
from app.rag.indexing import tokenize
from app.schemas.document import RetrievedChunk

#: Page types that answer a question about a company's own commercial terms.
COMMERCIAL_PAGE_TYPES: frozenset[PageType] = frozenset(
    {PageType.PRICING, PageType.PRODUCT, PageType.SERVICE, PageType.SOLUTION}
)

PAGE_TYPE_BOOST = 0.55
HEADING_BOOST = 0.25
COVERAGE_BOOST = 0.35
POSITION_DECAY = 0.02


@dataclass
class RerankWeights:
    """Tunable weights, kept in one place so they can be reasoned about."""

    page_type: float = PAGE_TYPE_BOOST
    heading: float = HEADING_BOOST
    coverage: float = COVERAGE_BOOST
    position: float = POSITION_DECAY


def coverage(query: str, chunk: RetrievedChunk) -> float:
    """The fraction of the query's distinct terms the chunk contains."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    chunk_terms = set(tokenize(f"{chunk.heading or ''} {chunk.text}"))
    return len(query_terms & chunk_terms) / len(query_terms)


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    weights: RerankWeights | None = None,
) -> list[RetrievedChunk]:
    """Re-score and re-order an already-filtered set.

    Order is stable for equal scores, so two runs over the same retrieval
    produce the same context and therefore the same prompt.
    """
    weights = weights or RerankWeights()
    query_terms = set(tokenize(query))
    rescored: list[tuple[float, str, RetrievedChunk]] = []

    for position, chunk in enumerate(chunks):
        bonus = 0.0
        if chunk.page_type in COMMERCIAL_PAGE_TYPES:
            bonus += weights.page_type
        if query_terms and chunk.heading:
            heading_terms = set(tokenize(chunk.heading))
            if query_terms & heading_terms:
                bonus += weights.heading
        bonus += weights.coverage * coverage(query, chunk)
        bonus += weights.position * max(0.0, 1.0 - position / max(1, len(chunks)))

        rescored.append(
            (chunk.score + bonus, chunk.id, chunk.model_copy())
        )

    ordered = sorted(rescored, key=lambda item: (-item[0], item[1]))
    for score, _, chunk in ordered:
        chunk.score = round(score, 6)
    return [chunk for _, _, chunk in ordered]
