"""Retrieval, in the order the plan requires: filter first, rank second.

The ordering is not an implementation detail, it is the property. If ranking ran
first, a query like "pricing" over a mixed corpus would return the strongest
pricing chunks *from any company*, and the filter would then throw most of them
away — including, on a large corpus, most of what was wanted, since a rival with
a detailed pricing page will outrank the target's thin one. The run would report
"no pricing found" while its own top-k was full of the right chunks.

So :func:`retrieve` filters first, scores only the survivors, and returns at
most ``k`` of them. A filter that matches nothing returns nothing: there is no
"top up with loosely related results" fallback, because a fallback that returns
the wrong company's data is worse than an empty list that a report renders as
"Not publicly available".

Results are returned as :class:`RetrievedChunk`, carrying the filters that
admitted them, so an analyst can tell a retrieved chunk from an inferred one
without re-deriving anything.
"""

from __future__ import annotations

from app.rag.indexing import InvertedIndex
from app.rag.metadata import matches_all, validate_filters
from app.schemas.document import RetrievedChunk

#: Below this, a chunk is not "about" the query. A zero-score chunk is a
#: lexical accident — a shared stopword-like token, a number in a URL — and
#: handing it to an analyst invites them to read a fact into it.
MIN_SCORE = 0.05


def retrieve(
    index: InvertedIndex,
    query: str,
    *,
    filters: dict[str, str] | None = None,
    k: int = 8,
    min_score: float = MIN_SCORE,
) -> list[RetrievedChunk]:
    """Up to ``k`` chunks matching ``query`` and every filter."""
    checked = validate_filters(filters)
    if k <= 0 or not index.documents:
        return []

    # Prefilter, then rank. The order is the contract; see module docstring.
    candidates: list[tuple[int, list[str]]] = []
    for position, chunk in enumerate(index.documents):
        passed, matched_by = matches_all(chunk, checked)
        if passed:
            candidates.append((position, matched_by))

    if not candidates:
        return []

    scores = index.score(query)
    ranked = sorted(
        (
            (position, score, matched_by)
            for position, matched_by in candidates
            if (score := scores.get(position, 0.0)) >= min_score
        ),
        key=lambda item: (-item[1], index.documents[item[0]].id),
    )

    return [
        RetrievedChunk(
            **index.documents[position].model_dump(),
            score=round(score, 6),
            matched_by=matched_by,
            index_id=index.index_id or None,
        )
        for position, score, matched_by in ranked[:k]
    ]


def candidate_positions(
    index: InvertedIndex, filters: dict[str, str] | None
) -> list[int]:
    """Positions passing every filter, without ranking them.

    Exposed so the ordering claim is testable directly: a caller can assert the
    filter shrank the candidate set *before* any score was computed.
    """
    checked = validate_filters(filters)
    return [
        position
        for position, chunk in enumerate(index.documents)
        if matches_all(chunk, checked)[0]
    ]


def all_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Distinct source ids in a result set, in a stable order.

    This is the set an analyst's output is allowed to cite. Handing it to the
    citation check is what makes "the model cited a source nobody retrieved"
    a detectable condition rather than a published one.
    """
    return sorted({chunk.source_id for chunk in chunks if chunk.source_id})
