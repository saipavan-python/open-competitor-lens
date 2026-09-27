"""The index: an inverted term index over chunks, with no external dependency.

The scorer is deliberately lexical rather than an embedding lookup. Two reasons,
in order of importance:

1. **It runs offline and deterministically.** A run has to produce the same
   retrieval for the same inputs whether or not a vector backend is configured,
   because a comparison report that cites different sources on a re-run is not
   reproducible and cannot be reviewed. Every ranking decision must be one a
   reader could check.
2. **It is inspectable.** When an analyst cites a pricing figure, the chain from
   that figure back to a chunk, a document and a URL is a walk through this
   index, not an opaque vector neighbourhood.

The trade is real and worth naming: a lexical index matches surface words, so
it will find "pricing" and "costs" separately and will miss a paraphrase that
shares no tokens. That is a recall limitation, not a correctness one — a chunk it
fails to retrieve is a finding the run reports as absent, which is the safe
direction. Where embeddings are configured later, this is the seam: `index`
gains a vector side, and `rank` blends the two scores.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from app.schemas.document import Chunk
from app.utils.hashing import stable_id

#: Words carrying no retrieval signal. Kept short: an aggressive stop list
#: discards real query terms ("free", "enterprise", "vs") that are exactly the
#: ones a pricing question turns on.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "this", "to", "was", "were", "what", "when", "which", "will",
        "with",
    }
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

#: Short for BM25's k1: chunks here are 1500 characters, so saturation happens
#: quickly and a larger k1 would flatten the difference between a term that
#: appears once and one that appears five times.
BM25_K1 = 1.2
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric terms, stopwords removed, order preserved."""
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if token not in STOPWORDS and len(token) > 1
    ]


@dataclass
class InvertedIndex:
    """Term -> chunk positions, plus the corpus statistics BM25 needs."""

    index_id: str = ""
    documents: list[Chunk] = field(default_factory=list)
    postings: dict[str, set[int]] = field(default_factory=dict)
    #: Per-chunk term frequencies. Postings are a *set* of positions, so they
    #: answer "which chunks contain this term" but not "how often" — and BM25
    #: needs the second to tell a passing mention from a page about the subject.
    term_counts: list[Counter] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    _position_by_id: dict[str, int] = field(default_factory=dict, repr=False)
    _by_id: dict[str, Chunk] = field(default_factory=dict, repr=False)

    @property
    def size(self) -> int:
        return len(self.documents)

    @property
    def average_length(self) -> float:
        return (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._by_id

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def add(self, chunks: list[Chunk]) -> InvertedIndex:
        """Add chunks, replacing any with the same id.

        Re-indexing a document is normal — a re-crawl produces a new version of
        the same page — so a chunk id is a statement about content, not about
        when it was first seen.
        """
        for chunk in chunks:
            existing = self._position_by_id.get(chunk.id)
            if existing is not None:
                self._remove_at(existing)
            position = len(self.documents)
            counts = Counter(tokenize(f"{chunk.heading or ''} {chunk.title or ''} {chunk.text}"))
            self.documents.append(chunk)
            self.lengths.append(sum(counts.values()))
            self.term_counts.append(counts)
            self._position_by_id[chunk.id] = position
            self._by_id[chunk.id] = chunk
            for term in counts:
                self.postings.setdefault(term, set()).add(position)
        return self

    def _remove_at(self, position: int) -> None:
        """Drop a chunk, keeping the postings' positions valid.

        Everything after ``position`` shifts down by one, so the whole tail is
        rewritten. Chunks-per-document is in the hundreds, so this is cheap, and
        it keeps postings as plain integer sets — no tombstones to leak.
        """
        chunk = self.documents[position]
        del self._position_by_id[chunk.id]
        del self._by_id[chunk.id]
        del self.documents[position]
        del self.term_counts[position]
        del self.lengths[position]
        for term, positions in list(self.postings.items()):
            positions.discard(position)
            for other in list(positions):
                if other > position:
                    positions.discard(other)
                    positions.add(other - 1)
            if not positions:
                del self.postings[term]

    def score(self, query: str) -> dict[int, float]:
        """BM25 scores by position. Terms absent from the index score nothing."""
        terms = tokenize(query)
        if not terms or not self.documents:
            return {}

        total = len(self.documents)
        average = self.average_length or 1.0
        scores: dict[int, float] = {}
        for term in set(terms):
            positions = self.postings.get(term)
            if not positions:
                continue
            # BM25's idf, floored at zero: a term present in more than half the
            # corpus carries no discriminating information and would otherwise
            # subtract from every candidate it appears in.
            idf = math.log(1.0 + (total - len(positions) + 0.5) / (len(positions) + 0.5))
            idf = max(0.0, idf)
            for position in positions:
                length = self.lengths[position] or 1
                tf = self.term_counts[position][term]
                denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (length / average))
                scores[position] = scores.get(position, 0.0) + idf * (tf / denominator)
        return scores

    def finish(self) -> InvertedIndex:
        if not self.index_id:
            self.index_id = stable_id("rag-index", *(chunk.id for chunk in self.documents))
        return self


def build_index(chunks: list[Chunk]) -> InvertedIndex:
    """A ready-to-query index over ``chunks``."""
    return InvertedIndex().add(chunks).finish()
