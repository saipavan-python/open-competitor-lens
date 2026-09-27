"""Document and fetched-page schemas.

``FetchedPage`` is the raw retrieval result. ``Document`` is the cleaned,
chunkable, hash-identified artefact that gets indexed and cited.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from app.core.constants import PageType, RetrievalStatus
from app.schemas.common import DomainModel, utcnow


class FetchedPage(DomainModel):
    """The result of retrieving one URL, successful or not.

    A failed fetch is still a value: it records *why* a source is absent, which
    is what lets a run report "not publicly available" honestly instead of
    silently omitting the category.
    """

    url: str
    final_url: str | None = None
    status: int | None = None
    content_type: str | None = None
    text: str = ""
    html: str = ""
    title: str | None = None
    retrieved_at: datetime = Field(default_factory=utcnow)
    retrieval_status: RetrievalStatus = RetrievalStatus.OK
    content_hash: str | None = None
    error: str | None = None
    fetched_via: str | None = None
    elapsed_ms: int | None = None

    @property
    def ok(self) -> bool:
        return self.retrieval_status in (RetrievalStatus.OK, RetrievalStatus.BROWSER)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def effective_url(self) -> str:
        return self.final_url or self.url


class Document(DomainModel):
    """A cleaned page ready for chunking, indexing and citation."""

    id: str
    run_id: str
    source_id: str
    url: str
    canonical_url: str
    company: str | None = None
    competitor: str | None = None
    title: str | None = None
    page_type: PageType = PageType.OTHER
    content: str = ""
    content_hash: str = ""
    language: str | None = None
    char_count: int = 0
    retrieved_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive_hash(self) -> Document:
        # `object.__setattr__` throughout, not `self.x = ...`. `DomainModel` sets
        # `validate_assignment=True`, so a plain assignment re-enters this
        # validator; that terminates for a non-empty document (the second pass
        # finds a non-zero `char_count` and stops) but recurses forever on a
        # whitespace-only one, which strips to "" and so re-assigns 0 every time.
        # An empty page is an ordinary outcome of a crawl, not an edge case.
        if not self.content_hash:
            object.__setattr__(self, "content_hash", f"pending:{self.id}")
        if not self.char_count:
            object.__setattr__(self, "char_count", len(self.content or ""))
        return self


class Chunk(DomainModel):
    """One retrievable piece of a document, with its provenance attached.

    Metadata travels on the chunk rather than being looked up from its parent,
    because a chunk retrieved on its own has to answer "which company, which
    page, which source" without a second query. A chunk that cannot answer that
    cannot be cited, and an uncitable chunk in an analyst's context is a
    hallucination waiting to happen.
    """

    id: str
    document_id: str
    run_id: str
    source_id: str
    url: str
    text: str
    ordinal: int = 0
    company: str | None = None
    competitor: str | None = None
    page_type: PageType = PageType.OTHER
    title: str | None = None
    heading: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def entity(self) -> str | None:
        """Whichever company this chunk is about, target or competitor."""
        return self.competitor or self.company


class RetrievedChunk(Chunk):
    """A chunk as returned by retrieval, with its score and provenance trail.

    The kept list of ``matched_by`` names the metadata filters that admitted
    this chunk, so a surprising result can be traced to the filter that allowed
    it rather than guessed at.
    """

    score: float = 0.0
    matched_by: list[str] = Field(default_factory=list)
    index_id: str | None = None
