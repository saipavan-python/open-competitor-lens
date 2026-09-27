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
        if not self.content_hash:
            object.__setattr__(self, "content_hash", f"pending:{self.id}")
        if not self.char_count:
            self.char_count = len(self.content)
        return self
