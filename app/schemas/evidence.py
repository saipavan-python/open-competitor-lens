"""Evidence layer: the traceability backbone of the whole system.

Nothing in a report may be asserted without pointing at a :class:`Source` the
run actually retrieved. These validators make that a schema-level guarantee
rather than a prompt-level request.
"""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator

from app.core.constants import (
    NON_PRIMARY_SOURCE_TYPES,
    SOURCE_AUTHORITY,
    ClaimType,
    Confidence,
    PageType,
    RetrievalStatus,
    SourceType,
)
from app.schemas.common import DomainModel, utcnow


def _host(url: str) -> str | None:
    try:
        host = urlparse(url).hostname
    except ValueError:
        return None
    return host.lower().removeprefix("www.") if host else None


class Source(DomainModel):
    """A page the system actually retrieved and stored.

    ``text`` holds the extracted body. It exists so
    :mod:`app.services.evidence_service` can verify that a quoted passage really
    appears in the retrieved page rather than being produced from model memory.
    """

    id: str
    url: str
    canonical_url: str | None = None
    title: str | None = None
    source_type: SourceType | None = None
    domain: str | None = None
    company: str | None = None
    competitor: str | None = None
    retrieved_at: datetime = Field(default_factory=utcnow)
    published_at: datetime | None = None
    updated_at: datetime | None = None
    content_hash: str | None = None
    retrieval_status: RetrievalStatus = RetrievalStatus.OK
    page_type: PageType = PageType.OTHER
    language: str | None = None
    text: str | None = None
    discovery_query: str | None = None
    notes: str | None = None

    @field_validator("source_type", mode="before")
    @classmethod
    def _default_source_type(cls, value: object) -> object:
        """An unclassified source is ``unknown``, never optimistically primary."""
        return SourceType.UNKNOWN if value is None else value

    @field_validator("url")
    @classmethod
    def _url_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Source.url is required")
        return value.strip()

    @field_validator("domain", mode="before")
    @classmethod
    def _derive_domain(cls, value: object, info: object) -> object:
        if value:
            return value
        url = getattr(info, "data", {}).get("url") if info else None
        return _host(url) if isinstance(url, str) else None

    @property
    def authority(self) -> int:
        """Authority weight used to resolve conflicts between sources."""
        return SOURCE_AUTHORITY.get(self.source_type, 0)

    @property
    def is_primary(self) -> bool:
        return self.source_type not in NON_PRIMARY_SOURCE_TYPES

    def supports(self, confidence: Confidence) -> bool:
        """Whether this source is strong enough to back ``confidence``."""
        if confidence is Confidence.HIGH:
            return self.is_primary
        return True


class Evidence(DomainModel):
    """One source-backed fact.

    ``claim_key`` groups several evidences that speak to the same underlying
    fact, which is how source conflicts are detected without discarding either
    side.
    """

    id: str
    claim: str
    source_id: str
    source_url: str
    evidence_text: str | None = None
    claim_type: ClaimType
    confidence: Confidence
    retrieved_at: datetime = Field(default_factory=utcnow)
    claim_key: str | None = None
    company: str | None = None
    competitor: str | None = None
    category: str | None = None
    source_type: SourceType | None = None
    published_at: datetime | None = None
    conflicts_with: list[str] = Field(default_factory=list)
    superseded_by: str | None = None

    @model_validator(mode="after")
    def _enforce_traceability(self) -> Evidence:
        if self.confidence is Confidence.HIGH and not self.source_id.strip():
            raise ValueError(
                "A high-confidence claim must cite a source_id; "
                "uncited claims are not permitted to be high confidence"
            )
        if self.claim_type is ClaimType.INFERRED and self.confidence is Confidence.HIGH:
            raise ValueError(
                "An inference cannot carry high confidence; the source does not state it"
            )
        if self.claim_type is ClaimType.UNKNOWN and self.confidence is Confidence.HIGH:
            raise ValueError("An unknown claim cannot carry high confidence")
        return self

    @property
    def is_actionable(self) -> bool:
        """Whether this evidence can support a finding in the report body."""
        return self.claim_type in (ClaimType.OBSERVED, ClaimType.REPORTED)


class EvidenceBundle(DomainModel):
    """A claim plus every piece of evidence gathered for it."""

    claim: str
    evidence: list[Evidence] = Field(default_factory=list)
    conflicts: bool = False
    best_confidence: Confidence = Confidence.LOW

    @property
    def evidence_ids(self) -> list[str]:
        return [item.id for item in self.evidence]


class Citation(DomainModel):
    """A resolvable link from a report claim back to a public URL."""

    claim_id: str
    source_id: str
    url: str
    title: str | None = None
    source_type: SourceType | None = None
    retrieved_at: datetime | None = None
    quote: str | None = None
