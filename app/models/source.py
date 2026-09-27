"""Source, document and evidence tables - the traceability backbone."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import ClaimType, Confidence, PageType, RetrievalStatus, SourceType
from app.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    json_column,
    utcnow,
)


class SourceModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A page the run retrieved.

    ``content_hash`` is unique per run: refetching an unchanged page updates the
    existing row rather than creating a second copy of the same evidence.
    """

    __tablename__ = "sources"
    __table_args__ = (
        sa.UniqueConstraint("run_id", "canonical_url", name="uq_source_run_canonical"),
        sa.UniqueConstraint("run_id", "content_hash", name="uq_source_run_hash"),
        sa.Index("ix_source_run_competitor", "run_id", "competitor"),
    )

    run_id: Mapped[str] = mapped_column(
        sa.String(36), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), index=True
    )
    url: Mapped[str] = mapped_column(sa.String(4096), nullable=False)
    canonical_url: Mapped[str] = mapped_column(sa.String(4096), nullable=False)
    title: Mapped[str | None] = mapped_column(sa.String(1024))
    source_type: Mapped[SourceType] = mapped_column(
        sa.String(32), default=SourceType.UNKNOWN, nullable=False, index=True
    )
    domain: Mapped[str | None] = mapped_column(sa.String(255), index=True)
    company: Mapped[str | None] = mapped_column(sa.String(512), index=True)
    competitor: Mapped[str | None] = mapped_column(sa.String(512), index=True)
    retrieved_at: Mapped[object] = mapped_column(
        sa.DateTime(timezone=True), default=utcnow, nullable=False
    )
    published_at: Mapped[object | None] = mapped_column(sa.DateTime(timezone=True))
    updated_at_page: Mapped[object | None] = mapped_column(sa.DateTime(timezone=True))
    content_hash: Mapped[str | None] = mapped_column(sa.String(128))
    retrieval_status: Mapped[RetrievalStatus] = mapped_column(
        sa.String(32), default=RetrievalStatus.OK, nullable=False, index=True
    )
    page_type: Mapped[PageType] = mapped_column(
        sa.String(32), default=PageType.OTHER, nullable=False, index=True
    )
    language: Mapped[str | None] = mapped_column(sa.String(16))
    text: Mapped[str | None] = mapped_column(sa.Text)
    discovery_query: Mapped[str | None] = mapped_column(sa.String(1024))
    error: Mapped[str | None] = mapped_column(sa.Text)
    metadata_json: Mapped[dict] = mapped_column(sa.JSON, default=dict, nullable=False)

    run: Mapped[ResearchRunModel] = relationship(back_populates="sources")  # noqa: F821
    documents: Mapped[list[DocumentModel]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )
    evidence: Mapped[list[EvidenceModel]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class DocumentModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A cleaned, chunkable page."""

    __tablename__ = "documents"
    __table_args__ = (sa.UniqueConstraint("run_id", "content_hash", name="uq_document_run_hash"),)

    run_id: Mapped[str] = mapped_column(
        sa.String(36), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), index=True
    )
    source_id: Mapped[str] = mapped_column(
        sa.String(36), sa.ForeignKey("sources.id", ondelete="CASCADE"), index=True
    )
    url: Mapped[str] = mapped_column(sa.String(4096), nullable=False)
    canonical_url: Mapped[str] = mapped_column(sa.String(4096), nullable=False)
    company: Mapped[str | None] = mapped_column(sa.String(512), index=True)
    competitor: Mapped[str | None] = mapped_column(sa.String(512), index=True)
    title: Mapped[str | None] = mapped_column(sa.String(1024))
    page_type: Mapped[PageType] = mapped_column(sa.String(32), default=PageType.OTHER)
    content: Mapped[str] = mapped_column(sa.Text, default="", nullable=False)
    content_hash: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    language: Mapped[str | None] = mapped_column(sa.String(16))
    char_count: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)
    retrieved_at: Mapped[object] = mapped_column(sa.DateTime(timezone=True), default=utcnow)
    metadata_json: Mapped[dict] = mapped_column(sa.JSON, default=dict, nullable=False)

    source: Mapped[SourceModel] = relationship(back_populates="documents")


@sa.event.listens_for(DocumentModel, "before_insert")
@sa.event.listens_for(DocumentModel, "before_update")
def _derive_document_char_count(
    _mapper: object, _connection: object, target: DocumentModel
) -> None:
    """Keep ``char_count`` in step with ``content``.

    A Python-side column default would only fire at INSERT and would go stale
    whenever the content was later corrected, so the value is derived on every
    write instead.
    """
    target.char_count = len(target.content or "")


class EvidenceModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One source-backed fact, with a foreign key to the page that proves it."""

    __tablename__ = "evidence"
    __table_args__ = (sa.Index("ix_evidence_run_claim_key", "run_id", "claim_key"),)

    run_id: Mapped[str] = mapped_column(
        sa.String(36), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), index=True
    )
    source_id: Mapped[str] = mapped_column(
        sa.String(36), sa.ForeignKey("sources.id", ondelete="CASCADE"), index=True
    )
    claim: Mapped[str] = mapped_column(sa.Text, nullable=False)
    source_url: Mapped[str] = mapped_column(sa.String(4096), nullable=False)
    evidence_text: Mapped[str | None] = mapped_column(sa.Text)
    claim_type: Mapped[ClaimType] = mapped_column(sa.String(16), nullable=False, index=True)
    confidence: Mapped[Confidence] = mapped_column(sa.String(16), nullable=False, index=True)
    claim_key: Mapped[str | None] = mapped_column(sa.String(512), index=True)
    company: Mapped[str | None] = mapped_column(sa.String(512), index=True)
    competitor: Mapped[str | None] = mapped_column(sa.String(512), index=True)
    category: Mapped[str | None] = mapped_column(sa.String(64), index=True)
    source_type: Mapped[SourceType | None] = mapped_column(sa.String(32))
    retrieved_at: Mapped[object] = mapped_column(sa.DateTime(timezone=True), default=utcnow)
    conflicts_with: Mapped[list[str]] = json_column()
    superseded_by: Mapped[str | None] = mapped_column(sa.String(36))

    source: Mapped[SourceModel] = relationship(back_populates="evidence")


class FindingModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A reportable, evidence-linked statement."""

    __tablename__ = "findings"

    run_id: Mapped[str] = mapped_column(
        sa.String(36), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), index=True
    )
    company: Mapped[str] = mapped_column(sa.String(512), nullable=False, index=True)
    category: Mapped[str] = mapped_column(sa.String(64), nullable=False, index=True)
    statement: Mapped[str] = mapped_column(sa.Text, nullable=False)
    observation: Mapped[str | None] = mapped_column(sa.Text)
    interpretation: Mapped[str | None] = mapped_column(sa.Text)
    evidence_ids: Mapped[list[str]] = json_column()
    confidence: Mapped[Confidence] = mapped_column(sa.String(16), default=Confidence.LOW)
    value: Mapped[str | None] = mapped_column(sa.Text)
    unknown: Mapped[bool] = mapped_column(sa.Boolean, default=False)
