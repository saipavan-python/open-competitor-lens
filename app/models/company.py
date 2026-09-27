"""Company and competitor tables."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, json_column


class CompanyModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A company known to the system, target or competitor."""

    __tablename__ = "companies"

    name: Mapped[str] = mapped_column(sa.String(512), nullable=False, index=True)
    canonical_name: Mapped[str | None] = mapped_column(sa.String(512))
    website: Mapped[str | None] = mapped_column(sa.String(2048))
    domain: Mapped[str | None] = mapped_column(sa.String(255), index=True)
    industry: Mapped[str | None] = mapped_column(sa.String(255))
    country: Mapped[str | None] = mapped_column(sa.String(128))
    description: Mapped[str | None] = mapped_column(sa.Text)
    identity_key: Mapped[str] = mapped_column(sa.String(255), unique=True, index=True)
    aliases: Mapped[list[str]] = json_column()
    domains: Mapped[list[str]] = json_column()

    runs: Mapped[list[ResearchRunModel]] = relationship(  # noqa: F821
        back_populates="company", cascade="all, delete-orphan"
    )


class CompetitorModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A validated competitor of a specific run."""

    __tablename__ = "competitors"
    __table_args__ = (sa.UniqueConstraint("run_id", "identity_key", name="uq_competitor_run"),
                      )

    run_id: Mapped[str] = mapped_column(
        sa.String(36), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(512), nullable=False, index=True)
    canonical_name: Mapped[str | None] = mapped_column(sa.String(512))
    website: Mapped[str | None] = mapped_column(sa.String(2048))
    domain: Mapped[str | None] = mapped_column(sa.String(255))
    industry: Mapped[str | None] = mapped_column(sa.String(255))
    country: Mapped[str | None] = mapped_column(sa.String(128))
    description: Mapped[str | None] = mapped_column(sa.Text)
    identity_key: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    validated: Mapped[bool] = mapped_column(sa.Boolean, default=False, index=True)
    relevance_reason: Mapped[str | None] = mapped_column(sa.Text)
    is_target: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    evidence_ids: Mapped[list[str]] = json_column()
    discovery_queries: Mapped[list[str]] = json_column()

    run: Mapped[ResearchRunModel] = relationship(  # noqa: F821
        back_populates="competitors"
    )
