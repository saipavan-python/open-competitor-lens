"""Analysis tables: products, pricing, social, reviews, technology, market."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import Confidence
from app.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    json_column,
    utcnow,
)


def _run_fk() -> Mapped[str]:
    return mapped_column(
        sa.String(36), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), index=True
    )


def _company() -> Mapped[str]:
    return mapped_column(sa.String(512), nullable=False, index=True)


class ProductModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "products"
    __table_args__ = (sa.UniqueConstraint("run_id", "company", "name", name="uq_product"),)

    run_id: Mapped[str] = _run_fk()
    company: Mapped[str] = _company()
    name: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text)
    features: Mapped[list[str]] = json_column()
    target_segments: Mapped[list[str]] = json_column()
    use_cases: Mapped[list[str]] = json_column()
    differentiators: Mapped[list[str]] = json_column()
    deployment_model: Mapped[str | None] = mapped_column(sa.String(255))
    business_model: Mapped[str | None] = mapped_column(sa.String(255))
    url: Mapped[str | None] = mapped_column(sa.String(4096))
    evidence_ids: Mapped[list[str]] = json_column()


class PricingPlanModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "pricing_plans"

    run_id: Mapped[str] = _run_fk()
    company: Mapped[str] = _company()
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    price: Mapped[float | None] = mapped_column(sa.Float)
    currency: Mapped[str | None] = mapped_column(sa.String(8))
    billing_period: Mapped[str | None] = mapped_column(sa.String(32))
    price_display: Mapped[str | None] = mapped_column(sa.String(255))
    features: Mapped[list[str]] = json_column()
    limitations: Mapped[list[str]] = json_column()
    is_free: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    is_contact_sales: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    pricing_page_url: Mapped[str | None] = mapped_column(sa.String(4096))
    evidence_ids: Mapped[list[str]] = json_column()


class ReviewModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "reviews"

    run_id: Mapped[str] = _run_fk()
    company: Mapped[str] = _company()
    platform: Mapped[str | None] = mapped_column(sa.String(128), index=True)
    theme: Mapped[str] = mapped_column(sa.String(512), nullable=False, index=True)
    sentiment: Mapped[str] = mapped_column(sa.String(16), nullable=False, index=True)
    mention_count: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)
    quote: Mapped[str | None] = mapped_column(sa.Text)
    source_url: Mapped[str | None] = mapped_column(sa.String(4096))
    evidence_ids: Mapped[list[str]] = json_column()


class SocialProfileModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "social_profiles"
    __table_args__ = (
        sa.UniqueConstraint("run_id", "company", "platform", name="uq_social_profile"),
    )

    run_id: Mapped[str] = _run_fk()
    company: Mapped[str] = _company()
    platform: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    profile_url: Mapped[str] = mapped_column(sa.String(2048), nullable=False)
    handle: Mapped[str | None] = mapped_column(sa.String(255))
    content_themes: Mapped[list[str]] = json_column()
    visible_activity: Mapped[str | None] = mapped_column(sa.Text)
    public_metrics: Mapped[dict] = mapped_column(sa.JSON, default=dict, nullable=False)
    observations: Mapped[list[str]] = json_column()
    evidence_ids: Mapped[list[str]] = json_column()


class TechnologySignalModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "technology_signals"
    __table_args__ = (
        sa.UniqueConstraint("run_id", "company", "technology", "category", name="uq_tech"),
    )

    run_id: Mapped[str] = _run_fk()
    company: Mapped[str] = _company()
    technology: Mapped[str] = mapped_column(sa.String(255), nullable=False, index=True)
    category: Mapped[str] = mapped_column(sa.String(64), nullable=False, index=True)
    confidence: Mapped[Confidence] = mapped_column(sa.String(16), default=Confidence.LOW)
    basis: Mapped[str | None] = mapped_column(sa.Text)
    source_id: Mapped[str | None] = mapped_column(sa.String(36))
    source_url: Mapped[str | None] = mapped_column(sa.String(4096))
    observed_at: Mapped[object | None] = mapped_column(sa.DateTime(timezone=True))


class MarketInsightModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "market_insights"

    run_id: Mapped[str] = _run_fk()
    observation: Mapped[str] = mapped_column(sa.Text, nullable=False)
    interpretation: Mapped[str | None] = mapped_column(sa.Text)
    potential_opportunity: Mapped[str | None] = mapped_column(sa.Text)
    kind: Mapped[str] = mapped_column(sa.String(32), default="insight", index=True)
    companies: Mapped[list[str]] = json_column()
    evidence_ids: Mapped[list[str]] = json_column()
    confidence: Mapped[Confidence] = mapped_column(sa.String(16), default=Confidence.LOW)
    detected_at: Mapped[object] = mapped_column(sa.DateTime(timezone=True), default=utcnow)


class OpportunityModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "opportunities"

    run_id: Mapped[str] = _run_fk()
    opportunity: Mapped[str] = mapped_column(sa.Text, nullable=False)
    reasoning: Mapped[str] = mapped_column(sa.Text, nullable=False)
    category: Mapped[str | None] = mapped_column(sa.String(64), index=True)
    confidence: Mapped[Confidence] = mapped_column(sa.String(16), default=Confidence.LOW)
    affected_competitors: Mapped[list[str]] = json_column()
    evidence_ids: Mapped[list[str]] = json_column()
    kind: Mapped[str] = mapped_column(sa.String(16), default="opportunity")


class ReportModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A stored report. Multiple rows per run are allowed - one per repair round."""

    __tablename__ = "reports"

    run_id: Mapped[str] = mapped_column(
        sa.String(36), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), index=True
    )
    report_json: Mapped[dict] = mapped_column(sa.JSON, default=dict, nullable=False)
    report_markdown: Mapped[str | None] = mapped_column(sa.Text)
    report_html: Mapped[str | None] = mapped_column(sa.Text)
    is_latest: Mapped[bool] = mapped_column(sa.Boolean, default=True, index=True)
    repair_round: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)
