"""Research run, question, and stored-request tables."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import ProgressStage, RunStatus
from app.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    json_column,
    json_dict_column,
    utcnow,
)


class ResearchRunModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One research run and its lifecycle."""

    __tablename__ = "research_runs"

    company_id: Mapped[str | None] = mapped_column(
        sa.String(36), sa.ForeignKey("companies.id", ondelete="SET NULL"), index=True
    )
    company_name: Mapped[str] = mapped_column(sa.String(512), nullable=False, index=True)
    website: Mapped[str | None] = mapped_column(sa.String(2048))
    industry: Mapped[str | None] = mapped_column(sa.String(255))
    country: Mapped[str | None] = mapped_column(sa.String(128))
    status: Mapped[RunStatus] = mapped_column(
        sa.String(32), default=RunStatus.QUEUED, nullable=False, index=True
    )
    stage: Mapped[ProgressStage] = mapped_column(
        sa.String(32), default=ProgressStage.INITIALIZING, nullable=False
    )
    progress: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)
    current_step: Mapped[str | None] = mapped_column(sa.String(255))
    request: Mapped[dict] = json_dict_column()
    research_goal: Mapped[str | None] = mapped_column(sa.Text)
    report_language: Mapped[str] = mapped_column(sa.String(8), default="en", nullable=False)
    research_depth: Mapped[str] = mapped_column(sa.String(16), default="standard", nullable=False)
    enable_graph_rag: Mapped[bool] = mapped_column(sa.Boolean, default=False, nullable=False)
    warnings: Mapped[list[str]] = json_column()
    failed_steps: Mapped[list[str]] = json_column()
    completed_steps: Mapped[list[str]] = json_column()
    errors: Mapped[list] = json_column()
    started_at: Mapped[object | None] = mapped_column(sa.DateTime(timezone=True))
    completed_at: Mapped[object | None] = mapped_column(sa.DateTime(timezone=True))
    providers: Mapped[dict] = json_dict_column()

    company: Mapped[CompanyModel | None] = relationship(  # noqa: F821
        back_populates="runs"
    )
    competitors: Mapped[list[CompetitorModel]] = relationship(  # noqa: F821
        back_populates="run", cascade="all, delete-orphan"
    )
    questions: Mapped[list[ResearchQuestionModel]] = relationship(  # noqa: F821
        back_populates="run", cascade="all, delete-orphan"
    )
    sources: Mapped[list[SourceModel]] = relationship(  # noqa: F821
        back_populates="run", cascade="all, delete-orphan"
    )

    def mark_started(self) -> None:
        self.status = RunStatus.RUNNING
        self.started_at = utcnow()

    def mark_completed(self) -> None:
        self.status = RunStatus.COMPLETED
        self.progress = 100
        self.completed_at = utcnow()


class ResearchQuestionModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A planned question and how well research answered it."""

    __tablename__ = "research_questions"

    run_id: Mapped[str] = mapped_column(
        sa.String(36), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), index=True
    )
    question_id: Mapped[str] = mapped_column(sa.String(64), nullable=False, index=True)
    question: Mapped[str] = mapped_column(sa.Text, nullable=False)
    category: Mapped[str] = mapped_column(sa.String(64), nullable=False, index=True)
    priority: Mapped[str] = mapped_column(sa.String(16), default="medium", nullable=False)
    competitor: Mapped[str | None] = mapped_column(sa.String(512))
    status: Mapped[str] = mapped_column(sa.String(32), default="pending", nullable=False)
    required_sources: Mapped[list[str]] = json_column()
    answer_summary: Mapped[str | None] = mapped_column(sa.Text)
    evidence_ids: Mapped[list[str]] = json_column()

    run: Mapped[ResearchRunModel] = relationship(back_populates="questions")  # noqa: F821
