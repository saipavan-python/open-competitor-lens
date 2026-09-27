"""Research request, plan, question and run-status schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from app.core.constants import AnalysisCategory, ProgressStage, RunStatus
from app.schemas.common import DomainModel, utcnow

ResearchDepth = Literal["quick", "standard", "deep"]
QuestionStatus = Literal["pending", "answered", "partially_answered", "unanswered"]


class ReportConfig(DomainModel):
    """Per-run overrides for scope and optional layers (spec section 95)."""

    report_language: str = "en"
    competitor_count: int = Field(default=5, ge=1, le=50)
    enable_graph_rag: bool = False
    enable_browser: bool = True
    max_pages_per_competitor: int = Field(default=40, ge=1, le=500)
    research_depth: ResearchDepth = "standard"


class StartAnalysisRequest(DomainModel):
    """The ``POST /api/v1/analysis`` body (spec section 53)."""

    company_name: str
    website: str | None = None
    industry: str | None = None
    country: str | None = None
    competitor_count: int = Field(default=5, ge=1, le=50)
    research_goal: str = (
        "Compare products, pricing, marketing, SEO, social presence, reviews, "
        "technology signals and market gaps"
    )
    report_language: str = "en"
    research_depth: ResearchDepth = "standard"
    enable_graph_rag: bool = False
    enable_browser: bool = True
    max_pages_per_competitor: int = Field(default=40, ge=1, le=500)
    categories: list[AnalysisCategory] = Field(default_factory=list)

    @field_validator("company_name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("company_name must not be blank")
        return value.strip()

    @field_validator("website")
    @classmethod
    def _website_scheme(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        candidate = value.strip()
        if not candidate.startswith(("http://", "https://")):
            candidate = f"https://{candidate}"
        return candidate

    def to_report_config(self) -> ReportConfig:
        return ReportConfig(
            report_language=self.report_language,
            competitor_count=self.competitor_count,
            enable_graph_rag=self.enable_graph_rag,
            enable_browser=self.enable_browser,
            max_pages_per_competitor=self.max_pages_per_competitor,
            research_depth=self.research_depth,
        )


class ResearchQuestion(DomainModel):
    """One planned question, tracked so coverage can be reported honestly."""

    id: str
    category: AnalysisCategory
    question: str
    priority: Literal["high", "medium", "low"] = "medium"
    competitor: str | None = None
    required_sources: list[str] = Field(default_factory=list)
    status: QuestionStatus = "pending"
    answer_summary: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class SearchStrategy(DomainModel):
    """How the plan intends to search, kept separate from the questions."""

    queries_per_competitor: int = Field(default=6, ge=1, le=30)
    preferred_source_types: list[str] = Field(default_factory=list)
    use_browser_fallback: bool = True
    use_graph_rag: bool = False
    notes: str | None = None


class ResearchPlan(DomainModel):
    """The planner's output (spec section 9)."""

    research_scope: list[str] = Field(default_factory=list)
    research_questions: list[ResearchQuestion] = Field(default_factory=list)
    search_strategy: SearchStrategy = Field(default_factory=SearchStrategy)
    required_sources: list[str] = Field(default_factory=list)
    priority_levels: dict[str, str] = Field(default_factory=dict)
    excluded_categories: list[str] = Field(default_factory=list)

    @property
    def question_count(self) -> int:
        return len(self.research_questions)


class ResearchCoverage(DomainModel):
    """Question coverage, reported rather than forced to 100% (spec section 80)."""

    questions_total: int = 0
    questions_answered: int = 0
    questions_partially_answered: int = 0
    questions_unanswered: int = 0

    @property
    def answered_ratio(self) -> float:
        if not self.questions_total:
            return 0.0
        return round(self.questions_answered / self.questions_total, 3)


class RunStatusResponse(DomainModel):
    """The ``GET /api/v1/analysis/{run_id}/status`` body (spec section 54)."""

    run_id: str
    status: RunStatus
    current_step: str = ProgressStage.INITIALIZING.value
    progress: int = Field(default=0, ge=0, le=100)
    completed_steps: list[str] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ResearchRunView(DomainModel):
    """A stored run as returned by ``GET /api/v1/analysis/{run_id}``."""

    run_id: str
    company_name: str
    website: str | None = None
    industry: str | None = None
    country: str | None = None
    status: RunStatus
    stage: ProgressStage = ProgressStage.INITIALIZING
    progress: int = 0
    request: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None


class TargetResearchRequest(DomainModel):
    """Re-run only part of a finished run (spec section 105)."""

    run_id: str
    competitors: list[str] | None = None
    categories: list[AnalysisCategory] | None = None
