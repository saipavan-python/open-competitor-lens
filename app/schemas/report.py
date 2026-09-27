"""Report schemas: the final deliverable and its evidence-quality section."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.core.constants import RunStatus
from app.schemas.common import DomainModel
from app.schemas.company import CompanyProfile
from app.schemas.comparison import ComparisonMatrix, CompetitorCoverage, CrossCompetitorPattern
from app.schemas.competitor import ValidatedCompetitor
from app.schemas.evidence import Source
from app.schemas.finding import ClaimVerdictRecord, Finding, QualityReport
from app.schemas.market import MarketAnalysis, Opportunity, Risk
from app.schemas.marketing import MarketingAnalysis
from app.schemas.pricing import PricingAnalysis
from app.schemas.products import ProductAnalysis
from app.schemas.research import ResearchCoverage
from app.schemas.reviews import ReviewAnalysis
from app.schemas.seo import SeoAnalysis
from app.schemas.social import SocialAnalysis
from app.schemas.technology import TechnologyAnalysis

#: The 19 sections of the report, in order (spec section 62).
REPORT_SECTIONS: tuple[str, ...] = (
    "executive_summary",
    "target_company_overview",
    "competitor_landscape",
    "competitor_profiles",
    "product_comparison",
    "feature_comparison",
    "pricing_comparison",
    "marketing_analysis",
    "seo_analysis",
    "social_analysis",
    "review_analysis",
    "technology_analysis",
    "market_analysis",
    "cross_competitor_patterns",
    "observed_gaps",
    "potential_opportunities",
    "risks_and_constraints",
    "evidence_quality",
    "sources",
)

SECTION_TITLES: dict[str, str] = {
    "executive_summary": "1. Executive Summary",
    "target_company_overview": "2. Target Company Overview",
    "competitor_landscape": "3. Competitor Landscape",
    "competitor_profiles": "4. Competitor Profiles",
    "product_comparison": "5. Product Comparison",
    "feature_comparison": "6. Feature Comparison",
    "pricing_comparison": "7. Pricing Comparison",
    "marketing_analysis": "8. Marketing Analysis",
    "seo_analysis": "9. SEO Analysis",
    "social_analysis": "10. Social Media Analysis",
    "review_analysis": "11. Customer Review Analysis",
    "technology_analysis": "12. Technology Signals",
    "market_analysis": "13. Market Analysis",
    "cross_competitor_patterns": "14. Cross-Competitor Patterns",
    "observed_gaps": "15. Observed Gaps",
    "potential_opportunities": "16. Potential Opportunities",
    "risks_and_constraints": "17. Risks / Constraints",
    "evidence_quality": "18. Evidence Quality",
    "sources": "19. Sources",
}


class ReportMetadata(DomainModel):
    """Provenance of the report itself."""

    run_id: str
    company: str
    website: str | None = None
    industry: str | None = None
    country: str | None = None
    research_started_at: datetime
    research_completed_at: datetime
    research_goal: str | None = None
    report_language: str = "en"
    research_depth: str = "standard"
    model_profiles: dict[str, str] = Field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return round((self.research_completed_at - self.research_started_at).total_seconds(), 2)


class EvidenceQuality(DomainModel):
    """How much of the report is actually backed (spec sections 80-82)."""

    major_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    conflicting_claims: int = 0
    stale_claims: int = 0
    total_sources: int = 0
    primary_sources: int = 0
    secondary_sources: int = 0
    snippet_only_sources: int = 0
    coverage: ResearchCoverage = Field(default_factory=ResearchCoverage)
    competitor_coverage: list[CompetitorCoverage] = Field(default_factory=list)
    confidence_distribution: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @property
    def supported_ratio(self) -> float:
        if not self.major_claims:
            return 0.0
        return round(self.supported_claims / self.major_claims, 3)


class ReportBundle(DomainModel):
    """The complete machine-readable report (spec section 64).

    This is the object the API serves, the CLI writes to
    ``data/reports/<run_id>.json`` and the Markdown/HTML renderers consume.
    """

    metadata: ReportMetadata
    target_company: CompanyProfile | None = None
    competitors: list[ValidatedCompetitor] = Field(default_factory=list)
    competitor_profiles: dict[str, dict[str, object]] = Field(default_factory=dict)
    product_comparison: ComparisonMatrix | None = None
    pricing_comparison: dict[str, PricingAnalysis] = Field(default_factory=dict)
    marketing_analysis: dict[str, MarketingAnalysis] = Field(default_factory=dict)
    seo_analysis: dict[str, SeoAnalysis] = Field(default_factory=dict)
    social_analysis: dict[str, SocialAnalysis] = Field(default_factory=dict)
    review_analysis: dict[str, ReviewAnalysis] = Field(default_factory=dict)
    technology_analysis: dict[str, TechnologyAnalysis] = Field(default_factory=dict)
    market_analysis: MarketAnalysis | None = None
    product_analysis: dict[str, ProductAnalysis] = Field(default_factory=dict)
    website_analysis: dict[str, dict[str, object]] = Field(default_factory=dict)
    cross_competitor_patterns: list[CrossCompetitorPattern] = Field(default_factory=list)
    observed_gaps: list[str] = Field(default_factory=list)
    opportunities: list[Opportunity] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    insights: list[Finding] = Field(default_factory=list)
    evidence_quality: EvidenceQuality = Field(default_factory=EvidenceQuality)
    quality_report: QualityReport | None = None
    claim_verdicts: list[ClaimVerdictRecord] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    evidence: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    run_status: RunStatus = RunStatus.COMPLETED
    progress: int = 100
    report_markdown: str | None = None
    report_html: str | None = None
