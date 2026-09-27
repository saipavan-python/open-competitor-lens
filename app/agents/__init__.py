"""Agents: the steps of the analysis, each one a boundary that can refuse.

An agent is a named unit of work with a declared output schema and a budget. The
base class holds what every agent needs — the run's context, a status that can
express partial success, and a result that carries a payload *or* a refusal — so
that no individual agent has to reinvent the answer to "what do I return when I
could not do the job?".

That question is the reason this layer exists in this shape. In a system whose
whole purpose is not to make things up, "I could not determine this" has to be
as easy to return as an answer, or it will get replaced by a confident one. An
agent that cannot report a failure structurally will report a success with
hollow contents instead.
"""

from app.agents.analysis_base import CategoryAnalyst
from app.agents.base import Agent, AgentContext, AgentResult, AgentStatus
from app.agents.company_agent import (
    CompanyAgent,
    CompanyCandidate,
    CompanyResolution,
    name_similarity,
)
from app.agents.comparison import ComparisonAgent
from app.agents.competitor_agent import (
    VALIDATION_SIGNALS,
    CandidateSignals,
    CompetitorAgent,
    CompetitorSelection,
    ValidationOutcome,
    Validator,
)
from app.agents.evidence_agent import EvidenceAgent
from app.agents.fact_checker import FactCheckerAgent
from app.agents.market_agent import MarketAnalyst
from app.agents.marketing_agent import MarketingAnalyst
from app.agents.opportunity import OPPORTUNITY_FIELDS, OpportunityAgent
from app.agents.planner import PlannerAgent, PlannerInput
from app.agents.pricing_agent import PricingAnalyst
from app.agents.product_agent import ProductAnalyst
from app.agents.report_agent import ReportAgent
from app.agents.review_agent import ReviewAnalyst
from app.agents.seo_agent import SeoAnalyst
from app.agents.social_agent import SocialAnalyst
from app.agents.supervisor import Supervisor
from app.agents.technology_agent import TechnologyAnalyst
from app.agents.website_agent import WebsiteAnalyst

#: The nine analysts, in report order. Task 11 builds the comparison from these
#: and Task 12 the graph, so the order is part of the contract rather than an
#: implementation detail: a report whose pricing section precedes its product
#: section reads as though the analysis had no structure.
ANALYSTS: tuple[type, ...] = (
    WebsiteAnalyst,
    ProductAnalyst,
    PricingAnalyst,
    MarketingAnalyst,
    SeoAnalyst,
    SocialAnalyst,
    ReviewAnalyst,
    TechnologyAnalyst,
    MarketAnalyst,
)

#: Analyst classes by category, for the graph to look one up by name.
ANALYSTS_BY_CATEGORY: dict[str, type] = {analyst.category: analyst for analyst in ANALYSTS}

#: The steps that turn the analysts' payloads into the report, in the order the
#: graph runs them. Kept beside the analysts because the sequence is the contract:
#: a comparison built before the analysts have run compares nothing, and a report
#: assembled before the fact check would print claims nobody has checked.
ASSEMBLY_AGENTS: tuple[type, ...] = (
    ComparisonAgent,
    OpportunityAgent,
    EvidenceAgent,
    FactCheckerAgent,
    ReportAgent,
)


def analyst_for(category: str) -> type:
    """The analyst class for a category, or a clear error listing the real ones."""
    try:
        return ANALYSTS_BY_CATEGORY[category]
    except KeyError as exc:
        raise KeyError(
            f"Unknown analysis category {category!r}. "
            f"Known categories: {', '.join(sorted(ANALYSTS_BY_CATEGORY))}."
        ) from exc


__all__ = [
    "ANALYSTS",
    "ANALYSTS_BY_CATEGORY",
    "ASSEMBLY_AGENTS",
    "OPPORTUNITY_FIELDS",
    "Agent",
    "AgentContext",
    "AgentResult",
    "AgentStatus",
    "CandidateSignals",
    "CategoryAnalyst",
    "ComparisonAgent",
    "CompanyAgent",
    "CompanyCandidate",
    "CompanyResolution",
    "CompetitorAgent",
    "CompetitorSelection",
    "EvidenceAgent",
    "FactCheckerAgent",
    "MarketingAnalyst",
    "MarketAnalyst",
    "OpportunityAgent",
    "PlannerAgent",
    "PlannerInput",
    "PricingAnalyst",
    "ProductAnalyst",
    "ReportAgent",
    "ReviewAnalyst",
    "SeoAnalyst",
    "SocialAnalyst",
    "Supervisor",
    "TechnologyAnalyst",
    "VALIDATION_SIGNALS",
    "ValidationOutcome",
    "Validator",
    "WebsiteAnalyst",
    "analyst_for",
    "name_similarity",
]
