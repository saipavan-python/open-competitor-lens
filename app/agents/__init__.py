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
from app.agents.competitor_agent import (
    VALIDATION_SIGNALS,
    CandidateSignals,
    CompetitorAgent,
    CompetitorSelection,
    ValidationOutcome,
    Validator,
)
from app.agents.market_agent import MarketAnalyst
from app.agents.marketing_agent import MarketingAnalyst
from app.agents.planner import PlannerAgent, PlannerInput
from app.agents.pricing_agent import PricingAnalyst
from app.agents.product_agent import ProductAnalyst
from app.agents.review_agent import ReviewAnalyst
from app.agents.seo_agent import SeoAnalyst
from app.agents.social_agent import SocialAnalyst
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
    "Agent",
    "AgentContext",
    "AgentResult",
    "AgentStatus",
    "CandidateSignals",
    "CategoryAnalyst",
    "CompanyAgent",
    "CompanyCandidate",
    "CompanyResolution",
    "CompetitorAgent",
    "CompetitorSelection",
    "MarketingAnalyst",
    "MarketAnalyst",
    "PlannerAgent",
    "PlannerInput",
    "PricingAnalyst",
    "ProductAnalyst",
    "ReviewAnalyst",
    "SeoAnalyst",
    "SocialAnalyst",
    "TechnologyAnalyst",
    "VALIDATION_SIGNALS",
    "ValidationOutcome",
    "Validator",
    "WebsiteAnalyst",
    "analyst_for",
    "name_similarity",
]
