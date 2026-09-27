"""Market schemas.

The three fields on :class:`MarketInsight` - observation, interpretation,
potential opportunity - are the spec's requirement that market reasoning never
collapses into a single undifferentiated assertion.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from app.core.constants import Confidence
from app.schemas.common import DomainModel


class MarketInsight(DomainModel):
    """Observed market evidence, our reading of it, and what it might allow."""

    observation: str
    interpretation: str | None = None
    potential_opportunity: str | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    companies: list[str] = Field(default_factory=list)


class MarketTrend(DomainModel):
    """A directional pattern observed across several sources."""

    trend: str
    supporting_observations: list[str] = Field(default_factory=list)
    affected_companies: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW


class MarketAnalysis(DomainModel):
    """The market agent's output for a whole run."""

    market: str | None = None
    country: str | None = None
    industry: str | None = None
    trends: list[MarketTrend] = Field(default_factory=list)
    insights: list[MarketInsight] = Field(default_factory=list)
    customer_needs: list[str] = Field(default_factory=list)
    emerging_features: list[str] = Field(default_factory=list)
    common_business_models: list[str] = Field(default_factory=list)
    pricing_patterns: list[str] = Field(default_factory=list)
    underserved_segments: list[str] = Field(default_factory=list)
    new_entrants: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW


class Opportunity(DomainModel):
    """A gap-derived opportunity.

    ``evidence`` must be non-empty: an opportunity with no evidence behind it is
    a guess, and the schema refuses to represent one. The rule is enforced here
    rather than left to the agent that builds them, because an opportunity is
    advice — the one part of a report a reader is most likely to act on, and the
    part where an unsupported suggestion costs the most.
    """

    opportunity: str
    reasoning: str
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    affected_competitors: list[str] = Field(default_factory=list)
    category: str | None = None
    effort: str | None = None
    impact: str | None = None

    @model_validator(mode="after")
    def _require_evidence(self) -> Opportunity:
        """Refuse an opportunity with nothing behind it.

        A model validator rather than a field one, and the reason is the default:
        a field validator does not run when a field is left to its
        ``default_factory``, so the default of an empty list — the exact case this
        exists to catch — would have passed straight through.
        """
        if not [item for item in self.evidence if item and item.strip()]:
            raise ValueError(
                "an opportunity must cite at least one evidence id; a suggestion "
                "nothing in the research supports is a guess presented as advice"
            )
        return self

    def __len__(self) -> int:  # pragma: no cover - convenience only
        return 1


class Risk(DomainModel):
    """A risk or constraint surfaced by the evidence."""

    risk: str
    reasoning: str
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    severity: str | None = None
    affected_competitors: list[str] = Field(default_factory=list)
