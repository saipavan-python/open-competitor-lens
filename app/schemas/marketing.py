"""Marketing schemas.

Findings are split into ``observations`` (visible on a page) and ``inferences``
(our reading of them). The split is enforced by the field names so a reader can
never mistake a marketing interpretation for a marketing fact.
"""

from __future__ import annotations

from pydantic import Field

from app.core.constants import Confidence
from app.schemas.common import DomainModel


class MarketingSignal(DomainModel):
    """One observable marketing signal with the page that showed it."""

    signal: str
    observation: str
    interpretation: str | None = None
    source_id: str | None = None
    channel: str | None = None


class MarketingAnalysis(DomainModel):
    """Observable marketing surface for one company."""

    company: str
    brand_messaging: list[str] = Field(default_factory=list)
    value_proposition: str | None = None
    content_themes: list[str] = Field(default_factory=list)
    calls_to_action: list[str] = Field(default_factory=list)
    case_studies: list[str] = Field(default_factory=list)
    lead_magnets: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    signals: list[MarketingSignal] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
