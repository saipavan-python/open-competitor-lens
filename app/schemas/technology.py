"""Technology signal schemas.

A signal records what was observed and ``basis`` records where it was observed.
``confidence`` is the analyst's confidence in the *signal*, not an assertion that
the technology is definitively in production.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.core.constants import Confidence
from app.schemas.common import DomainModel

TechCategory = Literal[
    "frontend", "backend", "database", "cloud", "payments", "analytics",
    "ai_ml", "authentication", "monitoring", "cdn", "hosting", "other",
]


class TechnologySignal(DomainModel):
    """One publicly observable technical signal."""

    technology: str
    category: TechCategory
    confidence: Confidence = Confidence.LOW
    basis: str = "publicly observable signal"
    source_id: str | None = None
    source_url: str | None = None
    observed_at: str | None = None


class TechnologyAnalysis(DomainModel):
    """All technology signals found for one company."""

    company: str
    signals: list[TechnologySignal] = Field(default_factory=list)
    categories_covered: list[str] = Field(default_factory=list)
    job_postings_seen: int = 0
    github_orgs: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
