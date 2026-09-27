"""Review schemas.

Review text is never fabricated. Quotations are short and always carry the
source that published them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.core.constants import Confidence
from app.schemas.common import DomainModel

Sentiment = Literal["positive", "negative", "neutral", "mixed"]


class ReviewQuote(DomainModel):
    """A short verbatim excerpt with its origin."""

    text: str
    source_id: str
    source_url: str
    platform: str | None = None
    rating: float | None = None
    published_at: str | None = None


class ReviewTheme(DomainModel):
    """A recurring theme across reviews."""

    theme: str
    sentiment: Sentiment
    mention_count: int = 0
    companies: list[str] = Field(default_factory=list)
    quotes: list[ReviewQuote] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW


class ReviewPlatformSummary(DomainModel):
    """What one review platform publicly shows for a company."""

    platform: str
    url: str | None = None
    aggregate_rating: float | None = None
    review_count: int | None = None
    rating_observed_at: str | None = None
    evidence: list[str] = Field(default_factory=list)


class ReviewAnalysis(DomainModel):
    """The review agent's output for one company."""

    company: str
    platforms: list[ReviewPlatformSummary] = Field(default_factory=list)
    themes: list[ReviewTheme] = Field(default_factory=list)
    positive_themes: list[ReviewTheme] = Field(default_factory=list)
    negative_themes: list[ReviewTheme] = Field(default_factory=list)
    repeated_complaints: list[str] = Field(default_factory=list)
    feature_requests: list[str] = Field(default_factory=list)
    support_issues: list[str] = Field(default_factory=list)
    pricing_complaints: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
