"""SEO schemas.

Only signals readable from the page itself are recorded. No ranking, traffic or
backlink figures appear here, because this system has no verified source for
them and must never imply it does.
"""

from __future__ import annotations

from pydantic import Field

from app.core.constants import Confidence
from app.schemas.common import NOT_PUBLIC, DomainModel


class SeoAnalysis(DomainModel):
    """Public, page-level SEO signals for one company."""

    company: str
    title_tag: str | None = None
    meta_descriptions: list[str] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    has_canonical: bool | None = None
    canonical_tags: list[str] = Field(default_factory=list)
    robots_txt_present: bool | None = None
    robots_directives: list[str] = Field(default_factory=list)
    sitemap_present: bool | None = None
    sitemap_url_count: int | None = None
    structured_data_found: bool = False
    structured_data_types: list[str] = Field(default_factory=list)
    blog_present: bool | None = None
    content_themes: list[str] = Field(default_factory=list)
    keyword_themes: list[str] = Field(default_factory=list)
    indexable_page_signals: list[str] = Field(default_factory=list)
    pages_checked: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    evidence: list[str] = Field(default_factory=list)

    @property
    def ranking_data(self) -> str:
        """Rankings are never established by this system."""
        return NOT_PUBLIC
