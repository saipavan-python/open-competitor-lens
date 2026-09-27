"""Search-layer schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from app.core.constants import SourceType
from app.schemas.common import DomainModel, utcnow


class SearchHit(DomainModel):
    """One search result.

    A hit is a *discovery* artefact. It may seed a crawl, but
    ``source_type`` is :attr:`SourceType.SEARCH_SNIPPET` until the page behind
    it is actually retrieved.
    """

    url: str
    title: str | None = None
    snippet: str | None = None
    domain: str | None = None
    rank: int = 0
    provider: str | None = None
    retrieved_at: datetime = Field(default_factory=utcnow)
    raw_score: float | None = None

    @field_validator("url")
    @classmethod
    def _url_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("SearchHit.url is required")
        return value.strip()

    @property
    def source_type(self) -> SourceType:
        """Search results are always snippet-tier until retrieved."""
        return SourceType.SEARCH_SNIPPET

    @property
    def is_snippet_only(self) -> bool:
        return True


class SearchRequest(DomainModel):
    """A single search call."""

    query: str
    limit: int = Field(default=10, ge=1, le=50)
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    country: str | None = None
    search_depth: str = "basic"

    @field_validator("query")
    @classmethod
    def _query_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("SearchRequest.query is required")
        return value.strip()


class SearchResponse(DomainModel):
    """Search results plus the provider that produced them."""

    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    provider: str | None = None
    duration_ms: int | None = None
    truncated: bool = False
