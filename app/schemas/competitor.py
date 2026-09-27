"""Competitor candidate and validation schemas."""

from __future__ import annotations

from pydantic import Field, field_validator

from app.schemas.common import DomainModel
from app.schemas.company import CompanyIdentity


class CompetitorCandidate(DomainModel):
    """A company the discovery step thinks might compete.

    Deliberately carries no score: a raw LLM or search ranking is not evidence
    of relevance, and exposing one invites downstream code to trust it.
    """

    name: str
    website: str | None = None
    description: str | None = None
    industry: str | None = None
    country: str | None = None
    discovery_query: str | None = None
    discovery_source_urls: list[str] = Field(default_factory=list)
    identity: CompanyIdentity | None = None
    validated: bool = False
    evidence: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("CompetitorCandidate.name is required")
        return value.strip()


class ValidationSignal(DomainModel):
    """One piece of evidence that a candidate really is a competitor."""

    signal: str
    supported: bool
    source_ids: list[str] = Field(default_factory=list)
    detail: str | None = None


class ValidatedCompetitor(CompetitorCandidate):
    """A candidate that survived validation.

    ``relevance_reason`` must be non-empty: a competitor is only ever included
    with a stated, evidence-backed reason for its inclusion.
    """

    relevance_reason: str
    is_target: bool = False
    signals: list[ValidationSignal] = Field(default_factory=list)
    rejected_reason: str | None = None

    @field_validator("relevance_reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("A validated competitor must state a relevance_reason")
        return value.strip()
