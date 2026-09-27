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


class CandidateSignals(DomainModel):
    """A candidate together with the signals gathered about it.

    The unit the validator judges. Separating it from
    :class:`CompetitorCandidate` is what makes the validator usable without a
    model, a network or a run: a candidate plus signals is everything the verdict
    needs, so the part of this system that must not fail when a provider is down
    has no dependency on one.
    """

    candidate: CompetitorCandidate
    signals: list[ValidationSignal] = Field(default_factory=list)


class ValidationOutcome(DomainModel):
    """The verdict on one candidate, accepted or not.

    Rejected candidates are kept rather than discarded, because a reader who
    wants to know why a company the model proposed is *absent* from the report
    has no other way to find out.
    """

    candidate: CompetitorCandidate
    accepted: bool
    reason: str
    signals: list[ValidationSignal] = Field(default_factory=list)
    primary_signals: int = 0

    def to_competitor(self) -> ValidatedCompetitor:
        """The accepted form, carrying the reason it was kept."""
        if not self.accepted:
            raise ValueError(
                f"refusing to promote a rejected candidate ({self.reason}) to a "
                f"ValidatedCompetitor"
            )
        return ValidatedCompetitor(
            **self.candidate.model_dump(),
            relevance_reason=self.reason,
            signals=list(self.signals),
        )


class CompetitorSelection(DomainModel):
    """The outcome of selecting competitors from a pool.

    This is the discovery agent's declared output, so it lives with the schemas:
    the graph needs to read it, and a type that only the agent module can import
    would force the graph to depend on the agent to consume its results.
    """

    competitors: list[ValidatedCompetitor] = Field(default_factory=list)
    rejected: list[ValidationOutcome] = Field(default_factory=list)
    status: str = "ok"

    def __len__(self) -> int:
        return len(self.competitors)
