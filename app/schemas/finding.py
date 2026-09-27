"""Finding and insight schemas.

A :class:`Finding` is a reportable statement. It must carry at least one
evidence id, and it keeps ``observation`` and ``interpretation`` in separate
fields so a reader can see where the fact ends and the reasoning begins.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.core.constants import AnalysisCategory, ClaimVerdict, Confidence, InsightType
from app.schemas.common import DomainModel


class Finding(DomainModel):
    """One evidence-backed statement about one company."""

    id: str
    company: str
    category: AnalysisCategory
    statement: str
    observation: str | None = None
    interpretation: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    value: str | None = None
    unknown: bool = False

    @field_validator("company", "statement")
    @classmethod
    def _required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Finding requires a non-empty company and statement")
        return value.strip()

    @model_validator(mode="after")
    def _require_evidence(self) -> Finding:
        if not self.evidence_ids:
            raise ValueError(
                f"Finding {self.id!r} has no evidence_ids; a finding must be traceable "
                "to at least one source"
            )
        return self


class Insight(DomainModel):
    """A cross-company structured insight (spec section 38)."""

    type: InsightType
    observation: str
    interpretation: str | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    companies: list[str] = Field(default_factory=list)
    category: AnalysisCategory | None = None


class ClaimVerdictRecord(DomainModel):
    """The fact checker's ruling on one claim."""

    claim_id: str
    claim: str
    verdict: ClaimVerdict
    reason: str
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    confidence_before: Confidence = Confidence.LOW
    confidence_after: Confidence = Confidence.LOW
    suggested_wording: str | None = None
    repaired: bool = False


class QualityIssue(DomainModel):
    """One problem found by quality control."""

    code: str
    severity: Literal["critical", "warning", "info"] = "warning"
    message: str
    location: str | None = None
    field: str | None = None


class QualityReport(DomainModel):
    """The output of the quality control stage (spec section 66)."""

    passed: bool = True
    critical_issues: list[QualityIssue] = Field(default_factory=list)
    warnings: list[QualityIssue] = Field(default_factory=list)
    repair_required: bool = False
    repair_rounds_used: int = 0
    repair_exhausted: bool = False
    checks_run: list[str] = Field(default_factory=list)
