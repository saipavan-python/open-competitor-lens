"""Comparison schemas.

The comparison engine's contract: a cell the research could not fill holds the
literal string "Not publicly available" - never ``None``, never ``0``, never a
guess. :meth:`ComparisonCell.unknown` is the only way such a cell is built.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from app.core.constants import Confidence
from app.schemas.common import NOT_PUBLIC, DomainModel


class ComparisonCell(DomainModel):
    """One value in a comparison table, with its provenance."""

    value: Any = None
    is_known: bool = False
    display: str = NOT_PUBLIC
    source_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW

    @classmethod
    def unknown(cls, *, reason: str | None = None) -> ComparisonCell:
        """Build a cell for information that research did not establish.

        The sentinel is stored in ``value`` as well as ``display`` so a consumer
        that reads ``value`` directly still cannot mistake an absent value for a
        real one.
        """
        return cls(
            value=NOT_PUBLIC,
            is_known=False,
            display=NOT_PUBLIC if not reason else f"{NOT_PUBLIC} ({reason})",
            confidence=Confidence.LOW,
        )

    @classmethod
    def known(
        cls, value: Any, *, source_ids: list[str] | None = None,
        confidence: Confidence = Confidence.MEDIUM,
    ) -> ComparisonCell:
        return cls(
            value=value,
            is_known=True,
            display=str(value),
            source_ids=source_ids or [],
            confidence=confidence,
        )


class ComparisonRow(DomainModel):
    """One dimension compared across every company."""

    field: str
    label: str | None = None
    category: str | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    cells: dict[str, ComparisonCell] = Field(default_factory=dict)

    def display_for(self, company: str) -> str:
        if company in self.cells:
            return self.cells[company].display
        return NOT_PUBLIC


class ComparisonMatrix(DomainModel):
    """The full cross-competitor comparison."""

    companies: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    dimensions: list[ComparisonRow] = Field(default_factory=list)
    evidence_quality: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @field_validator("rows", mode="before")
    @classmethod
    def _absent_is_never_none(cls, value: object) -> object:
        """Replace a missing cell with the sentinel.

        ``None`` in a comparison table is ambiguous - it reads as "no data" in
        some renderers and as a real empty value in others. The sentinel makes
        "we looked and found nothing" explicit in every consumer.
        """
        if not isinstance(value, list):
            return value
        normalised: list[Any] = []
        for row in value:
            if isinstance(row, dict) and isinstance(row.get("values"), dict):
                row = {**row, "values": {k: (NOT_PUBLIC if v is None else v)
                                         for k, v in row["values"].items()}}
            normalised.append(row)
        return normalised

    def row(self, field: str) -> ComparisonRow | None:
        for dimension in self.dimensions:
            if dimension.field == field:
                return dimension
        return None


class CrossCompetitorPattern(DomainModel):
    """A pattern that appears across multiple companies (spec section 14)."""

    pattern: str
    pattern_type: str = "shared"
    companies: list[str] = Field(default_factory=list)
    count: int = 0
    observation: str
    interpretation: str | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW


class CompetitorCoverage(DomainModel):
    """Per-competitor category coverage, including the categories that failed."""

    company: str
    categories_covered: list[str] = Field(default_factory=list)
    categories_failed: list[str] = Field(default_factory=list)
    sources_retrieved: int = 0
    evidence_count: int = 0
