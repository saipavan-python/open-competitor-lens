"""The comparison engine.

A comparison table makes claims by juxtaposition. Put Acme's price in a cell and
Beta's beside it and the reader sees a ranking, whether or not one was intended —
and if Beta's cell is empty, the reader sees a company with no price rather than a
company whose price nobody looked up. Every way this service could be wrong
quietly produces that mistake, so the shape of the code is mostly refusals:

* an absent value is the sentinel, never ``None``, never ``0``, never blank;
* a company absent from the analyses gets a full row of sentinels rather than
  being dropped from the table;
* ranking is refused outright on a dimension no company supplied a number for,
  because "cheapest" is not a ranking of one known price against five unknowns.

That last one is the reason this is a service rather than a helper inside the
comparison agent. Deciding *not to rank* is a judgement about the whole table,
and it belongs where the whole table is visible.

The typed analysis payloads are reduced to comparable fields by
:meth:`ComparisonService.fields_from`, which reads them field by field rather
than dumping ``model_dump()`` into the table. Dumping would put every
``notes`` list and every ``confidence`` enum into a reader-facing dimension, and
would carry the sentinel through as though it were a value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.core.constants import Confidence
from app.schemas.common import NOT_PUBLIC
from app.schemas.company import CompanyProfile
from app.schemas.comparison import (
    ComparisonCell,
    ComparisonMatrix,
    ComparisonRow,
    CompetitorCoverage,
)
from app.schemas.market import MarketAnalysis
from app.schemas.marketing import MarketingAnalysis
from app.schemas.pricing import PricingAnalysis
from app.schemas.products import ProductAnalysis
from app.schemas.reviews import ReviewAnalysis
from app.schemas.seo import SeoAnalysis
from app.schemas.social import SocialAnalysis
from app.schemas.technology import TechnologyAnalysis

#: The typed analyses that can be reduced to comparable fields, and the fields
#: worth comparing in each. Ordered target-first, and the order a table is built
#: in when no explicit list is given.
ANALYSIS_FIELDS: dict[str, tuple[str, ...]] = {
    "pricing": (
        "pricing_model",
        "billing_periods",
        "has_free_tier",
        "has_trial",
        "has_enterprise_tier",
        "currency",
        "pricing_page_url",
    ),
    "product": (
        "products",
        "features",
        "segments",
        "use_cases",
        "differentiators",
        "positioning",
    ),
    "website": (
        "description",
        "industry",
        "country",
        "target_segments",
        "value_proposition",
    ),
    "marketing": (
        "messaging",
        "channels",
        "case_studies",
    ),
    "seo": (
        "title_tag",
        "meta_description",
        "has_sitemap",
        "has_robots_txt",
        "structured_data",
    ),
    "social": (
        "platforms",
        "total_followers",
    ),
    "review": (
        "overall_rating",
        "review_count",
        "themes",
    ),
    "technology": (
        "technologies",
        "languages",
    ),
    "market": (
        "market_size",
        "trends",
        "gaps",
    ),
}

#: Payload types this service knows how to reduce, by schema class.
_REDUCERS: dict[type, tuple[str, ...]] = {
    PricingAnalysis: ANALYSIS_FIELDS["pricing"],
    ProductAnalysis: ANALYSIS_FIELDS["product"],
    CompanyProfile: ANALYSIS_FIELDS["website"],
    MarketingAnalysis: ANALYSIS_FIELDS["marketing"],
    SeoAnalysis: ANALYSIS_FIELDS["seo"],
    SocialAnalysis: ANALYSIS_FIELDS["social"],
    ReviewAnalysis: ANALYSIS_FIELDS["review"],
    TechnologyAnalysis: ANALYSIS_FIELDS["technology"],
    MarketAnalysis: ANALYSIS_FIELDS["market"],
}

#: Fields that describe *how well supported* a cell is rather than the value
#: itself, and so belong on the cell's metadata instead of in the table body.
_PROVENANCE_FIELDS = frozenset({"source_ids", "confidence", "reason"})


@dataclass(frozen=True)
class RankedDimension:
    """One company, placed on a dimension, with the value that placed it."""

    field: str
    company: str
    value: float
    rank: int


class ComparisonService:
    """Builds the cross-company table, and refuses the conclusions it cannot support."""

    def build(
        self,
        companies: list[str],
        analyses: Mapping[str, Mapping[str, Any]],
        *,
        fields: list[str] | None = None,
        labels: Mapping[str, str] | None = None,
        notes: list[str] | None = None,
    ) -> ComparisonMatrix:
        """The comparison table for ``companies``.

        ``analyses`` maps a company name to its comparable fields, already
        reduced by :meth:`fields_from`. A company missing from it, a field
        missing from it, and a value that is empty all produce the sentinel.
        """
        ordered = self._dimensions(companies, analyses, fields)
        dimensions: list[ComparisonRow] = []
        for field in ordered:
            dimensions.append(
                ComparisonRow(
                    field=field,
                    label=self._label(field, labels),
                    cells={
                        company: self._cell(analyses.get(company, {}).get(field, _MISSING))
                        for company in companies
                    },
                    values={
                        company: self._cell(
                            analyses.get(company, {}).get(field, _MISSING)
                        ).display
                        for company in companies
                    },
                )
            )

        return ComparisonMatrix(
            companies=list(companies),
            rows=[
                {
                    "field": row.field,
                    "label": row.label,
                    "values": row.values,
                }
                for row in dimensions
            ],
            dimensions=dimensions,
            evidence_quality={
                company: self.quality_score(analyses.get(company, {})) for company in companies
            },
            notes=list(notes or []),
        )

    def rank(
        self, matrix: ComparisonMatrix, field: str
    ) -> list[RankedDimension] | None:
        """Companies ordered by ``field``, or ``None`` if that is not meaningful.

        Refused unless *every* company in the table has a known number for the
        field. A ranking is a statement about the whole set — "the most expensive"
        is a claim about the companies that lost as much as about the one that
        won — so one unknown price makes it unsupportable, and a partial ranking
        would let a caller announce a winner the data never established.

        ``None`` rather than an empty list, because the two are different
        findings: an empty list reads as "nobody scored", while a refusal reads as
        "this dimension cannot be ranked", which is the thing a reader needs to be
        told.
        """
        row = matrix.row(field)
        if row is None:
            return None
        if any(company not in row.cells for company in matrix.companies):
            return None
        values: list[tuple[str, float]] = []
        for company in matrix.companies:
            cell = row.cells[company]
            if not cell.is_known:
                return None
            value = cell.value
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            values.append((company, value))
        if not values:
            return None
        values.sort(key=lambda item: item[1], reverse=True)
        return [
            RankedDimension(field=field, company=company, value=value, rank=index)
            for index, (company, value) in enumerate(values, start=1)
        ]

    def fields_from(self, analysis: Any) -> dict[str, Any]:
        """Reduce a typed analysis payload to its comparable fields.

        Refuses a type it has no field list for. Guessing — dumping the model and
        letting the table show whatever falls out — would put a reader-facing
        dimension named ``notes`` or ``evidence`` into the comparison, and would
        quietly widen the table whenever a schema gains a field.
        """
        wanted = self._fields_for(type(analysis))
        if wanted is None:
            raise TypeError(
                f"ComparisonService cannot reduce {type(analysis).__name__} to "
                f"comparison fields; known analyses: "
                f"{', '.join(sorted(m.__name__ for m in _REDUCERS))}"
            )
        fields: dict[str, Any] = {}
        for name in wanted:
            value = getattr(analysis, name, None)
            fields[name] = NOT_PUBLIC if self._is_absent(value) else value
        return fields

    def quality_score(self, fields: Mapping[str, Any]) -> float:
        """How much of a company's row is actually cited, from 0.0 to 1.0.

        A proportion of cited cells rather than a source count, so a company with
        one sourced cell out of nine does not score like a company whose every
        cell is sourced.
        """
        if not fields:
            return 0.0
        cited = sum(1 for value in fields.values() if self._cited(value))
        return round(cited / len(fields), 3)

    def coverage(
        self,
        company: str,
        *,
        covered: list[str] | None = None,
        failed: list[str] | None = None,
        sources: int = 0,
        evidence: int = 0,
    ) -> CompetitorCoverage:
        """Per-company category coverage, including the categories that failed."""
        return CompetitorCoverage(
            company=company,
            categories_covered=list(covered or []),
            categories_failed=list(failed or []),
            sources_retrieved=sources,
            evidence_count=evidence,
        )

    # -- internals --------------------------------------------------------

    def _dimensions(
        self,
        companies: list[str],
        analyses: Mapping[str, Mapping[str, Any]],
        fields: list[str] | None,
    ) -> list[str]:
        """Which fields the table has rows for, in a deliberate order.

        An explicit list wins and is honoured even where no company supplied the
        field, because a report's shape is chosen deliberately and an analyst that
        found nothing should leave a gap in a decided table rather than remove the
        row.

        Without one, the order is canonical rather than incidental: known analysis
        fields appear in the order :data:`ANALYSIS_FIELDS` declares them, and
        anything else is sorted. Deriving the order from whichever mapping arrived
        first would emit the same table in a different sequence on a second run
        whose payload happened to be built with its keys in a different order —
        two reports of identical data that differ for no reason a reader can see.
        """
        if fields is not None:
            return list(dict.fromkeys(fields))
        supplied: set[str] = set()
        for company in companies:
            supplied.update(analyses.get(company, {}))
        canonical = [
            name
            for wanted in ANALYSIS_FIELDS.values()
            for name in wanted
            if name in supplied
        ]
        ordered = list(dict.fromkeys(canonical))
        ordered.extend(sorted(supplied.difference(ordered)))
        return ordered

    def _cell(self, value: Any) -> ComparisonCell:
        """One cell. The only place the sentinel is produced."""
        if value is _MISSING:
            return ComparisonCell.unknown()
        if isinstance(value, Mapping) and "value" in value:
            return self._from_mapping(value)
        if self._is_absent(value):
            return ComparisonCell.unknown()
        return ComparisonCell.known(value)

    def _from_mapping(self, value: Mapping[str, Any]) -> ComparisonCell:
        """A cell written as ``{"value": ..., "source_ids": [...]}``.

        Reached only for a mapping that *has* a ``value`` key, so a structured
        value like ``{"name": "Pro", "price": 30}`` stays a value instead of being
        read as a wrapper around an absent inner value and becoming a sentinel.

        The provenance is separated from the value so a citation does not become a
        reader-facing cell value, and so the same cell can carry a confidence
        without a second argument at the call site.
        """
        inner = value.get("value", _MISSING)
        confidence = value.get("confidence")
        if isinstance(confidence, Confidence):
            level = confidence
        elif isinstance(confidence, str):
            try:
                level = Confidence(confidence)
            except ValueError:
                level = Confidence.LOW
        else:
            level = Confidence.MEDIUM
        if inner is _MISSING or self._is_absent(inner):
            return ComparisonCell.unknown(reason=value.get("reason"))
        sources = [str(item) for item in value.get("source_ids", []) or []]
        return ComparisonCell.known(inner, source_ids=sources, confidence=level)

    def _fields_for(self, analysis_type: type) -> tuple[str, ...] | None:
        """The field list for a payload type, by exact class or by isinstance.

        Exact class first, so a subclass of a known analysis does not silently
        inherit the parent's field list and lose its own.
        """
        if analysis_type in _REDUCERS:
            return _REDUCERS[analysis_type]
        for known, wanted in _REDUCERS.items():
            if isinstance(analysis_type, type) and issubclass(analysis_type, known):
                return wanted
        return None

    @staticmethod
    def _is_absent(value: Any) -> bool:
        """Whether a value means "we did not establish this".

        ``0`` and ``False`` are **not** absent. A free tier costs nothing and a
        "no trial" answer is an answer; treating either as missing would discard
        the two most decision-relevant values a pricing table can hold, and the
        ``or NOT_PUBLIC`` idiom that causes it is the single likeliest bug in a
        module like this.
        """
        if value is None or value is _MISSING:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, dict, set)):
            return len(value) == 0
        return False

    @classmethod
    def _cited(cls, value: Any) -> bool:
        """Whether a field's value carries at least one source id."""
        if isinstance(value, Mapping):
            return bool(value.get("source_ids"))
        return False

    @staticmethod
    def _label(field: str, labels: Mapping[str, str] | None) -> str:
        """A readable label, never empty.

        An empty label renders as a blank table header, which is worse than an
        ugly one: the dimension is still there, just unlabelled.
        """
        if labels and labels.get(field):
            return str(labels[field])
        return field.replace("_", " ").strip().title()


class _Missing:
    """Distinguishes "the analyst said nothing" from "the analyst said nothing yet".

    ``None`` cannot carry that distinction — a payload that omitted a field and a
    payload whose field is explicitly unknown are the same value, and the second
    one deserves to say why. Using ``None`` as the sentinel would also make the
    absent-value rule impossible to state precisely.
    """

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"

    def __bool__(self) -> bool:
        return False


_MISSING = _Missing()
