"""Rendering a report for a reader, and measuring how well it is evidenced.

The renderer emits all nineteen sections unconditionally. That is the design
decision this module exists to make: a section with nothing behind it says
"nothing was retrieved for this category" rather than not appearing, because a
report that silently omits a category is indistinguishable from one where that
category was researched and found to be uninteresting. The reader needs to be able
to tell those apart, and the only way to tell them apart is for the gap to be
printed.

The sentinel travels with the data. ``"Not publicly available"`` is rendered as
itself in a table cell and in prose, never as a blank, a dash, a zero or an
omitted row, so a reader scanning the table cannot mistake an unresearched field
for a free one.

:meth:`ReportService.quality` computes the report's own evidence section from the
evidence records, so the number a reader is shown as "supported claims" is
counted from the run's actual evidence rather than reported by whatever step
produced the prose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.core.constants import NON_PRIMARY_SOURCE_TYPES, ClaimVerdict
from app.core.logging import get_logger
from app.schemas.common import NOT_PUBLIC
from app.schemas.comparison import ComparisonMatrix
from app.schemas.report import (
    REPORT_SECTIONS,
    SECTION_TITLES,
    EvidenceQuality,
    ReportBundle,
)

log = get_logger(__name__)

#: Printed in place of a section that has no content, so the gap is visible.
EMPTY_SECTION = "_No public information was retrieved for this section._"

#: Printed for a table cell the research did not fill. Identical to the sentinel
#: used in the data, so a reader sees the same words in prose and in tables.
ABSENT = NOT_PUBLIC


class ReportService:
    """Renders a report to Markdown and HTML, and measures its evidence."""

    # -- rendering --------------------------------------------------------

    def render(self, bundle: ReportBundle, *, include_html: bool = True) -> ReportBundle:
        """Fill ``report_markdown`` (and ``report_html``), returning the bundle.

        Mutates and returns rather than returning a string, so the caller can hold
        one object that is both the data and its rendered form — a report whose
        rendered text disagrees with its fields is a bug waiting to be reported by
        a reader instead.
        """
        bundle.report_markdown = self.markdown(bundle)
        if include_html:
            bundle.report_html = self.html(bundle)
        return bundle

    def missing_sections(self, bundle: ReportBundle) -> list[str]:
        """Which of the nineteen sections have no content.

        Used by the report agent to state its own gaps, so the same definition of
        "empty" governs the prose and the data.
        """
        content = self._section_content(bundle)
        return [name for name in REPORT_SECTIONS if is_empty(content.get(name))]

    def markdown(self, bundle: ReportBundle) -> str:
        """The whole report as Markdown."""
        content = self._section_content(bundle)
        parts: list[str] = [f"# {_title(bundle)}", ""]
        for name in REPORT_SECTIONS:
            parts.append(f"## {SECTION_TITLES.get(name, name)}")
            parts.append("")
            parts.append(self._section_markdown(name, content.get(name)))
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def html(self, bundle: ReportBundle) -> str:
        """The whole report as HTML.

        Escaped once, here, at the point of emission — and not in the Markdown.
        The report is assembled from model output and crawled pages, so any of it
        can contain a ``<script>``. Escaping the Markdown as well would corrupt it
        twice over: a reader of the Markdown file would see ``&#x27;`` where an
        apostrophe belongs, and every entity here would be escaped a second time.
        """
        content = self._section_content(bundle)
        parts: list[str] = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            f"<title>{_escape(_title(bundle))}</title></head><body>",
            f"<h1>{_escape(_title(bundle))}</h1>",
        ]
        for name in REPORT_SECTIONS:
            body = self._section_markdown(name, content.get(name))
            parts.append(f"<h2>{_escape(SECTION_TITLES.get(name, name))}</h2>")
            parts.append(f"<pre>{_escape(body)}</pre>")
        parts.append("</body></html>")
        return "\n".join(parts)

    @staticmethod
    def _feature_rows(matrix: ComparisonMatrix | None) -> ComparisonMatrix | None:
        """Just the feature rows of the product comparison.

        The bundle has one matrix but the report has two sections: what the
        products are (5) and what they can do (6). Splitting on the row's own
        category keeps each section about one question instead of printing the
        same table twice.
        """
        if matrix is None:
            return None
        rows = [row for row in matrix.dimensions if (row.category or "") == "feature"]
        if not rows:
            return None
        return ComparisonMatrix(
            companies=list(matrix.companies),
            dimensions=rows,
            evidence_quality=dict(matrix.evidence_quality),
            notes=list(matrix.notes),
        )

    def _section_content(self, bundle: ReportBundle) -> dict[str, Any]:
        """Every section's content, by section name.

        The map lives here rather than in the agent because deciding which field
        backs which section is a rendering concern, and the agent layer is above
        this one.

        The executive summary is computed here rather than read from the bundle.
        ``report_markdown`` holds the *whole* rendered document, so a summary taken
        from it would be the entire previous render — and the second render would
        nest the first report inside its own summary.
        """
        content = sections_of(bundle)
        content["executive_summary"] = self.summary(bundle)
        return content

    def summary(self, bundle: ReportBundle) -> str:
        """The report's opening paragraph, stated from the run's own numbers.

        Derived and counted rather than written by a model, for the same reason
        the rest of this service is: a headline asserting more than the report
        contains is the one sentence a reader is most likely to quote back.
        """
        target = bundle.metadata.company or "the target company"
        lines = [
            f"This report compares {target} against "
            f"{len(bundle.competitors)} competitor(s), built from "
            f"{len(bundle.sources)} retrieved source(s) and "
            f"{bundle.evidence_quality.major_claims} checked claim(s)."
        ]
        if bundle.evidence_quality.supported_ratio:
            lines.append(
                f"{bundle.evidence_quality.supported_claims} of those claims are "
                f"supported by evidence "
                f"(ratio {bundle.evidence_quality.supported_ratio})."
            )
        if bundle.opportunities:
            lines.append(
                f"{len(bundle.opportunities)} opportunity/ies are offered, each citing "
                f"the evidence that revealed the gap."
            )
        else:
            lines.append(
                "No opportunity is offered: none of the observed gaps was backed by "
                "retrieved evidence, and this report does not assert one anyway."
            )
        if bundle.failed_steps:
            lines.append(
                f"{len(bundle.failed_steps)} step(s) of the run did not complete; see "
                f"the affected sections."
            )
        return " ".join(lines)

    def _section_markdown(self, name: str, value: Any) -> str:
        """One section, in the shape its content calls for."""
        if is_empty(value):
            return EMPTY_SECTION
        if isinstance(value, ComparisonMatrix):
            return self._matrix_markdown(value)
        if name == "sources":
            return self._sources_markdown(value)
        if name == "evidence_quality":
            return self._quality_markdown(value)
        if name == "target_company_overview":
            return self._profile_markdown(value)
        if name == "competitor_landscape":
            return self._competitors_markdown(value)
        if isinstance(value, Mapping):
            return self._mapping_markdown(value)
        if isinstance(value, (list, tuple)):
            return self._list_markdown(value)
        return str(value)

    def _matrix_markdown(self, matrix: ComparisonMatrix) -> str:
        """A comparison table, with the sentinel in every unfilled cell."""
        if not matrix.companies:
            return EMPTY_SECTION
        lines = [
            "| Dimension | " + " | ".join(_cell(name) for name in matrix.companies) + " |"
        ]
        lines.append("| --- | " + " | ".join("---" for _ in matrix.companies) + " |")
        for dimension in matrix.dimensions:
            label = dimension.label or dimension.field
            cells = [_cell(dimension.display_for(name)) for name in matrix.companies]
            lines.append(f"| {_cell(label)} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _competitors_markdown(self, competitors: Sequence[Any]) -> str:
        """The landscape, one readable block per competitor.

        Rendered field by field rather than as the record's repr: a landscape line
        reading ``name='Globex' website=None validated=False`` tells a reader
        nothing that "Globex — not yet validated" does not, and the report is
        written for readers.
        """
        lines: list[str] = []
        for record in competitors:
            lines.append(f"**{_text(_attr(record, 'name'))}**")
            details = [
                f"Website: {_attr(record, 'website')}",
                f"Industry: {_attr(record, 'industry')}",
                f"Country: {_attr(record, 'country')}",
            ]
            lines.extend(f"- {detail}" for detail in details)
            verified = bool(_attr(record, "validated"))
            check = "validated by this run" if verified else "not validated by this run"
            lines.append(f"- Identity: {check}")
            reason = _attr(record, "relevance_reason")
            if reason:
                lines.append(f"- Why a competitor: {_text(reason)}")
            if _attr(record, "rejected_reason"):
                lines.append(f"- Rejected: {_text(_attr(record, 'rejected_reason'))}")
            lines.append("")
        return "\n".join(lines).strip() or EMPTY_SECTION

    def _sources_markdown(self, sources: Sequence[Any]) -> str:
        """Every source, including the ones that were only discovery artefacts."""
        lines: list[str] = []
        for source in sources:
            url = getattr(source, "url", "")
            title = getattr(source, "title", None) or url
            kind = getattr(source, "source_type", None)
            kind_text = getattr(kind, "value", None) or "unknown"
            lines.append(f"- [{_text(title)}]({url}) — {kind_text}")
        return "\n".join(lines) or EMPTY_SECTION

    def _quality_markdown(self, quality: EvidenceQuality) -> str:
        """The evidence section, counted rather than asserted."""
        return "\n".join(
            [
                f"- Claims made: {quality.major_claims}",
                f"- Supported: {quality.supported_claims}",
                f"- Unsupported: {quality.unsupported_claims}",
                f"- Conflicting: {quality.conflicting_claims}",
                f"- Stale: {quality.stale_claims}",
                f"- Sources retrieved: {quality.total_sources}",
                f"- Primary sources: {quality.primary_sources}",
                f"- Snippet-only sources: {quality.snippet_only_sources}",
                f"- Supported ratio: {quality.supported_ratio}",
            ]
        )

    def _profile_markdown(self, profile: Any) -> str:
        """The target company, field by field, sentinel for what is unknown."""
        if isinstance(profile, Mapping):
            items = profile.items()
        else:
            items = [
                (name, getattr(profile, name))
                for name in type(profile).model_fields
                if not name.startswith("_")
            ]
        lines = [
            f"- **{_text(name)}**: {_text(value) if value not in (None, '') else ABSENT}"
            for name, value in items
        ]
        return "\n".join(lines) or EMPTY_SECTION

    def _mapping_markdown(self, value: Mapping[str, Any]) -> str:
        """A per-company mapping, one block per company."""
        blocks: list[str] = []
        for company, payload in value.items():
            if is_empty(payload):
                blocks.append(f"**{_text(company)}**: {ABSENT}")
                continue
            if isinstance(payload, Mapping):
                fields = list(payload.items())
            else:
                fields = [
                    (name, getattr(payload, name, None))
                    for name in getattr(type(payload), "model_fields", {})
                ]
            body = "\n".join(
                f"  - {_text(name)}: {_text(item) if item not in (None, '') else ABSENT}"
                for name, item in fields
                if not name.startswith("_")
            )
            blocks.append(f"**{_text(company)}**\n{body}")
        return "\n\n".join(blocks) or EMPTY_SECTION

    def _list_markdown(self, items: Sequence[Any]) -> str:
        """A list of statements, opportunities or risks."""
        lines: list[str] = []
        for item in items:
            if isinstance(item, str):
                lines.append(f"- {item}")
                continue
            label = _first_of(
                item, ("pattern", "opportunity", "risk", "statement", "observation")
            )
            if label:
                lines.append(f"- {_text(label)}")
                continue
            lines.append(f"- {_text(item)}")
        return "\n".join(lines) or EMPTY_SECTION

    # -- measurement ------------------------------------------------------

    def quality(self, bundle: ReportBundle) -> EvidenceQuality:
        """Count the report's own evidence, from the run's records.

        Derived rather than supplied, and counted from the evidence rather than
        from the prose: a report that claims 90% of its claims are supported while
        its evidence records say otherwise should not be able to.
        """
        evidence = [item for item in bundle.evidence if _is_record(item)]
        verdicts = list(bundle.claim_verdicts)
        sources = list(bundle.sources)

        supported = sum(1 for record in verdicts if record.verdict is ClaimVerdict.SUPPORTED)
        quality = EvidenceQuality(
            major_claims=len(verdicts) or len(evidence),
            supported_claims=supported,
            unsupported_claims=sum(
                1
                for record in verdicts
                if record.verdict in (ClaimVerdict.UNSUPPORTED, ClaimVerdict.OVERSTATEMENT)
            ),
            conflicting_claims=sum(
                1 for record in verdicts if record.verdict is ClaimVerdict.CONTRADICTORY
            ),
            stale_claims=sum(1 for record in verdicts if record.verdict is ClaimVerdict.STALE),
            total_sources=len(sources),
            primary_sources=sum(1 for source in sources if _is_primary(source)),
            secondary_sources=sum(1 for source in sources if _is_secondary(source)),
            snippet_only_sources=sum(1 for source in sources if not _has_text(source)),
        )
        quality.confidence_distribution = _confidence_distribution(evidence)
        if quality.total_sources and not quality.primary_sources:
            quality.notes.append(
                "no primary source was retrieved; every finding rests on a secondary "
                "or snippet-only page"
            )
        return quality


def _title(bundle: ReportBundle) -> str:
    """The report's title, from the run rather than a constant."""
    company = bundle.metadata.company or "Unknown company"
    return f"Competitive analysis: {company}"


def _text(value: Any) -> str:
    """A value as plain text for the Markdown rendering.

    A schema object that reached here was not recognised by any section-specific
    renderer, so it is described by its repr rather than printed as an opaque
    object. The Markdown is not HTML-escaped — :meth:`ReportService.html` escapes
    the finished document once, and escaping here as well would show a reader
    ``&#x27;`` in place of an apostrophe.
    """
    if value is None:
        return ABSENT
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value)


def _cell(value: Any) -> str:
    """A value as Markdown table-cell text.

    Pipes and newlines are the two characters that would silently break the table
    around it, so they are neutralised here. This is Markdown's own concern, and is
    unrelated to the HTML escaping done once at the end.
    """
    return _text(value).replace("|", "\\|").replace("\n", " ").strip()


def _attr(item: Any, name: str) -> Any:
    """One field of a schema object or a mapping."""
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _has_text(source: Any) -> bool:
    """Whether a source carries retrieved text, as opposed to being a bare URL."""
    return bool(getattr(source, "text", None) and str(source.text).strip())


def _source_type(source: Any) -> Any:
    """A source's type, or ``None`` when it was never recorded."""
    if isinstance(source, Mapping):
        return source.get("source_type")
    return getattr(source, "source_type", None)


def _is_primary(source: Any) -> bool:
    """Whether a source is a primary one.

    Requires a recorded type. A source whose type was never set is not counted as
    primary, because "we did not record what kind of page this was" is not the
    same claim as "this was the company's own site", and the report's evidence
    section should not be flattered by the difference.
    """
    kind = _source_type(source)
    return kind is not None and kind not in NON_PRIMARY_SOURCE_TYPES


def _is_secondary(source: Any) -> bool:
    """Whether a source is a snippet or otherwise not citable."""
    kind = _source_type(source)
    return kind is not None and kind in NON_PRIMARY_SOURCE_TYPES


def _is_record(item: Any) -> bool:
    """Whether an entry in ``bundle.evidence`` is usable as a record.

    The bundle types evidence as plain dicts but a step may hand over ``Evidence``
    objects before they are dumped, so both shapes are accepted rather than
    silently dropping the object form out of the count.
    """
    return isinstance(item, Mapping) or hasattr(item, "model_dump")





def _confidence_distribution(evidence: Sequence[Any]) -> dict[str, int]:
    """How many evidence records sit at each confidence level."""
    counts: dict[str, int] = {}
    for item in evidence:
        value = (
            item.get("confidence")
            if isinstance(item, Mapping)
            else getattr(item, "confidence", None)
        )
        level = str(value) if value else "low"
        counts[level] = counts.get(level, 0) + 1
    return counts


def _first_of(item: Any, names: Sequence[str]) -> str | None:
    """The first non-empty text field of a schema or a mapping."""
    for name in names:
        value = item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _escape(text: str) -> str:
    import html as _html

    return _html.escape(str(text or ""), quote=True)


#: Kept in the module namespace so a caller can check the string it will render.
EMPTY_MARKER = EMPTY_SECTION


def sections_of(bundle: ReportBundle) -> dict[str, Any]:
    """The bundle's own content, keyed by report section name.

    The single place the mapping from a section name to the field holding it is
    written down, so the renderer and the "which sections are empty" check cannot
    drift apart. Every one of the nineteen sections appears: a section the
    renderer does not know about would be reported as missing while holding
    content, or printed as empty while holding a full analysis.
    """
    return {
        "executive_summary": bundle.report_markdown or "",
        "target_company_overview": bundle.target_company,
        "competitor_landscape": bundle.competitors,
        "competitor_profiles": bundle.competitor_profiles,
        "product_comparison": _without_feature_rows(bundle.product_comparison),
        "feature_comparison": _feature_rows(bundle.product_comparison),
        "pricing_comparison": bundle.pricing_comparison,
        "marketing_analysis": bundle.marketing_analysis,
        "seo_analysis": bundle.seo_analysis,
        "social_analysis": bundle.social_analysis,
        "review_analysis": bundle.review_analysis,
        "technology_analysis": bundle.technology_analysis,
        "market_analysis": bundle.market_analysis,
        "cross_competitor_patterns": bundle.cross_competitor_patterns,
        "observed_gaps": bundle.observed_gaps,
        "potential_opportunities": bundle.opportunities,
        "risks_and_constraints": bundle.risks,
        "evidence_quality": bundle.evidence_quality,
        "sources": bundle.sources,
    }


def _without_feature_rows(matrix: ComparisonMatrix | None) -> ComparisonMatrix | None:
    """The product comparison minus the rows section 6 already prints."""
    if matrix is None:
        return None
    rows = [row for row in matrix.dimensions if (row.category or "") != "feature"]
    if not rows:
        return None
    return ComparisonMatrix(
        companies=list(matrix.companies),
        dimensions=rows,
        evidence_quality=dict(matrix.evidence_quality),
        notes=list(matrix.notes),
    )


def _feature_rows(matrix: ComparisonMatrix | None) -> ComparisonMatrix | None:
    """Just the feature rows of the product comparison.

    The bundle holds one matrix but the report has two sections: what the products
    are (5) and what they can do (6). Splitting on each row's own category keeps
    each section about one question instead of printing the same table twice.
    """
    if matrix is None:
        return None
    rows = [row for row in matrix.dimensions if (row.category or "") == "feature"]
    if not rows:
        return None
    return ComparisonMatrix(
        companies=list(matrix.companies),
        dimensions=rows,
        evidence_quality=dict(matrix.evidence_quality),
        notes=list(matrix.notes),
    )


def is_empty(value: Any) -> bool:
    """Whether a section has nothing worth printing.

    The single definition of "empty" for the whole renderer, so the "which
    sections are missing" check and the renderer itself cannot disagree about it.
    The sentinel is a real value and does not count as empty, because a section
    saying "not publicly available" has said something.
    """
    if value is None:
        return True
    if isinstance(value, (str, list, tuple, dict, set)):
        return len(value) == 0
    return not value.dimensions if isinstance(value, ComparisonMatrix) else False
