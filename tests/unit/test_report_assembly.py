"""The assembly layer: comparison agent, opportunity agent, evidence agent, report.

These are the steps that turn nine analysts' payloads into one document, and the
tests concentrate on the ways that document can be quietly wrong:

* a table that implies a value nobody retrieved;
* an opportunity with no evidence behind the gap it names;
* a claim whose quote does not appear in the page it cites;
* a report that prints a tidy, complete-looking document with categories missing.

The recurring assertion is that a gap is *visible* rather than absent. A report
that silently drops an unresearched category is indistinguishable from one where
that category was researched and found uninteresting, so every test here checks
for the sentinel or the "no information retrieved" marker rather than for an
absence.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from app.agents.base import AgentContext, AgentResult, AgentStatus
from app.agents.comparison import ComparisonAgent
from app.agents.evidence_agent import EvidenceAgent
from app.agents.opportunity import OpportunityAgent
from app.agents.report_agent import ReportAgent
from app.core.constants import ClaimType, ClaimVerdict, Confidence, RunStatus
from app.schemas.common import NOT_PUBLIC
from app.schemas.comparison import (
    ComparisonCell,
    ComparisonMatrix,
    ComparisonRow,
    CrossCompetitorPattern,
)
from app.schemas.evidence import Source, SourceType
from app.schemas.finding import QualityReport
from app.schemas.market import Opportunity
from app.schemas.pricing import PricingAnalysis
from app.schemas.report import REPORT_SECTIONS, SECTION_TITLES, ReportBundle
from app.services.evidence_service import EvidenceService
from app.services.report_service import ReportService

TARGET = "Acme"
OTHER = "Globex"


def _source(
    source_id: str = "s1",
    *,
    url: str = "https://globex.example/pricing",
    title: str = "Globex Pricing",
    text: str = "Every plan includes a 14-day trial. A free tier is available.",
    source_type: SourceType = SourceType.OFFICIAL_SITE,
) -> Source:
    return Source(
        id=source_id, url=url, title=title, source_type=source_type, text=text
    )


def _matrix() -> ComparisonMatrix:
    return ComparisonMatrix(
        companies=[TARGET, OTHER],
        dimensions=[
            ComparisonRow(
                field="has_free_tier",
                label="Free tier",
                category="pricing",
                cells={
                    TARGET: ComparisonCell.unknown(),
                    OTHER: ComparisonCell.known(True, source_ids=["s1"]),
                },
            ),
            ComparisonRow(
                field="has_sso",
                label="SSO",
                category="feature",
                cells={
                    TARGET: ComparisonCell.known(True, source_ids=["s1"]),
                    OTHER: ComparisonCell.unknown(),
                },
            ),
        ],
        evidence_quality={TARGET: 0.5, OTHER: 1.0},
    )


def _run(agent, extras: dict, company: str = TARGET) -> AgentResult:
    return asyncio.run(agent.run(AgentContext(company=company, extras=extras)))


class TestComparisonAgent:
    """The agent wraps the service without changing what the service decides."""

    def test_a_single_company_is_refused_rather_than_tabled(self) -> None:
        """One column is not a comparison; the table would look like a deliverable."""
        result = _run(ComparisonAgent(), {"companies": [], "analyses": {}})
        assert result.status is AgentStatus.FAILED
        assert "at least two companies" in result.errors[0]

    def test_a_target_with_no_competitors_is_refused(self) -> None:
        result = _run(ComparisonAgent(), {"companies": [], "analyses": {}}, company="")
        assert result.status is AgentStatus.FAILED

    def test_the_matrix_is_built_from_every_companys_fields(self) -> None:
        matrix = _run(
            ComparisonAgent(),
            {
                "companies": [OTHER],
                "analyses": {
                    OTHER: {"has_free_tier": True, "has_sso": False},
                    TARGET: {"has_free_tier": False, "has_sso": True},
                },
            },
        ).payload
        assert isinstance(matrix, ComparisonMatrix)
        assert matrix.companies[0] == TARGET, "the target is compared against the others"
        assert {row.field for row in matrix.dimensions} >= {
            "has_free_tier",
            "has_sso",
        }

    def test_the_target_is_not_listed_twice(self) -> None:
        """A company that is both the target and a discovered competitor is one company."""
        matrix = _run(
            ComparisonAgent(),
            {"companies": [TARGET, OTHER], "analyses": {OTHER: {"has_sso": True}}},
        ).payload
        assert matrix.companies.count(TARGET) == 1

    def test_a_company_with_no_evidence_is_partial_not_failed(self) -> None:
        """A table with gaps is still a table, and the gaps are the finding."""
        result = _run(
            ComparisonAgent(),
            {"companies": [OTHER], "analyses": {OTHER: {"has_sso": True}}},
        )
        assert result.status is AgentStatus.PARTIAL
        assert result.payload.evidence_quality.get(TARGET, 0.0) == 0.0
        assert any(TARGET in warning for warning in result.warnings)

    def test_a_fully_evidenced_comparison_is_clean(self) -> None:
        """Cells carrying provenance score, so a cited comparison is not reported partial."""
        result = _run(
            ComparisonAgent(),
            {
                "companies": [OTHER],
                "analyses": {
                    OTHER: {"has_sso": {"value": True, "source_ids": ["s1"]}},
                    TARGET: {"has_sso": {"value": True, "source_ids": ["s2"]}},
                },
            },
        )
        assert result.status is AgentStatus.OK
        assert result.warnings == []
        assert result.payload.evidence_quality == {TARGET: 1.0, OTHER: 1.0}

    def test_an_uncited_comparison_is_reported_as_partial(self) -> None:
        """A value with no source id is a claim, and the score says so."""
        result = _run(
            ComparisonAgent(),
            {
                "companies": [OTHER],
                "analyses": {
                    OTHER: {"has_sso": {"value": True, "source_ids": ["s1"]}},
                    TARGET: {"has_sso": True},
                },
            },
        )
        assert result.status is AgentStatus.PARTIAL
        assert result.payload.evidence_quality[TARGET] == 0.0

    def test_an_unreducible_payload_becomes_a_gap_not_a_crash(self) -> None:
        result = _run(
            ComparisonAgent(),
            {
                "companies": [OTHER],
                "analyses": {
                    TARGET: "a string that is not an analysis",
                    OTHER: {"has_sso": True},
                },
            },
        )
        row = result.payload.row("has_sso")
        assert row is not None
        assert row.display_for(OTHER) == "True"
        assert row.display_for(TARGET) == NOT_PUBLIC

    def test_a_comparison_names_what_it_compared(self) -> None:
        result = _run(
            ComparisonAgent(),
            {
                "companies": [OTHER],
                "analyses": {OTHER: {"has_sso": True}, TARGET: {"has_sso": False}},
            },
        )
        assert any("companies compared across" in note for note in result.payload.notes)


class TestOpportunityAgent:
    """An opportunity exists only where a gap is backed by evidence."""

    def test_no_comparison_means_no_opportunity(self) -> None:
        result = _run(OpportunityAgent(), {})
        assert result.status is AgentStatus.FAILED
        assert "comparison matrix" in result.errors[0]

    def test_an_evidenced_gap_becomes_an_opportunity(self) -> None:
        result = _run(
            OpportunityAgent(),
            {
                "comparison": _matrix(),
                "evidence_by_company": {OTHER: {"has_free_tier": ["s1"]}},
            },
        )
        assert result.status is AgentStatus.OK
        opportunities = result.payload
        assert len(opportunities) == 1
        assert opportunities[0].evidence == ["s1"]
        assert OTHER in opportunities[0].affected_competitors
        assert "free tier" in opportunities[0].opportunity

    def test_a_gap_with_no_evidence_is_never_offered(self) -> None:
        """The competitor is known to have the feature, but nothing retrieved shows it."""
        result = _run(
            OpportunityAgent(),
            {"comparison": _matrix(), "evidence_by_company": {OTHER: {}}},
        )
        assert result.status is AgentStatus.PARTIAL
        assert result.payload is None
        assert "no observed gap was backed by evidence" in result.warnings[0]

    def test_a_gap_with_no_evidence_is_counted_so_the_gap_is_visible(self) -> None:
        result = _run(
            OpportunityAgent(),
            {"comparison": _matrix(), "evidence_by_company": {OTHER: {}}},
        )
        assert result.status is AgentStatus.PARTIAL

    def test_evidence_from_another_company_does_not_back_the_gap(self) -> None:
        result = _run(
            OpportunityAgent(),
            {
                "comparison": _matrix(),
                "evidence_by_company": {TARGET: {"has_free_tier": ["s1"]}},
            },
        )
        assert result.status is AgentStatus.PARTIAL

    def test_a_field_the_target_has_is_not_a_gap(self) -> None:
        result = _run(
            OpportunityAgent(),
            {
                "comparison": _matrix(),
                "evidence_by_company": {OTHER: {"has_sso": ["s1"]}},
            },
        )
        assert result.status is AgentStatus.PARTIAL

    def test_the_opportunity_names_the_observation_and_the_evidence(self) -> None:
        result = _run(
            OpportunityAgent(),
            {
                "comparison": _matrix(),
                "evidence_by_company": {OTHER: {"has_free_tier": ["s1"]}},
            },
        )
        opportunity = result.payload[0]
        assert "no publicly available value for the target" in opportunity.reasoning
        assert "the run retrieved" in opportunity.reasoning
        assert opportunity.evidence, "the schema refuses an opportunity with no evidence"

    def test_the_opportunity_agent_cannot_build_an_unevidenced_opportunity(self) -> None:
        """The rule is a property of the data, not only of this agent."""
        with pytest.raises(ValueError, match="evidence"):
            Opportunity(opportunity="do something", reasoning="because", evidence=[])


class TestEvidenceAgent:
    """Verification failures are reported, never swallowed."""

    def test_a_claim_whose_quote_is_in_the_page_is_attached(self) -> None:
        result = _run(
            EvidenceAgent(),
            {
                "sources": [_source()],
                "claims": [
                    {
                        "claim": "Globex offers a free tier",
                        "quote": "A free tier is available",
                        "source_id": "s1",
                        "claim_type": ClaimType.OBSERVED,
                        "confidence": Confidence.HIGH,
                    }
                ],
            },
        )
        assert result.status is AgentStatus.OK
        assert len(result.payload) == 1
        assert result.evidence_ids

    def test_a_claim_citing_a_page_the_run_never_retrieved_is_refused(self) -> None:
        """Search-snippet and invented citations are the failure this whole system exists to stop."""
        result = _run(
            EvidenceAgent(),
            {
                "sources": [_source()],
                "claims": [
                    {
                        "claim": "Globex is the market leader",
                        "quote": "market leader",
                        "source_id": "s2",
                    }
                ],
            },
        )
        assert result.status is AgentStatus.FAILED
        assert "did not retrieve" in result.warnings[0]

    def test_a_quote_that_is_not_in_the_page_is_refused(self) -> None:
        result = _run(
            EvidenceAgent(),
            {
                "sources": [_source()],
                "claims": [
                    {
                        "claim": "Globex serves 40,000 customers",
                        "quote": "40,000 customers",
                        "source_id": "s1",
                    }
                ],
            },
        )
        assert result.status is AgentStatus.FAILED
        assert result.payload is None

    def test_every_rejection_is_named_not_just_the_first(self) -> None:
        result = _run(
            EvidenceAgent(),
            {
                "sources": [_source()],
                "claims": [
                    {"claim": "one", "quote": "not in page", "source_id": "s1"},
                    {"claim": "two", "quote": "also not in page", "source_id": "s1"},
                    {
                        "claim": "Globex offers a free tier",
                        "quote": "A free tier is available",
                        "source_id": "s1",
                    },
                ],
            },
        )
        assert result.status is AgentStatus.PARTIAL
        warning = result.warnings[0]
        assert "2 claim(s) could not be verified" in warning
        assert "'one'" in warning and "'two'" in warning

    def test_a_claim_with_no_quote_is_refused(self) -> None:
        result = _run(
            EvidenceAgent(),
            {"sources": [_source()], "claims": [{"claim": "x", "source_id": "s1"}]},
        )
        assert result.status is AgentStatus.FAILED
        assert "no quote to verify" in result.warnings[0]

    def test_an_untyped_claim_is_reported_not_observed(self) -> None:
        """An unstated claim type must not license a statistic later."""
        result = _run(
            EvidenceAgent(),
            {
                "sources": [_source()],
                "claims": [
                    {
                        "claim": "Globex offers a free tier",
                        "quote": "A free tier is available",
                        "source_id": "s1",
                    }
                ],
            },
        )
        assert result.payload[0].claim_type is ClaimType.REPORTED

    def test_a_source_without_retrieved_text_is_not_registered(self) -> None:
        """A source with no text can never verify a quote, so it is not a source."""
        snippet = Source(
            id="s9", url="https://x.example", source_type=SourceType.SEARCH_SNIPPET
        )
        result = _run(
            EvidenceAgent(),
            {
                "sources": [snippet],
                "claims": [
                    {"claim": "x", "quote": "anything", "source_id": "s9"}
                ],
            },
        )
        assert result.status is AgentStatus.FAILED

    def test_a_source_given_as_a_dict_is_not_guessed_into_a_source(self) -> None:
        result = _run(
            EvidenceAgent(),
            {
                "sources": [{"id": "s1", "url": "https://x.example"}],
                "claims": [{"claim": "x", "quote": "y", "source_id": "s1"}],
            },
        )
        assert result.status is AgentStatus.FAILED


class TestReportAssembly:
    """The report prints its gaps instead of hiding them."""

    def test_every_section_appears_even_when_it_is_empty(self) -> None:
        """A category that was never researched must be visible, not omitted."""
        bundle = _bundle()
        for name in REPORT_SECTIONS:
            assert SECTION_TITLES[name] in bundle.report_markdown

    def test_an_empty_section_says_so_rather_than_being_blank(self) -> None:
        markdown = _bundle().report_markdown
        assert "No public information was retrieved" in markdown

    def test_the_sentinel_reaches_the_rendered_document(self) -> None:
        assert NOT_PUBLIC in _bundle().report_markdown

    def test_the_comparison_table_shows_the_sentinel_not_a_blank(self) -> None:
        markdown = _bundle().report_markdown
        row = next(line for line in markdown.splitlines() if "Free tier" in line)
        assert NOT_PUBLIC in row
        assert "| |" not in row, "a blank cell reads as a real empty value"

    def test_the_sentinel_is_not_itself_treated_as_empty(self) -> None:
        """A cell saying 'not publicly available' has said something."""
        matrix = ComparisonMatrix(
            companies=[TARGET],
            dimensions=[
                ComparisonRow(
                    field="has_free_tier",
                    label="Free tier",
                    category="pricing",
                    cells={TARGET: ComparisonCell.unknown()},
                )
            ],
        )
        missing = ReportService().missing_sections(_bundle(comparison=matrix))
        assert "product_comparison" not in missing

    def test_a_matrix_with_no_rows_is_reported_as_an_empty_section(self) -> None:
        missing = ReportService().missing_sections(
            _bundle(comparison=ComparisonMatrix(companies=[TARGET, OTHER]))
        )
        assert "product_comparison" in missing

    def test_a_pipe_in_a_value_does_not_break_the_table(self) -> None:
        matrix = ComparisonMatrix(
            companies=[TARGET, OTHER],
            dimensions=[
                ComparisonRow(
                    field="differentiators",
                    label="Differentiators",
                    cells={
                        TARGET: ComparisonCell.known("a | b"),
                        OTHER: ComparisonCell.known("c"),
                    },
                )
            ],
        )
        markdown = _bundle(comparison=matrix).report_markdown
        row = next(line for line in markdown.splitlines() if "Differentiators" in line)
        assert "a \\| b" in row

    def test_a_cross_competitor_pattern_reaches_its_section(self) -> None:
        """Section 14 is populated from the run, not left empty by an unpopulated field."""
        pattern = CrossCompetitorPattern(
            pattern="Both publish a 14-day trial",
            companies=[OTHER],
            count=2,
            observation="Both pricing pages mention a 14-day trial",
            evidence=["s1"],
        )
        bundle = _bundle(cross_competitor_patterns=[pattern])
        assert "Both publish a 14-day trial" in bundle.report_markdown
        assert "cross_competitor_patterns" not in ReportService().missing_sections(bundle)

    def test_a_quality_report_reaches_the_bundle(self) -> None:
        quality = QualityReport(
            passed=True,
            checks_run=["citations", "numbers", "claims"],
            repair_rounds_used=0,
            repair_exhausted=False,
        )
        bundle = _bundle(quality_report=quality)
        assert bundle.quality_report is not None
        assert bundle.quality_report.passed is True
        assert "numbers" in bundle.quality_report.checks_run

    def test_missing_sections_are_named_in_the_result(self) -> None:
        result = asyncio.run(ReportAgent().run(AgentContext(company=TARGET, extras={})))
        assert any("report sections have no content" in w for w in result.warnings)

    def test_the_run_never_fails_only_because_sections_are_missing(self) -> None:
        result = asyncio.run(ReportAgent().run(AgentContext(company=TARGET, extras={})))
        assert result.status is not AgentStatus.FAILED

    def test_the_report_is_not_claimed_to_be_complete_when_a_step_failed(self) -> None:
        bundle = _bundle(failed_steps=["seo"])
        assert bundle.run_status is RunStatus.PARTIAL

    def test_a_run_with_no_sources_at_all_is_not_reported_as_complete(self) -> None:
        bundle = _bundle(failed_steps=["crawl"], sources=[])
        assert bundle.run_status is RunStatus.FAILED

    def test_the_competitor_is_not_presented_as_validated_when_it_was_not(self) -> None:
        """The report must not claim a check the run did not perform."""
        bundle = _bundle()
        assert bundle.competitors[0].validated is False
        assert "not validated by this run" in bundle.report_markdown

    def test_a_competitor_reads_as_a_name_not_a_record_repr(self) -> None:
        markdown = _bundle().report_markdown
        assert "name=" not in markdown
        assert "validated=False" not in markdown

    def test_a_stray_script_tag_cannot_escape_the_html(self) -> None:
        bundle = _bundle(observed_gaps=["<script>alert(1)</script>"])
        assert "<script>alert(1)</script>" not in bundle.report_html
        assert "&lt;script&gt;" in bundle.report_html

    def test_the_markdown_is_not_html_escaped(self) -> None:
        """The HTML is escaped once at emission; the Markdown is plain text."""
        bundle = _bundle(sources=[_source(title="Globex's official pricing")])
        assert "Globex's official pricing" in bundle.report_markdown
        assert "&#x27;" not in bundle.report_markdown

    def test_rendering_twice_does_not_nest_the_report_in_its_own_summary(self) -> None:
        service = ReportService()
        first = service.render(_bundle())
        second = service.render(first)
        assert second.report_markdown == first.report_markdown

    def test_rendering_is_idempotent(self) -> None:
        service = ReportService()
        bundle = _bundle()
        assert service.render(bundle).report_markdown == service.render(bundle).report_markdown


class TestEvidenceQualitySection:
    """The evidence section is counted from the run, not asserted by it."""

    def test_the_summary_states_the_runs_own_numbers(self) -> None:
        bundle = _bundle(sources=[_source()])
        assert "1 retrieved source(s)" in bundle.report_markdown

    def test_a_run_with_no_evidence_says_it_offered_no_opportunity(self) -> None:
        bundle = _bundle(opportunities=[], sources=[])
        assert "No opportunity is offered" in bundle.report_markdown

    def test_a_run_that_offered_opportunities_says_so(self) -> None:
        assert "opportunity/ies are offered" in _bundle().report_markdown

    def test_an_untyped_source_is_not_counted_as_primary(self) -> None:
        """'We did not record what kind of page this was' is not 'official site'."""
        untyped = Source(id="s2", url="https://x.example", text="body")
        quality = ReportService().quality(_bundle(sources=[untyped]))
        assert quality.primary_sources == 0
        assert any("no primary source" in note for note in quality.notes)

    def test_a_snippet_only_source_is_counted_as_such(self) -> None:
        snippet = Source(
            id="s2", url="https://x.example", source_type=SourceType.SEARCH_SNIPPET
        )
        quality = ReportService().quality(_bundle(sources=[snippet]))
        assert quality.snippet_only_sources == 1
        assert quality.primary_sources == 0

    def test_evidence_given_as_objects_is_still_counted(self) -> None:
        """A step may hand over Evidence objects before they are dumped."""
        service = EvidenceService()
        source = service.register_fetched_many([_source()])[0]
        evidence = service.attach(
            "Globex offers a free tier",
            source=source,
            quote="A free tier is available",
            claim_type=ClaimType.OBSERVED,
            confidence=Confidence.HIGH,
        )
        bundle = _bundle(evidence=[evidence.model_dump()])
        assert ReportService().quality(bundle).major_claims == 1

    def test_the_confidence_distribution_is_counted_from_the_evidence(self) -> None:
        service = EvidenceService()
        source = service.register_fetched_many([_source()])[0]
        evidence = service.attach(
            "Globex offers a free tier",
            source=source,
            quote="A free tier is available",
            claim_type=ClaimType.OBSERVED,
            confidence=Confidence.HIGH,
        )
        quality = ReportService().quality(_bundle(evidence=[evidence.model_dump()]))
        assert quality.confidence_distribution.get("high") == 1


def _bundle(**overrides) -> ReportBundle:
    """A report assembled from a small, fully-evidenced run."""
    extras: dict = {
        "run_id": "r1",
        "competitors": [OTHER],
        "comparison": _matrix(),
        "opportunities": [
            Opportunity(
                opportunity=f"{OTHER} publishes a free tier where the target does not",
                reasoning="the comparison shows the gap and the evidence shows the feature",
                evidence=["s1"],
                affected_competitors=[OTHER],
                category="pricing",
            )
        ],
        "observed_gaps": ["No published pricing for the target"],
        "sources": [_source()],
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    extras.update(overrides)
    return asyncio.run(
        ReportAgent().run(AgentContext(company=TARGET, extras=extras))
    ).payload


def test_the_pricing_evidence_helper_drops_empty_ids() -> None:
    """An empty string is not an id, and indexing by one would silently drop the field."""
    from app.agents.opportunity import pricing_evidence

    analysis = PricingAnalysis(company=OTHER, evidence=["s1", "", "s2"])
    assert pricing_evidence(analysis) == ["s1", "s2"]


def test_the_fact_checker_and_supervisor_stay_importable_alongside_them() -> None:
    """Task 11's agents coexist, so the registry can list them all."""
    from app.agents.fact_checker import FactCheckerAgent
    from app.agents.supervisor import Supervisor

    assert FactCheckerAgent().name == "fact_checker"
    assert Supervisor is not None
    assert ClaimVerdict.SUPPORTED
