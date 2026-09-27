"""The last gate before a report reaches a reader.

Everything upstream in this system exists to stop an unsupported claim entering
the report. This stage is what makes that a property of the finished document
rather than a property of the code that produced it: it reads the report as a
reader would and asks, of every number and every claim in it, "what is this
standing on?"

The checks are deliberately mechanical. A quality stage that calls a model to
judge its own output is a model agreeing with itself, and its verdict would be
exactly as reliable as the generation that needed checking. So
:class:`QualityService` compares text against evidence it can point at, and
:class:`FactCheckerAgent` applies stated rules to stated evidence.

**Repair is bounded and it gives up out loud.** ``MAX_REPORT_REPAIR_ROUNDS`` is a
hard ceiling, and exceeding it sets ``repair_exhausted`` rather than looping. A
repair that could run forever would be a worse failure than the problem it fixes:
a run that hangs produces nothing, and a run that quietly drops a section
produces a report that lies about its own completeness.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.agents.fact_checker import (
    FactCheckerAgent,
    superlatives_in,
    unsupported_numeric_tokens,
)
from app.agents.supervisor import Supervisor
from app.core.constants import ClaimType, ClaimVerdict, Confidence, RunStatus, SourceType
from app.schemas.common import NOT_PUBLIC
from app.schemas.company import CompanyProfile
from app.schemas.comparison import ComparisonMatrix
from app.schemas.evidence import Evidence, Source
from app.schemas.finding import (
    ClaimVerdictRecord,
    Finding,
    QualityIssue,
    QualityReport,
)
from app.schemas.market import MarketAnalysis, Opportunity
from app.schemas.pricing import PricingAnalysis
from app.schemas.report import (
    REPORT_SECTIONS,
    ReportBundle,
    ReportMetadata,
)
from app.services.quality_service import MAX_REPORT_REPAIR_ROUNDS, QualityService
from app.utils.text import numeric_tokens

NOW = datetime.now(UTC)


def _source(id: str = "s1", **overrides) -> Source:
    defaults = {
        "id": id,
        "url": f"https://acme.example/{id}",
        "source_type": SourceType.OFFICIAL_SITE,
        "text": "Starter plans start at $10 per month.",
        "retrieved_at": NOW,
    }
    return Source(**{**defaults, **overrides})


def _evidence(
    id: str = "e1",
    *,
    claim: str = "Acme's starter plan costs $10 per month",
    claim_type: ClaimType = ClaimType.OBSERVED,
    confidence: Confidence = Confidence.HIGH,
    source_id: str = "s1",
    evidence_text: str = "Starter plans start at $10 per month.",
    published_at: datetime | None = None,
    conflicts_with: list[str] | None = None,
) -> Evidence:
    return Evidence(
        id=id,
        claim=claim,
        source_id=source_id,
        source_url=f"https://acme.example/{source_id}",
        evidence_text=evidence_text,
        claim_type=claim_type,
        confidence=confidence,
        company="Acme",
        claim_key=claim,
        published_at=published_at,
        conflicts_with=conflicts_with or [],
    )


def _metadata() -> ReportMetadata:
    return ReportMetadata(
        run_id="run-1",
        company="Acme",
        research_started_at=NOW,
        research_completed_at=NOW,
    )


def _report(**overrides) -> ReportBundle:
    """A minimal report, with a passing quality baseline unless overridden."""
    defaults = {
        "metadata": _metadata(),
        "target_company": CompanyProfile(name="Acme", evidence=["s1"]),
        "pricing_comparison": {
            "Acme": PricingAnalysis(
                company="Acme", pricing_model="per seat", evidence=["s1"]
            )
        },
        "sources": [_source()],
        "evidence": [_evidence().model_dump()],
    }
    return ReportBundle(**{**defaults, **overrides})


def _service() -> QualityService:
    return QualityService()


def _issues(report: QualityReport) -> set[str]:
    return {issue.code for issue in report.critical_issues + report.warnings}


class TestUnevidencedStatistics:
    """The plan's fourth required failure."""

    def test_a_number_in_the_report_with_no_observed_evidence_is_flagged(self) -> None:
        report = _report(
            observed_gaps=["Acme lost 45% of its enterprise customers last year."],
            evidence=[],
        )
        result = _service().check(report)
        assert "unevidenced_statistic" in _issues(result)
        assert result.passed is False

    def test_the_issue_is_critical_not_a_warning(self) -> None:
        """A report that prints an invented statistic is not fit to be sent."""
        report = _report(observed_gaps=["Revenue grew 300%."], evidence=[])
        result = _service().check(report)
        severities = {i.severity for i in result.critical_issues}
        assert "critical" in severities

    def test_the_issue_names_the_statistic(self) -> None:
        report = _report(observed_gaps=["Acme serves 12,000 customers."], evidence=[])
        issue = next(
            i for i in _service().check(report).critical_issues
            if i.code == "unevidenced_statistic"
        )
        assert "12" in issue.message or "12000" in issue.message or "12,000" in issue.message

    def test_a_statistic_backed_by_observed_evidence_is_not_flagged(self) -> None:
        report = _report(
            observed_gaps=["Acme serves 12,000 customers."],
            evidence=[_evidence(evidence_text="Acme serves 12,000 customers.").model_dump()],
        )
        assert "unevidenced_statistic" not in _issues(_service().check(report))

    def test_inferred_evidence_does_not_license_a_statistic(self) -> None:
        """An inference is a reading of a source, not the number itself."""
        report = _report(
            observed_gaps=["Acme serves 12,000 customers."],
            evidence=[
                _evidence(
                    claim_type=ClaimType.INFERRED,
                    confidence=Confidence.MEDIUM,
                    evidence_text="Acme serves many customers.",
                ).model_dump()
            ],
        )
        assert "unevidenced_statistic" in _issues(_service().check(report))

    def test_an_inferred_statistic_in_the_evidence_is_still_flagged(self) -> None:
        report = _report(
            observed_gaps=["Acme serves 12,000 customers."],
            evidence=[
                _evidence(
                    claim_type=ClaimType.INFERRED,
                    confidence=Confidence.MEDIUM,
                    evidence_text="Acme serves 12,000 customers.",
                ).model_dump()
            ],
        )
        assert "unevidenced_statistic" in _issues(_service().check(report))

    def test_the_markdown_body_is_checked_too(self) -> None:
        report = _report(
            report_markdown="# Report\n\nAcme has 900 employees.\n",
            evidence=[],
        )
        assert "unevidenced_statistic" in _issues(_service().check(report))

    def test_a_year_is_not_treated_as_a_statistic(self) -> None:
        """Otherwise every dated sentence in the report is a critical issue."""
        report = _report(
            observed_gaps=["Acme was founded in 2019 and rebranded in 2021."],
            evidence=[],
        )
        result = _service().check(report)
        assert "unevidenced_statistic" not in _issues(result)

    def test_a_currency_figure_is_a_statistic(self) -> None:
        """The counterpart to the year rule: a real figure is still caught."""
        report = _report(observed_gaps=["Acme raised $4M in Series A."], evidence=[])
        assert "unevidenced_statistic" in _issues(_service().check(report))

    def test_a_percentage_is_a_statistic(self) -> None:
        report = _report(observed_gaps=["Acme grew 40% year over year."], evidence=[])
        assert "unevidenced_statistic" in _issues(_service().check(report))

    def test_a_numbered_list_item_is_not_a_statistic(self) -> None:
        report = _report(observed_gaps=["1. Acme is a cloud vendor."], evidence=[])
        assert "unevidenced_statistic" not in _issues(_service().check(report))

    def test_a_section_number_is_not_a_statistic(self) -> None:
        report = _report(report_markdown="## 7. Pricing\n\nNo prices are public.\n")
        assert "unevidenced_statistic" not in _issues(_service().check(report))

    def test_a_url_is_not_a_statistic(self) -> None:
        report = _report(
            observed_gaps=["See https://acme.example/pricing/v2 for details."],
            evidence=[],
        )
        assert "unevidenced_statistic" not in _issues(_service().check(report))


class TestCitationIntegrity:
    def test_a_finding_citing_an_unknown_evidence_id_is_flagged(self) -> None:
        report = _report(
            findings=[
                Finding(
                    id="f1",
                    company="Acme",
                    category="pricing",
                    statement="Acme charges $10 per seat",
                    evidence_ids=["never-attached"],
                )
            ]
        )
        assert "dangling_citation" in _issues(_service().check(report))

    def test_a_finding_citing_real_evidence_is_not_flagged(self) -> None:
        report = _report(
            findings=[
                Finding(
                    id="f1",
                    company="Acme",
                    category="pricing",
                    statement="Acme charges $10 per seat",
                    evidence_ids=["e1"],
                )
            ]
        )
        assert "dangling_citation" not in _issues(_service().check(report))

    def test_a_report_with_no_sources_at_all_is_flagged(self) -> None:
        result = _service().check(_report(sources=[], evidence=[]))
        assert "no_sources" in _issues(result)

    def test_a_report_with_no_evidence_at_all_is_flagged(self) -> None:
        assert "no_evidence" in _issues(_service().check(_report(evidence=[])))

    def test_a_clean_report_passes(self) -> None:
        result = _service().check(_report())
        assert result.passed is True
        assert result.critical_issues == []

    def test_a_report_with_no_claims_at_all_is_still_flagged(self) -> None:
        """An empty report is not a clean report."""
        result = _service().check(_report(sources=[], evidence=[]))
        assert result.passed is False


class TestChecksAreRecorded:
    def test_the_report_names_the_checks_that_ran(self) -> None:
        result = _service().check(_report())
        assert "unevidenced_statistic" in result.checks_run

    def test_checking_is_side_effect_free(self) -> None:
        """Quality control must not make the report worse by looking at it."""
        report = _report(observed_gaps=["Acme grew 40%."], evidence=[])
        before = report.model_dump()
        _service().check(report)
        assert report.model_dump() == before


class TestRepair:
    """The plan's third required failure."""

    def test_repair_is_attempted_exactly_twice_then_gives_up(self) -> None:
        """Required: ``MAX_REPORT_REPAIR_ROUNDS=2``, then ``repair_exhausted``."""
        rounds: list[int] = []

        def stubborn(report: ReportBundle, round_number: int) -> None:
            rounds.append(round_number)

        report = _report(observed_gaps=["Acme grew 40% last year."], evidence=[])
        result = _service().repair(report, repair=stubborn)
        assert rounds == [1, 2]
        assert result.repair_exhausted is True
        assert result.repair_rounds_used == MAX_REPORT_REPAIR_ROUNDS

    def test_repair_stops_as_soon_as_the_report_is_clean(self) -> None:
        """A clean report must not be rewritten twice for the sake of the count."""
        rounds: list[int] = []

        def fixable(report: ReportBundle, round_number: int) -> None:
            rounds.append(round_number)
            report.observed_gaps = []

        report = _report(observed_gaps=["Acme serves 12,000 customers."])
        result = _service().repair(report, repair=fixable)
        assert rounds == [1]
        assert result.repair_exhausted is False
        assert result.passed is True

    def test_repair_defaults_to_the_built_in_actions(self) -> None:
        report = _report(observed_gaps=["Acme grew 40% last year."], evidence=[])
        _service().repair(report)
        assert report.quality_report is not None

    def test_repair_never_exceeds_the_ceiling_even_if_asked_to(self) -> None:
        """A caller passing a larger budget does not get one; the constant wins."""
        rounds: list[int] = []
        report = _report(observed_gaps=["Acme grew 40% last year."], evidence=[])
        _service().repair(report, rounds=99, repair=lambda r, n: rounds.append(n))
        assert len(rounds) == MAX_REPORT_REPAIR_ROUNDS

    def test_repair_of_an_already_clean_report_does_nothing(self) -> None:
        report = _report()
        result = _service().repair(report)
        assert result.repair_rounds_used == 0
        assert result.repair_exhausted is False

    def test_repair_records_the_issue_it_could_not_fix(self) -> None:
        """Exhausted is a state a reader or operator can act on."""
        report = _report(observed_gaps=["Acme grew 40% last year."], evidence=[])
        result = _service().repair(report, repair=lambda r, n: None)
        assert result.critical_issues

    def test_repair_attaches_its_outcome_to_the_report(self) -> None:
        report = _report(observed_gaps=["Acme grew 40% last year."], evidence=[])
        _service().repair(report, repair=lambda r, n: None)
        assert report.quality_report is not None
        assert report.quality_report.repair_exhausted is True


class TestFactChecker:
    """The plan's second required failure, part one: the ruling."""

    def test_a_market_leader_claim_with_no_evidence_is_unsupported(self) -> None:
        """Required: "is the market leader" has no evidence behind it."""
        checker = FactCheckerAgent()
        verdicts = checker.verify(
            [{"claim_id": "c1", "claim": "Acme is the market leader", "evidence_ids": []}],
            {"c1": []},
        )
        assert verdicts[0].verdict is ClaimVerdict.UNSUPPORTED
        assert verdicts[0].reason

    def test_an_unsupported_claim_has_its_confidence_cut(self) -> None:
        """The downgrade is a side effect, not only a label on the record."""
        checker = FactCheckerAgent()
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme is the market leader",
              "evidence_ids": [], "confidence": "high"}],
            {"c1": []},
        )[0]
        assert verdict.confidence_before is Confidence.HIGH
        assert verdict.confidence_after is Confidence.LOW

    def test_a_claim_backed_by_observed_evidence_is_supported(self) -> None:
        checker = FactCheckerAgent()
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme's starter plan costs $10",
              "evidence_ids": ["e1"]}],
            {"c1": [_evidence()]},
        )[0]
        assert verdict.verdict is ClaimVerdict.SUPPORTED
        assert verdict.supporting_evidence == ["e1"]

    def test_a_superlative_claim_is_an_overstatement_not_support(self) -> None:
        """One page cannot make a company the market leader; the claim exceeds it."""
        checker = FactCheckerAgent()
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme is the best vendor in the market",
              "evidence_ids": ["e1"]}],
            {"c1": [_evidence()]},
        )[0]
        assert verdict.verdict is ClaimVerdict.OVERSTATEMENT

    def test_a_number_absent_from_the_evidence_is_an_overstatement(self) -> None:
        checker = FactCheckerAgent()
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme serves 500,000 customers",
              "evidence_ids": ["e1"]}],
            {"c1": [_evidence(evidence_text="Acme serves some customers.")]},
        )[0]
        assert verdict.verdict is ClaimVerdict.OVERSTATEMENT

    def test_evidence_that_disagrees_with_itself_is_contradictory(self) -> None:
        checker = FactCheckerAgent()
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme's starter plan costs $10",
              "evidence_ids": ["e1", "e2"]}],
            {"c1": [_evidence(), _evidence(id="e2", conflicts_with=["e1"])]},
        )[0]
        assert verdict.verdict is ClaimVerdict.CONTRADICTORY
        assert verdict.contradicting_evidence

    def test_old_evidence_is_stale(self) -> None:
        checker = FactCheckerAgent(max_age_days=90)
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme's starter plan costs $10",
              "evidence_ids": ["e1"]}],
            {"c1": [_evidence(published_at=NOW - timedelta(days=400))]},
        )[0]
        assert verdict.verdict is ClaimVerdict.STALE

    def test_evidence_whose_contradiction_is_flagged_never_reads_as_supported(self) -> None:
        """The order of the rules is what stops a conflict hiding as support."""
        checker = FactCheckerAgent()
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme is the market leader",
              "evidence_ids": ["e1"]}],
            {"c1": [_evidence(conflicts_with=["e2"])]},
        )[0]
        assert verdict.verdict is not ClaimVerdict.SUPPORTED

    def test_a_claim_naming_an_evidence_id_that_does_not_exist_is_unsupported(self) -> None:
        checker = FactCheckerAgent()
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme's plan costs $10",
              "evidence_ids": ["never-attached"]}],
            {"c1": [_evidence()]},
        )[0]
        assert verdict.verdict is ClaimVerdict.UNSUPPORTED

    def test_every_verdict_carries_a_reason(self) -> None:
        checker = FactCheckerAgent()
        for claim in (
            "Acme is the market leader",
            "Acme's starter plan costs $10",
            "Acme serves 900,000 users",
        ):
            verdict = checker.verify(
                [{"claim_id": "c1", "claim": claim, "evidence_ids": ["e1"]}],
                {"c1": [_evidence()]},
            )[0]
            assert verdict.reason, claim

    def test_a_claim_with_no_id_is_still_checked(self) -> None:
        """An unlabelled claim is still a sentence in the report."""
        checker = FactCheckerAgent()
        verdict = checker.verify([{"claim": "Acme is the market leader"}], {})[0]
        assert verdict.verdict is ClaimVerdict.UNSUPPORTED
        assert verdict.claim_id

    def test_a_claim_the_index_knows_nothing_about_is_unsupported(self) -> None:
        checker = FactCheckerAgent()
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme is the market leader"}], {}
        )[0]
        assert verdict.verdict is ClaimVerdict.UNSUPPORTED

    def test_verifying_nothing_returns_nothing(self) -> None:
        assert FactCheckerAgent().verify([], {}) == []

    def test_confidence_is_never_raised_by_checking(self) -> None:
        checker = FactCheckerAgent()
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme's plan costs $10",
              "evidence_ids": ["e1"], "confidence": "low"}],
            {"c1": [_evidence()]},
        )[0]
        assert verdict.confidence_after is not Confidence.HIGH

    def test_two_independent_primary_sources_support_a_high_confidence_claim(self) -> None:
        checker = FactCheckerAgent()
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme's starter plan costs $10",
              "evidence_ids": ["e1", "e2"], "confidence": "high"}],
            {"c1": [
                _evidence(),
                _evidence(id="e2", source_id="s2", evidence_text="Starts at $10/month."),
            ]},
        )[0]
        assert verdict.verdict is ClaimVerdict.SUPPORTED
        assert verdict.confidence_after is Confidence.HIGH

    def test_a_single_source_cannot_hold_high_confidence(self) -> None:
        checker = FactCheckerAgent()
        verdict = checker.verify(
            [{"claim_id": "c1", "claim": "Acme's starter plan costs $10",
              "evidence_ids": ["e1"], "confidence": "high"}],
            {"c1": [_evidence()]},
        )[0]
        assert verdict.confidence_after is Confidence.MEDIUM

    def test_the_superlative_scanner_finds_a_leader_claim(self) -> None:
        assert superlatives_in("Acme is the market leader in CRM") == ["market leader"]

    def test_the_superlative_scanner_finds_nothing_in_a_factual_sentence(self) -> None:
        assert superlatives_in("Acme's starter plan costs $10 per month") == []

    def test_the_numeric_scanner_finds_a_number_the_evidence_lacks(self) -> None:
        assert "500,000" in unsupported_numeric_tokens(
            "Acme serves 500,000 customers", "Acme serves many customers."
        )

    def test_the_numeric_scanner_accepts_a_number_the_evidence_carries(self) -> None:
        assert unsupported_numeric_tokens(
            "Acme serves 12,000 customers", "Acme serves 12,000 customers."
        ) == []


class TestNumbersWrittenWithAUnitOrARate:
    """A price is the one thing this agent must never invent.

    Found by reviewing the whole branch before publishing: the number pattern
    ended in ``(?![\\w/])`` so that a figure followed by ``/`` matched nothing at
    all. ``$50/mo`` - the canonical way to write a price - tokenised to nothing,
    and ``2.4x`` backtracked to the bare figure ``2``. Both then passed the gate
    against evidence that did not contain them, so a model could invent a
    competitor's price and be certified as evidence-backed. The tests below are
    the ones that would have caught it; every one of them passed before.
    """

    def test_a_price_written_per_month_is_recognised_as_a_figure(self) -> None:
        assert "$50/mo" in numeric_tokens("Acme is $50/mo versus Beta $80/mo")

    def test_an_invented_price_is_flagged(self) -> None:
        """The headline case: the claim says $50/mo, the source says $20/mo."""
        assert unsupported_numeric_tokens(
            "Acme is $50/mo", "Beta lists a plan at $20 per month."
        ) == ["$50/mo"]

    def test_an_annual_price_is_recognised_and_flagged(self) -> None:
        assert unsupported_numeric_tokens(
            "Acme is priced at $99/year", "No pricing is published on this page."
        ) == ["$99/year"]

    def test_a_multiple_is_not_degraded_to_its_first_digits(self) -> None:
        """``2.4x`` used to match as ``2``, which the evidence usually contains."""
        assert numeric_tokens("Acme has 2.4x the market share") == ["2.4x"]

    def test_an_invented_multiple_is_flagged(self) -> None:
        assert unsupported_numeric_tokens(
            "Acme has 2.4x the market share of Beta",
            "Beta was founded in 2019 and has 2 offices.",
        ) == ["2.4x"]

    def test_a_price_written_per_user_is_recognised(self) -> None:
        assert "$10 per user" in numeric_tokens("costs $10 per user per month")

    def test_a_large_user_count_is_still_found(self) -> None:
        """The commas must not be mistaken for part of a path or a unit."""
        assert unsupported_numeric_tokens(
            "Acme serves 1,000,000 customers", "Beta says it serves 400,000 customers."
        ) == ["1,000,000"]

    def test_a_rating_out_of_five_is_still_a_figure(self) -> None:
        assert "4.9" in numeric_tokens("rated 4.9/5 by reviewers")

    def test_the_same_figure_written_two_ways_is_still_matched(self) -> None:
        """``$50/mo`` in a claim must match ``$50 per month`` in a quote."""
        assert unsupported_numeric_tokens(
            "Acme is $50/mo", "Acme lists $50 per month."
        ) == []

    def test_a_currency_symbol_in_the_claim_matches_a_bare_number(self) -> None:
        """The comparison is on the figure, so ``$10`` matches ``10`` in a quote.

        The old comparison ran on the whole matched token, currency included, so
        this honest case was flagged - the rule that only exists to catch
        invented figures was raising the alarm on correct ones.
        """
        assert unsupported_numeric_tokens(
            "Acme costs $10", "The page says 10 for the starter tier."
        ) == []

    def test_thousands_separators_do_not_hide_a_mismatch(self) -> None:
        assert unsupported_numeric_tokens(
            "serves 1,000,000 users", "serves 1000000 customers"
        ) == []

    @pytest.mark.parametrize(
        "text",
        [
            "node20 and k8s and x86 are not figures",
            "a 5GB disk",
            "Founded in 2019 in Dublin",
        ],
    )
    def test_identifiers_specs_and_years_are_still_not_figures(self, text: str) -> None:
        """The exclusions the pattern existed for must survive the fix."""
        assert numeric_tokens(text) == []


class TestRunTelemetryIsNotReaderProse:
    """A report that admits a step failed must still be able to pass.

    Found while reviewing the branch before publishing. ``warnings`` and
    ``failed_steps`` were in ``PROSE_FIELDS``, so every digit in them became a
    *critical* unevidenced statistic - "step 3 of 9" and "4 sources for 2
    competitors". Those are facts about this execution, which the run is its own
    evidence for, and no web citation could support them. Worse, the built-in
    repairs only remove refuted claims, so the issue was permanent: every run
    that reported a failure or a count was unpassable, and the quality gate's
    verdict carried no information.
    """

    def test_a_reported_step_failure_does_not_block_the_report(self) -> None:
        report = _report(
            failed_steps=["pricing analysis (step 3 of 9) failed: timeout"]
        )
        outcome = QualityService().check(report)
        assert outcome.passed, [issue.message for issue in outcome.critical_issues]

    def test_a_source_count_in_a_warning_does_not_block_the_report(self) -> None:
        report = _report(warnings=["Retrieved 4 sources for 2 competitors"])
        outcome = QualityService().check(report)
        assert outcome.passed, [issue.message for issue in outcome.critical_issues]

    def test_no_unevidenced_statistic_is_raised_against_telemetry(self) -> None:
        report = _report(
            warnings=["Retrieved 4 sources for 2 competitors"],
            failed_steps=["pricing analysis (step 3 of 9) failed: timeout"],
        )
        outcome = QualityService().check(report)
        codes = {issue.code for issue in outcome.critical_issues}
        assert "unevidenced_statistic" not in codes

    def test_real_prose_is_still_checked(self) -> None:
        """The exemption is for telemetry, not a loophole for the narrative."""
        report = _report(observed_gaps=["Acme serves 900,000 customers"])
        outcome = QualityService().check(report)
        assert "unevidenced_statistic" in {i.code for i in outcome.critical_issues}


class TestReportRepairRemovesTheSentence:
    """The plan's second required failure, part two: the effect on the report."""

    def test_the_repaired_report_no_longer_contains_the_claim(self) -> None:
        """Required: the sentence is gone, not merely annotated."""
        claim = "Acme is the market leader in customer relationship management"
        record = ClaimVerdictRecord(
            claim_id="c1",
            claim=claim,
            verdict=ClaimVerdict.UNSUPPORTED,
            reason="no evidence retrieved for this claim",
            confidence_after=Confidence.LOW,
        )
        report = _report(
            observed_gaps=[claim, "Acme is a cloud vendor for mid-market teams."],
            claim_verdicts=[record],
        )
        _service().repair(report)
        repaired = report
        assert claim not in (repaired.report_markdown or "")
        for gap in repaired.observed_gaps:
            assert "market leader" not in gap
        assert repaired.observed_gaps

    def test_the_surviving_claim_is_kept(self) -> None:
        record = ClaimVerdictRecord(
            claim_id="c1",
            claim="Acme is the market leader",
            verdict=ClaimVerdict.UNSUPPORTED,
            reason="no evidence",
        )
        report = _report(
            observed_gaps=["Acme is the market leader", "Acme sells a CRM."],
            claim_verdicts=[record],
        )
        _service().repair(report)
        repaired = report
        assert "Acme sells a CRM." in repaired.observed_gaps

    def test_the_markdown_body_loses_the_unverifiable_sentence_too(self) -> None:
        report = _report(
            report_markdown="# Gaps\n\nAcme is the market leader.\n\nAcme sells a CRM.\n",
            claim_verdicts=[
                ClaimVerdictRecord(
                    claim_id="c1",
                    claim="Acme is the market leader",
                    verdict=ClaimVerdict.UNSUPPORTED,
                    reason="no evidence",
                )
            ],
        )
        _service().repair(report)
        repaired = report
        assert "market leader" not in (repaired.report_markdown or "")
        assert "sells a CRM" in (repaired.report_markdown or "")

    def test_a_repaired_verdict_is_marked_as_repaired(self) -> None:
        report = _report(
            observed_gaps=["Acme is the market leader"],
            claim_verdicts=[
                ClaimVerdictRecord(
                    claim_id="c1",
                    claim="Acme is the market leader",
                    verdict=ClaimVerdict.UNSUPPORTED,
                    reason="no evidence",
                )
            ],
        )
        _service().repair(report)
        repaired = report
        assert repaired.claim_verdicts[0].repaired is True

    def test_a_supported_verdict_is_left_in_place(self) -> None:
        report = _report(
            observed_gaps=["Acme sells a CRM."],
            claim_verdicts=[
                ClaimVerdictRecord(
                    claim_id="c1",
                    claim="Acme sells a CRM.",
                    verdict=ClaimVerdict.SUPPORTED,
                    reason="observed on the vendor's own page",
                )
            ],
        )
        _service().repair(report)
        repaired = report
        assert repaired.observed_gaps == ["Acme sells a CRM."]
        assert repaired.claim_verdicts[0].repaired is False

    def test_repair_does_not_invent_a_replacement_sentence(self) -> None:
        """The failure mode of a repair that "softens" the claim instead of
        removing it: the softer version is still an assertion."""
        report = _report(
            observed_gaps=["Acme is the market leader"],
            claim_verdicts=[
                ClaimVerdictRecord(
                    claim_id="c1",
                    claim="Acme is the market leader",
                    verdict=ClaimVerdict.UNSUPPORTED,
                    reason="no evidence",
                    suggested_wording="Acme is a vendor in this market",
                )
            ],
        )
        _service().repair(report)
        repaired = report
        for gap in repaired.observed_gaps:
            assert "market leader" not in gap

    def test_repair_removes_a_finding_the_fact_checker_undermined(self) -> None:
        report = _report(
            findings=[
                Finding(
                    id="f1",
                    company="Acme",
                    category="market",
                    statement="Acme is the market leader",
                    evidence_ids=["e1"],
                )
            ],
            claim_verdicts=[
                ClaimVerdictRecord(
                    claim_id="f1",
                    claim="Acme is the market leader",
                    verdict=ClaimVerdict.UNSUPPORTED,
                    reason="no evidence",
                )
            ],
        )
        _service().repair(report)
        repaired = report
        assert all(f.id != "f1" for f in repaired.findings)

    def test_a_finding_with_no_verdict_is_not_removed(self) -> None:
        """Repair removes what was refuted, not what it did not examine."""
        report = _report(
            findings=[
                Finding(
                    id="f1",
                    company="Acme",
                    category="market",
                    statement="Acme sells a CRM",
                    evidence_ids=["e1"],
                )
            ]
        )
        _service().repair(report)
        repaired = report
        assert len(repaired.findings) == 1


class TestOpportunitiesRequireEvidence:
    def test_an_opportunity_cannot_be_constructed_without_evidence(self) -> None:
        """The schema's own docstring says this; make the schema enforce it."""
        with pytest.raises(ValueError, match="evidence"):
            Opportunity(opportunity="Offer a free tier", reasoning="Nobody offers one")

    def test_an_opportunity_with_evidence_is_fine(self) -> None:
        opportunity = Opportunity(
            opportunity="Offer a free tier",
            reasoning="No competitor publishes one",
            evidence=["e1"],
        )
        assert opportunity.evidence == ["e1"]

    def test_the_ceiling_is_two_rounds(self) -> None:
        assert MAX_REPORT_REPAIR_ROUNDS == 2


class TestSupervisor:
    def test_a_run_with_nothing_failed_is_completed(self) -> None:
        supervisor = Supervisor()
        supervisor.record_ok("pricing")
        assert supervisor.status is RunStatus.COMPLETED

    def test_one_failed_analyst_makes_the_run_partial_not_failed(self) -> None:
        """A report missing a section is far more useful than no report."""
        supervisor = Supervisor()
        supervisor.record_ok("pricing")
        supervisor.record_failure("review", "no sources retrieved")
        assert supervisor.status is RunStatus.PARTIAL

    def test_a_failed_target_identification_stops_the_run(self) -> None:
        """Everything downstream describes a company; without it there is none."""
        supervisor = Supervisor()
        supervisor.record_failure("company_identification", "no official domain")
        assert supervisor.status is RunStatus.FAILED
        assert supervisor.can_continue is False

    def test_a_partial_run_can_still_continue(self) -> None:
        supervisor = Supervisor()
        supervisor.record_failure("review", "no sources retrieved")
        assert supervisor.can_continue is True

    def test_every_failure_is_recorded_with_its_reason(self) -> None:
        supervisor = Supervisor()
        supervisor.record_failure("review", "no sources retrieved")
        assert supervisor.failed_steps == ["review: no sources retrieved"]

    def test_the_same_failure_is_not_recorded_twice(self) -> None:
        """A retried node that fails identically should not fill the list."""
        supervisor = Supervisor()
        supervisor.record_failure("review", "no sources retrieved")
        supervisor.record_failure("review", "no sources retrieved")
        assert len(supervisor.failed_steps) == 1

    def test_warnings_are_accumulated_and_make_the_run_partial(self) -> None:
        supervisor = Supervisor()
        supervisor.record_ok("pricing")
        supervisor.record_warning("pricing", "cited a source it did not retrieve")
        assert supervisor.status is RunStatus.PARTIAL
        assert supervisor.warnings

    def test_a_recovered_node_clears_its_failure(self) -> None:
        supervisor = Supervisor()
        supervisor.record_failure("review", "no sources retrieved")
        supervisor.record_ok("review")
        assert supervisor.failed_steps == []
        assert supervisor.status is RunStatus.COMPLETED

    def test_progress_is_the_fraction_of_optional_steps_that_ran(self) -> None:
        supervisor = Supervisor()
        supervisor.record_ok("pricing")
        supervisor.record_ok("product")
        assert supervisor.progress == 100

    def test_a_hard_failure_reports_zero_progress(self) -> None:
        supervisor = Supervisor()
        supervisor.record_failure("company_identification", "no official domain")
        assert supervisor.progress == 0

    def test_the_sentinel_is_never_reported_as_a_progress_value(self) -> None:
        """A report that says it is -1% complete has learned nothing."""
        supervisor = Supervisor()
        assert supervisor.progress >= 0


class TestMisc:
    def test_a_comparison_cell_sentinel_is_not_a_statistic(self) -> None:
        report = _report(
            product_comparison=ComparisonMatrix(
                companies=["Acme"], rows=[{"field": "price", "values": {"Acme": NOT_PUBLIC}}]
            ),
            evidence=[],
        )
        assert "unevidenced_statistic" not in _issues(_service().check(report))

    def test_an_empty_market_analysis_is_not_a_problem(self) -> None:
        report = _report(market_analysis=MarketAnalysis(), evidence=[])
        assert "missing_section" not in _issues(_service().check(report))

    def test_a_section_list_never_shrinks(self) -> None:
        assert len(REPORT_SECTIONS) == 19

    def test_quality_issue_defaults_to_a_warning(self) -> None:
        assert QualityIssue(code="x", message="y").severity == "warning"

    def test_a_quality_report_defaults_to_passing(self) -> None:
        assert QualityReport().passed is True
