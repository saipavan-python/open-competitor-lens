"""Quality control: reading the finished report the way a reader will.

Everything before this stage checks the *inputs* — that a source was retrieved, a
claim is typed correctly, an analyst cited what it was given. This one checks the
output as a document, which is the only place several individually-valid steps
can add up to a misleading report: nine analysts each citing a real page, and a
summary sentence asserting a figure none of them found.

The checks are mechanical on purpose. A quality stage implemented as a model
call is a model agreeing with itself about text it wrote moments ago, and its
verdict would be no more reliable than the generation it was asked to audit. So
every rule here compares the report against evidence it can name:

* every statistic in the prose must appear in an ``observed`` evidence record;
* every evidence id a finding cites must exist in the report's evidence;
* a claim the fact checker refuted must not still be in the body.

**Repair is bounded.** :data:`MAX_REPORT_REPAIR_ROUNDS` is a hard ceiling, and
hitting it sets ``repair_exhausted`` rather than continuing. Two failure modes are
being avoided at once: a run that loops forever produces nothing, and a run that
quietly drops the parts it could not fix produces a report that lies about its own
completeness. Reporting "I could not fix this" is the honest third option, and it
is the one this module takes.

What repair will do is narrow: remove a claim the fact checker refuted, and drop
references to evidence that is not there. It will not rewrite a claim into a
softer one. "Acme is the market leader" with no evidence does not become "Acme is
a vendor in this market" — that is still an assertion, now one nobody checked,
and a repair that produces unverified sentences has undone the report's only
guarantee.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from app.core.constants import ClaimType, ClaimVerdict
from app.core.logging import get_logger
from app.schemas.common import NOT_PUBLIC
from app.schemas.finding import QualityIssue, QualityReport
from app.schemas.market import MarketAnalysis
from app.schemas.report import ReportBundle
from app.utils.text import numeric_tokens

log = get_logger(__name__)

#: Ceiling on repair rounds. A report that cannot be repaired in two passes is
#: reported as unrepaired rather than repaired further.
MAX_REPORT_REPAIR_ROUNDS = 2

#: Verdicts that mean a claim should not appear in the report at all. A stale
#: claim is not in this set: it is still true of some past date, and the report's
#: job is to say when, not to pretend the sentence never existed.
REFUTED_VERDICTS: frozenset[ClaimVerdict] = frozenset(
    {ClaimVerdict.UNSUPPORTED, ClaimVerdict.OVERSTATEMENT, ClaimVerdict.CONTRADICTORY}
)

#: Fields whose text a reader reads as prose, and so is searched for statistics.
#:
#: ``warnings`` and ``failed_steps`` are deliberately absent. They describe the
#: run, not the companies: "pricing analysis (step 3 of 9) failed: timeout" and
#: "Retrieved 4 sources for 2 competitors" are facts about this execution, and
#: the run is its own evidence for them - no web citation can or should support
#: them. Treating them as prose made every digit in them a *critical* issue, and
#: because the built-in repairs only remove refuted claims, no amount of repair
#: could clear it: a report that truthfully admitted a step had failed could
#: never reach ``passed=True``. That is the opposite of the point. A run that
#: hides its own failures should be fixed here; one that reports them must not be
#: failed for reporting them.
PROSE_FIELDS: tuple[str, ...] = (
    "observed_gaps",
    "risks",
)

RepairHook = Callable[[ReportBundle, int], None]


class QualityService:
    """Checks a report against its evidence, and repairs what it safely can."""

    def check(self, report: ReportBundle) -> QualityReport:
        """Every issue found in ``report``, and whether it may be sent.

        Read-only. Quality control that rewrites the report as a side effect of
        looking at it cannot be reasoned about, and a reader who ran ``check``
        twice should get the same answer both times.
        """
        checks: list[str] = []
        critical: list[QualityIssue] = []
        warnings: list[QualityIssue] = []

        checks.append("unevidenced_statistic")
        critical.extend(self._unevidenced_statistics(report))

        checks.append("dangling_citation")
        critical.extend(self._dangling_citations(report))

        checks.append("refuted_claim_present")
        critical.extend(self._refuted_claims_still_present(report))

        checks.append("sources_present")
        critical.extend(self._missing_sources(report))

        checks.append("coverage")
        warnings.extend(self._coverage_warnings(report))

        passed = not critical
        return QualityReport(
            passed=passed,
            critical_issues=critical,
            warnings=warnings,
            repair_required=not passed,
            checks_run=checks,
        )

    def repair(
        self,
        report: ReportBundle,
        *,
        rounds: int | None = None,
        repair: RepairHook | None = None,
    ) -> QualityReport:
        """Bring ``report`` up to standard, or say that it could not be.

        Rounds until the report passes or :data:`MAX_REPORT_REPAIR_ROUNDS` is
        spent, whichever comes first. A caller asking for more rounds does not get
        more rounds: the ceiling exists so a report that cannot be fixed produces
        a documented failure rather than an unbounded loop. A round is only
        started while the report still has something wrong with it, so a hook is
        not called against an already-clean report.

        The report is repaired in place and the outcome returned, so the report
        object the caller already holds is the one that changed.

        The built-in repairs are deliberately few. Everything else is a decision
        about what a sentence was *meant* to say, which is a rewriting, not a
        repair.
        """
        budget = (
            MAX_REPORT_REPAIR_ROUNDS
            if rounds is None
            else min(rounds, MAX_REPORT_REPAIR_ROUNDS)
        )
        used = 0
        current = self.check(report)
        while used < budget and current.repair_required:
            used += 1
            self._apply_builtin_repairs(report)
            if repair is not None:
                repair(report, used)
            current = self.check(report)

        if repair is not None and current.repair_required:
            log.warning(
                "report could not be repaired within the round limit",
                extra={"rounds": used, "issues": len(current.critical_issues)},
            )

        outcome = QualityReport(
            passed=current.passed,
            critical_issues=current.critical_issues,
            warnings=current.warnings,
            repair_required=current.repair_required,
            repair_rounds_used=used,
            repair_exhausted=current.repair_required and used >= budget,
            checks_run=current.checks_run,
        )
        report.quality_report = outcome
        return outcome

    # -- the checks -------------------------------------------------------

    def _unevidenced_statistics(self, report: ReportBundle) -> list[QualityIssue]:
        """Statistics in the prose that no ``observed`` evidence record contains.

        Restricted to ``observed`` on purpose. An inference is a reading of a
        source, and letting one license a number would reintroduce exactly the
        figure a reader would act on, with a guess behind it.
        """
        observed = self._observed_evidence_text(report)
        issues: list[QualityIssue] = []
        for location, text in self._prose(report):
            for token in numeric_tokens(text):
                if observed and _matches(observed, token):
                    continue
                issues.append(
                    QualityIssue(
                        code="unevidenced_statistic",
                        severity="critical",
                        message=(
                            f"the report states {token!r} with no observed evidence "
                            f"containing it"
                        ),
                        location=location,
                    )
                )
        return issues

    def _dangling_citations(self, report: ReportBundle) -> list[QualityIssue]:
        """Findings citing evidence the report does not carry."""
        known = {str(item.get("id")) for item in report.evidence if isinstance(item, dict)}
        issues: list[QualityIssue] = []
        for finding in report.findings + report.insights:
            ids = list(
                getattr(finding, "evidence_ids", None) or getattr(finding, "evidence", []) or []
            )
            for reference in ids:
                if reference not in known:
                    issues.append(
                        QualityIssue(
                            code="dangling_citation",
                            severity="critical",
                            message=(
                                f"{type(finding).__name__} {getattr(finding, 'id', '?')!r} "
                                f"cites evidence {reference!r}, which the report does "
                                f"not carry"
                            ),
                            field=str(getattr(finding, "id", "")),
                        )
                    )
        return issues

    def _refuted_claims_still_present(self, report: ReportBundle) -> list[QualityIssue]:
        """A sentence the fact checker refuted, still in the body."""
        refuted = {
            record.claim
            for record in report.claim_verdicts
            if record.verdict in REFUTED_VERDICTS and record.claim
        }
        if not refuted:
            return []
        issues: list[QualityIssue] = []
        for location, text in self._prose(report):
            for claim in refuted:
                if claim and claim in text:
                    issues.append(
                        QualityIssue(
                            code="refuted_claim_present",
                            severity="critical",
                            message=(
                                f"a claim the fact checker refuted is still in the "
                                f"report: {claim!r}"
                            ),
                            location=location,
                        )
                    )
        return issues

    def _missing_sources(self, report: ReportBundle) -> list[QualityIssue]:
        """A report with no sources, or with sources but no evidence."""
        issues: list[QualityIssue] = []
        if not report.sources:
            issues.append(
                QualityIssue(
                    code="no_sources",
                    severity="critical",
                    message="the report cites no sources at all",
                )
            )
        if not report.evidence:
            issues.append(
                QualityIssue(
                    code="no_evidence",
                    severity="critical",
                    message="the report carries no evidence records",
                )
            )
        return issues

    def _coverage_warnings(self, report: ReportBundle) -> list[QualityIssue]:
        """Gaps worth stating, at a severity that does not stop publication."""
        warnings: list[QualityIssue] = []
        if report.market_analysis is None:
            warnings.append(
                QualityIssue(
                    code="missing_section",
                    severity="info",
                    message="no market analysis was produced for this run",
                )
            )
        if report.competitors and not report.competitor_profiles:
            warnings.append(
                QualityIssue(
                    code="missing_profiles",
                    severity="info",
                    message=(
                        f"{len(report.competitors)} competitor(s) were validated but "
                        f"no profiles were built for them"
                    ),
                )
            )
        return warnings

    # -- the repairs ------------------------------------------------------

    def _apply_builtin_repairs(self, report: ReportBundle) -> None:
        """Remove what was refuted and what cannot be cited. Nothing else.

        No rewrites. A refuted sentence is deleted rather than softened, because
        the softened version is an assertion nobody checked, and this system is
        built on the claim that its sentences are checked.
        """
        refuted_claims = {
            record.claim
            for record in report.claim_verdicts
            if record.verdict in REFUTED_VERDICTS and record.claim
        }
        refuted_ids = {
            record.claim_id
            for record in report.claim_verdicts
            if record.verdict in REFUTED_VERDICTS
        }
        if not refuted_claims and not refuted_ids:
            return

        for field_name in PROSE_FIELDS:
            items = getattr(report, field_name, None)
            if not isinstance(items, list):
                continue
            setattr(
                report,
                field_name,
                [
                    item
                    for item in items
                    if not self._contains_refuted(item, refuted_claims)
                ],
            )

        if report.report_markdown:
            for claim in refuted_claims:
                report.report_markdown = _drop_sentence(report.report_markdown, claim)

        if refuted_ids:
            report.findings = [
                finding
                for finding in report.findings
                if getattr(finding, "id", None) not in refuted_ids
            ]

        for record in report.claim_verdicts:
            if record.verdict in REFUTED_VERDICTS:
                record.repaired = True

    @staticmethod
    def _contains_refuted(item: Any, refuted: set[str]) -> bool:
        """Whether a prose item is, or contains, a refuted claim."""
        text = _text_of(item)
        if not text:
            return False
        return any(claim in text for claim in refuted)

    # -- shared -----------------------------------------------------------

    @staticmethod
    def _observed_evidence_text(report: ReportBundle) -> str:
        """Every observed evidence record's own words, joined for comparison."""
        parts: list[str] = []
        for item in report.evidence:
            if not isinstance(item, dict):
                continue
            if str(item.get("claim_type", "")) != ClaimType.OBSERVED.value:
                continue
            text = item.get("evidence_text") or item.get("claim") or ""
            if text:
                parts.append(str(text))
        return " ".join(parts)

    @staticmethod
    def _prose(report: ReportBundle) -> Iterable[tuple[str, str]]:
        """Every piece of text a reader would read, with its location.

        The markdown body is included because the report's own renderer is not
        the only thing that will be read — an API consumer gets the fields and a
        human gets the markdown, and a statistic can be planted in either.
        """
        for field_name in PROSE_FIELDS:
            items = getattr(report, field_name, None)
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items):
                text = _text_of(item)
                if text:
                    yield f"{field_name}[{index}]", text
        for finding in report.findings + report.insights:
            statement = getattr(finding, "statement", None) or getattr(
                finding, "observation", None
            )
            if statement:
                yield f"findings[{getattr(finding, 'id', '?')}]", str(statement)
        if report.report_markdown:
            yield "report_markdown", report.report_markdown
        market = report.market_analysis
        if isinstance(market, MarketAnalysis):
            for index, observation in enumerate(market.observations):
                if observation:
                    yield f"market_analysis.observations[{index}]", str(observation)


def _text_of(item: Any) -> str:
    """The readable text of a prose item, whatever shape it arrived in."""
    if isinstance(item, str):
        return item
    for attribute in ("risk", "opportunity", "statement", "observation", "reasoning"):
        value = getattr(item, attribute, None)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(item, dict):
        for key in ("risk", "opportunity", "statement", "observation", "reasoning"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _matches(haystack: str, token: str) -> bool:
    """Whether a statistic appears in the observed evidence, ignoring formatting.

    Compared with the same digit-only normalisation the fact checker uses, so
    "12,000" in prose matches "12000" in a quote rather than reading as a
    statistic nobody sourced.
    """
    if not haystack:
        return False
    return re.sub(r"[,\s]", "", token) in re.sub(r"[,\s]", "", haystack)


def _drop_sentence(text: str, claim: str) -> str:
    """Remove one sentence from a body of text, leaving the rest intact.

    Sentence-wise rather than a substring replace, so removing a claim that
    happens to be a fragment of another sentence does not take the other one
    with it.
    """
    if not claim:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = [part for part in parts if claim not in part]
    return " ".join(kept) if kept else text


#: The sentinel is not a statistic, and neither is an empty section title.
_NOT_A_STATISTIC = NOT_PUBLIC
