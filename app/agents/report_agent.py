"""Assembling the deliverable from everything the run produced.

This module decides *what* goes in the report; :mod:`app.services.report_service`
decides how it is printed. The split is not cosmetic. Assembly is where a field
could be invented, so it is a pure projection of the run's own results: every
section is filled from something an earlier step retrieved or refused, and a
section with nothing behind it is left empty for the renderer to state. Rendering
is where completeness becomes visible — all nineteen sections are emitted whether
or not they have content — and it lives one layer down, where it belongs.

No model is involved. The report is a container for nine analysts' findings, and a
model in that loop could only reorder or restate something a reader has already
been shown to be unverified. The ``final_report`` and ``synthesis`` prompts exist
and are wired for Task 12's graph, but this module does not call them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from app.agents.base import Agent, AgentContext, AgentResult, AgentStatus
from app.core.constants import RunStatus
from app.core.logging import get_logger
from app.schemas.competitor import ValidatedCompetitor
from app.schemas.report import (
    REPORT_SECTIONS,
    SECTION_TITLES,
    ReportBundle,
    ReportMetadata,
)
from app.services.report_service import ReportService

log = get_logger(__name__)


class ReportAgent(Agent):
    """Builds the report bundle from everything the run produced."""

    name = "report"
    goal = "Assemble every finding the run made into one report, gaps included."
    output_schema = ReportBundle

    def __init__(self, *, service: ReportService | None = None) -> None:
        self._service = service or ReportService()

    async def run(self, ctx: AgentContext) -> AgentResult:
        """The assembled report, with its missing sections stated.

        Never fails for a missing section. A run that analysed four of nine
        categories still has four categories' worth of report, and that document
        with its gaps named is more useful than an error.
        """
        started = ctx.extras.get("started_at") or datetime.now(UTC)
        bundle = ReportBundle(
            metadata=self._metadata(ctx, started),
            target_company=ctx.extras.get("target_company"),
            competitors=_competitors(ctx.extras.get("competitors", []) or []),
            competitor_profiles=dict(ctx.extras.get("competitor_profiles", {}) or {}),
            product_comparison=ctx.extras.get("comparison"),
            market_analysis=ctx.extras.get("market_analysis"),
            pricing_comparison=dict(ctx.extras.get("pricing_comparison", {}) or {}),
            marketing_analysis=dict(ctx.extras.get("marketing_analysis", {}) or {}),
            seo_analysis=dict(ctx.extras.get("seo_analysis", {}) or {}),
            social_analysis=dict(ctx.extras.get("social_analysis", {}) or {}),
            review_analysis=dict(ctx.extras.get("review_analysis", {}) or {}),
            technology_analysis=dict(ctx.extras.get("technology_analysis", {}) or {}),
            website_analysis=dict(ctx.extras.get("website_analysis", {}) or {}),
            product_analysis=dict(ctx.extras.get("product_analysis", {}) or {}),
            cross_competitor_patterns=list(
                ctx.extras.get("cross_competitor_patterns", []) or []
            ),
            opportunities=list(ctx.extras.get("opportunities", []) or []),
            observed_gaps=list(ctx.extras.get("observed_gaps", []) or []),
            risks=list(ctx.extras.get("risks", []) or []),
            findings=list(ctx.extras.get("findings", []) or []),
            insights=list(ctx.extras.get("insights", []) or []),
            claim_verdicts=list(ctx.extras.get("claim_verdicts", []) or []),
            quality_report=ctx.extras.get("quality_report"),
            sources=list(ctx.extras.get("sources", []) or []),
            evidence=list(ctx.extras.get("evidence", []) or []),
            warnings=list(ctx.extras.get("warnings", []) or []),
            failed_steps=list(ctx.extras.get("failed_steps", []) or []),
        )
        self._fill_derived(bundle, ctx)

        missing = self._service.missing_sections(bundle)
        self._service.render(bundle)

        result = AgentResult(
            agent=self.name,
            status=AgentStatus.PARTIAL if (missing or bundle.warnings) else AgentStatus.OK,
            payload=bundle,
            warnings=list(bundle.warnings),
        )
        if missing:
            result.warnings.append(
                f"{len(missing)} of {len(REPORT_SECTIONS)} report sections have no content: "
                f"{', '.join(SECTION_TITLES.get(name, name) for name in missing)}"
            )
        log.info(
            "report assembled",
            extra={"sections": len(REPORT_SECTIONS) - len(missing), "missing": len(missing)},
        )
        return result

    # -- assembly ---------------------------------------------------------

    def _metadata(self, ctx: AgentContext, started: datetime) -> ReportMetadata:
        """The report's own provenance, from the run rather than the clock alone."""
        target = ctx.extras.get("target_company")
        return ReportMetadata(
            run_id=str(ctx.extras.get("run_id", "")),
            company=ctx.company,
            website=getattr(target, "official_website", None),
            industry=getattr(target, "industry", None),
            country=getattr(target, "country", None),
            research_started_at=started,
            research_completed_at=ctx.extras.get("completed_at") or datetime.now(UTC),
            research_goal=ctx.research_goal or None,
            report_language=str(ctx.extras.get("language", "en")),
            research_depth=str(ctx.extras.get("depth", "standard")),
            model_profiles=dict(ctx.extras.get("model_profiles", {}) or {}),
        )

    def _fill_derived(self, bundle: ReportBundle, ctx: AgentContext) -> None:
        """The fields computed from the rest rather than supplied.

        Run status and evidence quality are derived, never passed in: a caller
        that could set them would be able to report a clean run regardless of what
        the run actually did.
        """
        status = ctx.extras.get("run_status")
        if isinstance(status, RunStatus):
            bundle.run_status = status
        elif bundle.failed_steps and not bundle.sources:
            bundle.run_status = RunStatus.FAILED
        elif bundle.failed_steps or bundle.warnings:
            bundle.run_status = RunStatus.PARTIAL
        bundle.evidence_quality = self._service.quality(bundle)
        bundle.progress = 100 if bundle.run_status is not RunStatus.FAILED else 0


def _competitors(items: Sequence[Any]) -> list[ValidatedCompetitor]:
    """The run's competitors, accepting names or full records.

    A step that only resolved a company's name still belongs in the report's
    landscape section, so a bare string becomes a record that says only its name.
    ``validated`` stays ``False`` on purpose: this agent did not validate it, and
    a report must not claim a check it did not perform.
    """
    records: list[ValidatedCompetitor] = []
    for item in items:
        if isinstance(item, ValidatedCompetitor):
            records.append(item)
            continue
        if isinstance(item, Mapping):
            records.append(ValidatedCompetitor.model_validate(item))
            continue
        name = str(item).strip()
        if name:
            records.append(
                ValidatedCompetitor(
                    name=name,
                    relevance_reason="listed as a competitor by an earlier step of this run",
                )
            )
    return records
