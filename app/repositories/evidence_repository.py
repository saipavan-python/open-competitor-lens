"""Evidence, finding and report repositories."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import ClaimType, Confidence, RunStatus
from app.models.analysis import ReportModel
from app.models.source import EvidenceModel, FindingModel


class EvidenceRepository:
    """Persistence for :class:`~app.models.source.EvidenceModel`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, run_id: str, claim: str, source_id: str, source_url: str,
        claim_type: ClaimType, confidence: Confidence, evidence_text: str | None = None,
        claim_key: str | None = None, company: str | None = None,
        competitor: str | None = None, category: str | None = None,
    ) -> EvidenceModel:
        evidence = EvidenceModel(
            run_id=run_id, claim=claim, source_id=source_id, source_url=source_url,
            claim_type=claim_type, confidence=confidence, evidence_text=evidence_text,
            claim_key=claim_key, company=company, competitor=competitor, category=category,
        )
        self._session.add(evidence)
        await self._session.flush()
        return evidence

    async def create_many(self, run_id: str, payloads: list[dict[str, Any]]) -> list[EvidenceModel]:
        created: list[EvidenceModel] = []
        for payload in payloads:
            created.append(await self.create(run_id=run_id, **payload))
        return created

    async def list_by_run(
        self, run_id: str, *, competitor: str | None = None, category: str | None = None
    ) -> list[EvidenceModel]:
        stmt = select(EvidenceModel).where(EvidenceModel.run_id == run_id)
        if competitor:
            stmt = stmt.where(EvidenceModel.competitor == competitor)
        if category:
            stmt = stmt.where(EvidenceModel.category == category)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_claim_key(self, run_id: str, claim_key: str) -> list[EvidenceModel]:
        result = await self._session.execute(
            select(EvidenceModel).where(
                EvidenceModel.run_id == run_id, EvidenceModel.claim_key == claim_key
            )
        )
        return list(result.scalars().all())

    async def count_by_confidence(self, run_id: str) -> dict[str, int]:
        result = await self._session.execute(
            select(EvidenceModel.confidence, func.count())
            .where(EvidenceModel.run_id == run_id).group_by(EvidenceModel.confidence)
        )
        return {str(level): count for level, count in result.all()}

    async def count_by_claim_type(self, run_id: str) -> dict[str, int]:
        result = await self._session.execute(
            select(EvidenceModel.claim_type, func.count())
            .where(EvidenceModel.run_id == run_id).group_by(EvidenceModel.claim_type)
        )
        return {str(kind): count for kind, count in result.all()}


class FindingRepository:
    """Persistence for :class:`~app.models.source.FindingModel`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, run_id: str, company: str, category: str, statement: str,
        evidence_ids: list[str], confidence: Confidence = Confidence.LOW,
        observation: str | None = None, interpretation: str | None = None,
        value: str | None = None,
    ) -> FindingModel:
        finding = FindingModel(
            run_id=run_id, company=company, category=category, statement=statement,
            evidence_ids=evidence_ids, confidence=confidence, observation=observation,
            interpretation=interpretation, value=value,
        )
        self._session.add(finding)
        await self._session.flush()
        return finding

    async def create_many(self, run_id: str, payloads: list[dict[str, Any]]
                          ) -> list[FindingModel]:
        return [await self.create(run_id=run_id, **payload) for payload in payloads]

    async def list_by_run(self, run_id: str, *, company: str | None = None
                          ) -> list[FindingModel]:
        stmt = select(FindingModel).where(FindingModel.run_id == run_id)
        if company:
            stmt = stmt.where(FindingModel.company == company)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_unlinked(self, run_id: str) -> int:
        """Findings that somehow carry no evidence. Should always be zero."""
        rows = await self.list_by_run(run_id)
        return sum(1 for row in rows if not row.evidence_ids)


class ReportRepository:
    """Persistence for stored report versions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(
        self, *, run_id: str, report_json: dict[str, Any],
        report_markdown: str | None = None, report_html: str | None = None,
        repair_round: int = 0,
    ) -> ReportModel:
        # Only one row per run may be flagged latest; a new version supersedes it.
        result = await self._session.execute(
            select(ReportModel).where(ReportModel.run_id == run_id, ReportModel.is_latest.is_(True))
        )
        for previous in result.scalars().all():
            previous.is_latest = False
        report = ReportModel(
            run_id=run_id, report_json=report_json, report_markdown=report_markdown,
            report_html=report_html, is_latest=True, repair_round=repair_round,
        )
        self._session.add(report)
        await self._session.flush()
        return report

    async def get_latest(self, run_id: str) -> ReportModel | None:
        result = await self._session.execute(
            select(ReportModel)
            .where(ReportModel.run_id == run_id, ReportModel.is_latest.is_(True))
            .order_by(ReportModel.created_at.desc())
        )
        return result.scalars().first()

    async def list_versions(self, run_id: str) -> list[ReportModel]:
        result = await self._session.execute(
            select(ReportModel).where(ReportModel.run_id == run_id)
            .order_by(ReportModel.created_at)
        )
        return list(result.scalars().all())


def summarise_run_status(failed_steps: list[str]) -> RunStatus:
    """Derive the run's terminal status from what failed.

    A run where some steps failed is ``partial``, not ``failed``: partial
    success is a real, reportable outcome (spec section 55).
    """
    return RunStatus.PARTIAL if failed_steps else RunStatus.COMPLETED


def count_by(items: list[Any], attribute: str) -> dict[str, int]:
    return dict(Counter(str(getattr(item, attribute, "unknown")) for item in items))
