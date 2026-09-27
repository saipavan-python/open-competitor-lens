"""Competitor repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.company import CompetitorModel


class CompetitorRepository:
    """Persistence for :class:`~app.models.company.CompetitorModel`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, run_id: str, name: str, identity_key: str, website: str | None = None,
        domain: str | None = None, validated: bool = False,
        relevance_reason: str | None = None, is_target: bool = False,
        evidence_ids: list[str] | None = None, description: str | None = None,
        industry: str | None = None, country: str | None = None,
        discovery_queries: list[str] | None = None,
    ) -> CompetitorModel:
        competitor = CompetitorModel(
            run_id=run_id, name=name, canonical_name=name, identity_key=identity_key,
            website=website, domain=domain, validated=validated,
            relevance_reason=relevance_reason, is_target=is_target,
            evidence_ids=evidence_ids or [], description=description, industry=industry,
            country=country, discovery_queries=discovery_queries or [],
        )
        self._session.add(competitor)
        await self._session.flush()
        return competitor

    async def list_by_run(self, run_id: str, *, validated_only: bool = False
                          ) -> list[CompetitorModel]:
        stmt = select(CompetitorModel).where(CompetitorModel.run_id == run_id)
        if validated_only:
            stmt = stmt.where(CompetitorModel.validated.is_(True))
        result = await self._session.execute(stmt.order_by(CompetitorModel.created_at))
        return list(result.scalars().all())

    async def get(self, competitor_id: str) -> CompetitorModel | None:
        return await self._session.get(CompetitorModel, competitor_id)

    async def list_recent(self, *, limit: int = 100, offset: int = 0
                          ) -> list[CompetitorModel]:
        result = await self._session.execute(
            select(CompetitorModel).order_by(CompetitorModel.created_at.desc())
            .limit(limit).offset(offset)
        )
        return list(result.scalars().all())
