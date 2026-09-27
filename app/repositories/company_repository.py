"""Company repository.

Companies are keyed by a normalised ``identity_key`` so "Acme", "Acme Ltd." and
"acme.com" resolve to one row. Without this the report ends up listing the same
competitor three times.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.company import CompanyModel


class CompanyRepository:
    """Persistence for :class:`~app.models.company.CompanyModel`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, name: str, identity_key: str, website: str | None = None,
        domain: str | None = None, industry: str | None = None,
        country: str | None = None, description: str | None = None,
        aliases: list[str] | None = None, domains: list[str] | None = None,
    ) -> CompanyModel:
        company = CompanyModel(
            name=name, canonical_name=name, identity_key=identity_key, website=website,
            domain=domain, industry=industry, country=country, description=description,
            aliases=aliases or [], domains=domains or ([domain] if domain else []),
        )
        self._session.add(company)
        await self._session.flush()
        return company

    async def get(self, company_id: str) -> CompanyModel | None:
        return await self._session.get(CompanyModel, company_id)

    async def get_by_identity(self, identity_key: str) -> CompanyModel | None:
        result = await self._session.execute(
            select(CompanyModel).where(CompanyModel.identity_key == identity_key)
        )
        return result.scalars().first()

    async def upsert(
        self, *, name: str, identity_key: str, website: str | None = None,
        domain: str | None = None, industry: str | None = None,
        country: str | None = None, description: str | None = None,
    ) -> CompanyModel:
        """Create the company, or return the existing row for this identity."""
        existing = await self.get_by_identity(identity_key)
        if existing is not None:
            # Later research may fill in fields the first pass could not.
            for field, value in (
                ("website", website), ("domain", domain), ("industry", industry),
                ("country", country), ("description", description),
            ):
                if value and not getattr(existing, field):
                    setattr(existing, field, value)
            await self._session.flush()
            return existing
        return await self.create(
            name=name, identity_key=identity_key, website=website, domain=domain,
            industry=industry, country=country, description=description,
        )

    async def list_all(self, *, limit: int = 100, offset: int = 0) -> list[CompanyModel]:
        result = await self._session.execute(
            select(CompanyModel).order_by(CompanyModel.created_at.desc())
            .limit(limit).offset(offset)
        )
        return list(result.scalars().all())
