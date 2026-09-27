"""Source and document repository.

Sources are deduplicated on ``(run_id, canonical_url)``: fetching the same page
twice updates one row, so ``SELECT count(*) FROM sources`` remains a truthful
count of distinct pages actually retrieved.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import PageType, RetrievalStatus, SourceType
from app.core.logging import get_logger
from app.models.source import DocumentModel, SourceModel
from app.utils.hashing import sha256_text

log = get_logger(__name__)


class SourceRepository:
    """Persistence for retrieved pages."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, run_id: str, url: str, canonical_url: str, domain: str | None = None,
        title: str | None = None, source_type: SourceType = SourceType.UNKNOWN,
        company: str | None = None, competitor: str | None = None,
        content_hash: str | None = None, text: str | None = None,
        page_type: PageType = PageType.OTHER, language: str | None = None,
        retrieval_status: RetrievalStatus = RetrievalStatus.OK,
        published_at: datetime | None = None,
        discovery_query: str | None = None, error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SourceModel:
        source = SourceModel(
            run_id=run_id, url=url, canonical_url=canonical_url, domain=domain,
            title=title, source_type=source_type, company=company, competitor=competitor,
            content_hash=content_hash or sha256_text(f"{canonical_url}\n{text or ''}"),
            text=text, page_type=page_type, language=language,
            retrieval_status=retrieval_status, published_at=published_at,
            discovery_query=discovery_query, error=error,
            metadata_json=metadata or {},
        )
        self._session.add(source)
        await self._session.flush()
        return source

    async def upsert(self, **kwargs: Any) -> SourceModel:
        """Insert, or return the existing source for this canonical URL."""
        run_id: str = kwargs["run_id"]
        canonical_url: str = kwargs["canonical_url"]
        result = await self._session.execute(
            select(SourceModel).where(
                SourceModel.run_id == run_id, SourceModel.canonical_url == canonical_url
            )
        )
        existing = result.scalars().first()
        if existing is None:
            return await self.create(**kwargs)
        for field in ("title", "source_type", "company", "competitor", "page_type",
                      "text", "language", "retrieval_status", "error"):
            value = kwargs.get(field)
            if value not in (None, ""):
                setattr(existing, field, value)
        if kwargs.get("text"):
            existing.content_hash = sha256_text(f"{canonical_url}\n{kwargs['text']}")
        await self._session.flush()
        return existing

    async def get(self, source_id: str) -> SourceModel | None:
        return await self._session.get(SourceModel, source_id)

    async def get_many(self, source_ids: list[str]) -> dict[str, SourceModel]:
        if not source_ids:
            return {}
        result = await self._session.execute(
            select(SourceModel).where(SourceModel.id.in_(source_ids))
        )
        return {row.id: row for row in result.scalars().all()}

    async def list_by_run(
        self, run_id: str, *, competitor: str | None = None, limit: int = 1000
    ) -> list[SourceModel]:
        stmt = select(SourceModel).where(SourceModel.run_id == run_id)
        if competitor:
            stmt = stmt.where(SourceModel.competitor == competitor)
        result = await self._session.execute(stmt.limit(limit))
        return list(result.scalars().all())

    async def add_document(
        self, *, run_id: str, source_id: str, url: str, canonical_url: str,
        content: str, company: str | None = None, competitor: str | None = None,
        title: str | None = None, page_type: PageType = PageType.OTHER,
        language: str | None = None, metadata: dict[str, Any] | None = None,
    ) -> DocumentModel:
        """Store a cleaned document, ignoring an exact duplicate within the run."""
        content_hash = sha256_text(content)
        result = await self._session.execute(
            select(DocumentModel).where(
                DocumentModel.run_id == run_id, DocumentModel.content_hash == content_hash
            )
        )
        if (existing := result.scalars().first()) is not None:
            return existing
        document = DocumentModel(
            run_id=run_id, source_id=source_id, url=url, canonical_url=canonical_url,
            content=content, content_hash=content_hash, company=company,
            competitor=competitor, title=title, page_type=page_type, language=language,
            char_count=len(content), metadata_json=metadata or {},
        )
        self._session.add(document)
        try:
            await self._session.flush()
        except IntegrityError:
            # Two concurrent collectors stored the same page; keep the first.
            await self._session.rollback()
            return DocumentModel(id="", run_id=run_id, source_id=source_id, url=url,
                                 canonical_url=canonical_url, content=content,
                                 content_hash=content_hash)
        return document

    async def list_documents(self, run_id: str, *, competitor: str | None = None
                             ) -> list[DocumentModel]:
        stmt = select(DocumentModel).where(DocumentModel.run_id == run_id)
        if competitor:
            stmt = stmt.where(DocumentModel.competitor == competitor)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
