"""Async SQLAlchemy engine and session management.

SQLite is the default so the project runs with no infrastructure; pointing
``DATABASE_URL`` at PostgreSQL switches driver and dialect with no code change.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


def _ensure_async_driver(url: str) -> str:
    """Translate a sync-style URL into its async driver form."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("sqlite://") and "+aiosqlite" not in url:
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


class Database:
    """Owns one engine and hands out sessions.

    Created once per application (or per test) and never used as a global.
    """

    def __init__(self, url: str | None = None, *, settings: Settings | None = None) -> None:
        resolved = url or (settings or get_settings()).database_url
        self.url = _ensure_async_driver(resolved)
        connect_args = {"check_same_thread": False} if self.url.startswith("sqlite") else {}
        self._engine: AsyncEngine = create_async_engine(
            self.url, echo=(settings or get_settings()).db_echo, connect_args=connect_args
        )
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        return self._sessionmaker

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A session that commits on success and rolls back on error."""
        async with self._sessionmaker() as active:
            try:
                yield active
                await active.commit()
            except Exception:
                await active.rollback()
                raise

    async def create_all(self) -> None:
        """Create the schema. Used by tests and ``scripts/init_db.py`` only.

        Production deployments must use Alembic migrations instead; see the
        README. This method exists so a developer can get a working database
        without a migration step.
        """
        from app.models import Base  # noqa: PLC0415 - avoid import cycle

        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def drop_all(self) -> None:
        from app.models import Base  # noqa: PLC0415 - avoid import cycle

        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)

    async def health_check(self) -> bool:
        """Return True when the database answers a trivial query."""
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001 - health check must not raise
            log.warning("database health check failed", extra={"error": str(exc)})
            return False
        return True

    async def dispose(self) -> None:
        await self._engine.dispose()
