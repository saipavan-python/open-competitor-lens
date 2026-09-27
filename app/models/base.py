"""Declarative base and the shared column conventions.

Conventions, applied everywhere so no table has to reinvent them:

* primary key is a UUID generated in Python (so a row can be built and referenced
  before it is flushed)
* every timestamp is stored timezone-aware UTC
* JSON columns use the portable ``sa.JSON`` type, which works on both SQLite and
  PostgreSQL
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base for every table."""

    type_annotation_map = {dict[str, Any]: sa.JSON, list[str]: sa.JSON}


class UUIDPrimaryKeyMixin:
    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=new_uuid)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


def json_column(**kwargs: Any) -> Mapped[Any]:
    """A JSON column that defaults to an empty list rather than NULL."""
    return mapped_column(sa.JSON, default=list, nullable=False, **kwargs)


def json_dict_column(**kwargs: Any) -> Mapped[Any]:
    return mapped_column(sa.JSON, default=dict, nullable=False, **kwargs)
