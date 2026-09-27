"""Shared schema primitives.

Every domain model in the system derives from :class:`DomainModel`, which fixes
``extra="forbid"`` so a typo in a field name is a validation error rather than a
silently dropped field that later shows up as a missing finding.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import (
    UNKNOWN_SENTINELS,
    ClaimType,
    Confidence,
    PageType,
    SourceType,
)

#: The four literal strings used whenever evidence could not be established.
#: Having one vocabulary for "we looked and found nothing" means a reader can
#: always tell an absent value from a missed one.
NOT_PUBLIC = "Not publicly available"
NOT_FOUND = "Not found during research"
UNABLE_TO_VERIFY = "Unable to verify"
CONFLICTING = "Conflicting public information"

#: Alias kept for readability at call sites that talk about missing data.
UNKNOWN_NOT_AVAILABLE = NOT_PUBLIC


def utcnow() -> datetime:
    """Timezone-aware current time. All timestamps in the system are UTC."""
    return datetime.now(UTC)


class DomainModel(BaseModel):
    """Base for every persisted or API-exposed model."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=False,
    )


class TimestampedModel(DomainModel):
    """A model that records when it was created."""

    created_at: datetime = Field(default_factory=utcnow)


class PagedQuery(DomainModel):
    """Standard pagination input."""

    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class HealthCheck(DomainModel):
    """One dependency's health, as returned by ``GET /api/v1/health``."""

    name: str
    status: str
    detail: str | None = None


class ApiError(DomainModel):
    """The error envelope every route returns. Never contains a traceback."""

    error: ApiErrorBody

    @classmethod
    def from_exception(cls, exc: Any) -> ApiError:
        from app.core.exceptions import AppError  # noqa: PLC0415 - avoid cycle

        if isinstance(exc, AppError):
            return cls(error=ApiErrorBody(**exc.to_dict()))
        return cls(error=ApiErrorBody(type="UnknownError",
                                      message="An unexpected error occurred.",
                                      retryable=False, provider=None, step=None))


class ApiErrorBody(DomainModel):
    type: str
    message: str
    retryable: bool = False
    provider: str | None = None
    step: str | None = None


ApiError.model_rebuild()


__all__ = [
    "CONFLICTING",
    "NOT_FOUND",
    "NOT_PUBLIC",
    "UNKNOWN_NOT_AVAILABLE",
    "UNKNOWN_SENTINELS",
    "ApiError",
    "ApiErrorBody",
    "ClaimType",
    "Confidence",
    "DomainModel",
    "HealthCheck",
    "PageType",
    "PagedQuery",
    "SourceType",
    "TimestampedModel",
    "utcnow",
]
