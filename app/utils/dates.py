"""Date parsing and UTC time handling."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime

_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_SLASH_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_MONTHS = {
    name: index
    for index, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}


def utcnow() -> datetime:
    """Timezone-aware current time. All stored timestamps are UTC."""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Attach UTC to a naive datetime, or convert an aware one."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def parse_date(value: str | None) -> date | None:
    """Best-effort parse of a date from a page. ``None`` when not found.

    Deliberately conservative: a wrong date is worse than no date, because
    freshness is one of the things the report is checked against.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()

    if (match := _ISO_DATE.search(text)) is not None:
        try:
            return date(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:
            return None

    if (match := _SLASH_DATE.search(text)) is not None:
        try:
            return date(int(match[3]), int(match[1]), int(match[2]))
        except ValueError:
            return None

    lowered = text.casefold()
    if (match := re.search(r"(\d{1,2})\s+([a-z]{3})[a-z]*\s+(\d{4})", lowered)) is not None:
        month = _MONTHS.get(match[2])
        if month:
            try:
                return date(int(match[3]), month, int(match[1]))
            except ValueError:
                return None
    if (match := re.search(r"([a-z]{3})[a-z]*\s+(\d{1,2}),?\s+(\d{4})", lowered)) is not None:
        month = _MONTHS.get(match[1])
        if month:
            try:
                return date(int(match[3]), month, int(match[2]))
            except ValueError:
                return None
    return None


def days_since(value: str | date | datetime | None, *, reference: datetime | None = None
               ) -> int | None:
    """Age in days, or ``None`` when the date could not be determined."""
    parsed = parse_date(value) if isinstance(value, str) else None
    if parsed is None and isinstance(value, (date, datetime)):
        parsed = value if isinstance(value, date) else value.date()
    if parsed is None:
        return None
    now = reference or utcnow()
    return (now.date() - parsed).days
