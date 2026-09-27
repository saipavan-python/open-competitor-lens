"""What changed between two runs of the same competitor.

Running a competitor analysis twice is only worth the second run if it can say
what moved. Without that, the reader has two documents and no way to tell a
pricing change from a page that was simply reworded — and the pull towards
reading every difference as a change is strong, because vendors edit their
marketing constantly and a naive diff of two crawls is almost all noise.

Three rules keep the signal legible, and each is a decision about what counts as
knowledge:

**Only comparable fields are compared.** A ``ChangeEvent`` is emitted for a field
that appears in both runs. A field that was known and is now gone from the
comparison table is not reported: no observation is not an observation of nothing,
and reporting it would tell the reader a price was withdrawn every time a crawl
missed a page. The reverse *is* a change — a field that appears for the first time
is a real development, and a first run is the one case where everything looks new
and nothing is, which is why an empty previous run reports nothing at all.

**Types are matched, not just values.** ``10`` and ``"10"`` are different facts
however alike they look in a table, and a case change is a change: vendors do
recase their plan names, and a reader comparing two runs wants to know.

**The field type comes from a declared mapping.** A field nobody mapped still
produces an event, typed ``other``, because an unmapped change is still worth
reporting and raising would make the detector unusable for any table it has not
seen. Substring matching is deliberate — ``acme_starter_price_monthly`` is a
price — and a caller can override the whole table.

The snapshots are plain mappings keyed by company and then by field, because the
durable artefact being compared is the comparison table, and requiring a
bespoke snapshot class would mean this module had to be taught about every field
the analysts might invent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.schemas.provider import ChangeEvent

log = get_logger(__name__)

#: Field name to change type. A substring match, longest key first, so a specific
#: mapping wins over a general one and an unmapped field is ``other`` rather than
#: an error.
DEFAULT_FIELD_TYPES: dict[str, str] = {
    "price": "pricing",
    "pricing": "pricing",
    "cost": "pricing",
    "plan": "pricing",
    "tier": "pricing",
    "seat": "pricing",
    "billing": "pricing",
    "free_tier": "feature",
    "trial": "feature",
    "feature": "feature",
    "product": "product",
    "service": "product",
    "integration": "product",
    "message": "marketing",
    "channel": "marketing",
    "rank": "seo",
    "keyword": "seo",
    "followers": "social",
    "rating": "review",
    "review": "review",
    "tech": "technology",
    "stack": "technology",
    "market": "market",
}

#: Used for a field the mapping does not cover.
OTHER_CHANGE = "other"

Clock = Callable[[], datetime]


@dataclass(frozen=True)
class CompetitorSnapshot:
    """One observation of every comparable field, for a set of companies."""

    companies: tuple[str, ...]
    values: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_fields(
        cls, fields: Mapping[str, Mapping[str, Any]]
    ) -> CompetitorSnapshot:
        """Build a snapshot from ``{company: {field: value}}``.

        Companies are sorted so two snapshots of the same run compare equal
        whatever order they were assembled in.
        """
        return cls(
            companies=tuple(sorted(fields)),
            values={company: dict(value) for company, value in fields.items()},
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CompetitorSnapshot:
        """Build from a snapshot, a plain mapping, or anything already snapshotted."""
        if isinstance(raw, CompetitorSnapshot):
            return raw
        return cls.from_fields(
            {
                company: dict(value)
                for company, value in dict(raw).items()
                if isinstance(value, Mapping)
            }
        )

    def fields(self, company: str) -> dict[str, Any]:
        """One company's fields, or an empty mapping for a company it never saw."""
        return dict(self.values.get(company, {}))

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {company: self.fields(company) for company in self.companies}


class ChangeDetector:
    """Compares two snapshots and reports what moved."""

    def __init__(
        self,
        *,
        field_types: Mapping[str, str] | None = None,
        now: Clock | None = None,
    ) -> None:
        self._types = dict(field_types or DEFAULT_FIELD_TYPES)
        # Longest key first, so "free_tier" is consulted before a bare "tier".
        self._ordered_keys = sorted(self._types, key=len, reverse=True)
        self._now = now or (lambda: datetime.now(UTC))

    def diff(
        self,
        previous: Mapping[str, Any] | CompetitorSnapshot,
        current: Mapping[str, Any] | CompetitorSnapshot,
    ) -> list[ChangeEvent]:
        """Every changed field, one event each, ordered by company then field.

        A first run compares against nothing and reports nothing: everything
        looking new is the expected state of a run with no history, not a finding.
        """
        before = CompetitorSnapshot.from_mapping(previous)
        after = CompetitorSnapshot.from_mapping(current)
        if not before.companies or not after.companies:
            return []

        events: list[ChangeEvent] = []
        for company in sorted(set(before.companies) & set(after.companies)):
            old_fields = before.fields(company)
            new_fields = after.fields(company)
            # Asymmetric on purpose. A field that appears is reported; a field that
            # disappears is not. Appearing means there is something new to say —
            # even if a missed crawl is the reason it appeared, the reader is better
            # served by "previously unobserved" than by silence. Disappearing almost
            # always means the retrieval failed, and reporting that as a change
            # would accuse a vendor of withdrawing a price because a page 404'd.
            for field in sorted(set(old_fields) | set(new_fields)):
                if field not in new_fields:
                    continue
                if field not in old_fields:
                    events.append(
                        ChangeEvent(
                            company=company,
                            change_type=self.change_type(field),
                            previous_value={},
                            new_value=_as_value_object(new_fields[field]),
                            detected_at=self._now().isoformat(),
                            source=_source_of(new_fields[field]),
                        )
                    )
                    continue
                old_value = _value_of(old_fields[field])
                new_value = _value_of(new_fields[field])
                if not _differs(old_value, new_value):
                    continue
                events.append(
                    ChangeEvent(
                        company=company,
                        change_type=self.change_type(field),
                        previous_value=_as_value_object(old_fields[field]),
                        new_value=_as_value_object(new_fields[field]),
                        detected_at=self._now().isoformat(),
                        source=_source_of(new_fields[field]) or _source_of(old_fields[field]),
                    )
                )
        if events:
            log.info(
                "competitor change detected",
                extra={"events": len(events), "companies": len({e.company for e in events})},
            )
        return events

    def change_type(self, field: str) -> str:
        """The declared change type for a field name, longest key first.

        Substring rather than exact matching, because field names arrive prefixed
        and suffixed — ``acme_starter_price_monthly`` is a price — and an exact
        lookup would type most real fields as ``other``.
        """
        lowered = field.lower()
        for key in self._ordered_keys:
            if key in lowered:
                return self._types[key]
        return OTHER_CHANGE

    def changes_for(self, events: Iterable[ChangeEvent], company: str) -> list[ChangeEvent]:
        """One company's events, which is what a per-competitor report section needs."""
        return [event for event in events if event.company == company]


def _value_of(raw: Any) -> Any:
    """The comparable value of a field, unwrapping a ``{"value": ...}`` envelope.

    A wrapper is a convenience for carrying provenance; the thing being compared
    is the value inside it, and comparing the wrapper would report a change every
    time a source id was added.
    """
    if isinstance(raw, Mapping):
        return raw.get("value")
    return raw


def _as_value_object(raw: Any) -> dict[str, Any]:
    """A field rendered as ``{"value": ...}`` for the event, keeping any wrapper."""
    if isinstance(raw, Mapping):
        return dict(raw)
    return {"value": raw}


def _source_of(raw: Any) -> str | None:
    """The URL a field's value came from, if it was recorded."""
    if isinstance(raw, Mapping):
        for key in ("url", "source", "source_url"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _differs(old: Any, new: Any) -> bool:
    """Whether two comparable values differ.

    A type change counts, deliberately: ``10`` and ``"10"`` render identically in a
    table and are different facts, and a vendor that switches a price field from a
    number to a string has changed something a reader comparing runs would want to
    know about.
    """
    if type(old) is not type(new):
        return True
    if isinstance(old, (list, tuple)):
        return list(old) != list(new)
    if isinstance(old, dict):
        return dict(old) != dict(new)
    return old != new
