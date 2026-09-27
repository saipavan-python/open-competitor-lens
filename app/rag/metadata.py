"""The metadata contract: which keys exist, and which may be filtered on.

Retrieval metadata is the join between a chunk and the rest of the run. An
analyst asks for "Beta's pricing" and gets back a list of text; whether that
list is *about Beta's pricing* is decided entirely by the metadata attached to
the chunks. So the keys are declared here, once, instead of being spelled
ad-hoc at every call site where a typo would silently widen a filter.

**Filtering is strict.** A filter on a key a chunk does not have excludes that
chunk rather than passing it. This is the difference between a filter and a
hint, and the permissive version is a real failure: ask for
``competitor="Beta"`` and a chunk that never recorded a competitor comes back
too, so a competitor's pricing page can be summarised as the target's pricing.
The empty result is honest; the leaky one is not.

``company`` and ``competitor`` are kept as first-class fields rather than living
only in ``metadata`` because the plan's filters name them directly, and because
``entity`` — whichever of the two applies — is the axis every analyst actually
retrieves along.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.core.logging import get_logger

log = get_logger(__name__)

#: Keys a caller may filter on. Anything else is a programming error, caught at
#: the boundary rather than silently matching nothing.
FILTERABLE_KEYS: frozenset[str] = frozenset(
    {
        "company",
        "competitor",
        "entity",
        "category",
        "page_type",
        "source_id",
        "source_type",
        "run_id",
        "document_id",
        "language",
    }
)

#: Keys that are always present, even when empty, so a filter has something
#: definite to compare against rather than a missing key.
CORE_KEYS: frozenset[str] = frozenset(
    {
        "company",
        "competitor",
        "entity",
        "page_type",
        "source_id",
        "run_id",
        "document_id",
        "url",
    }
)

#: Comparison for each filterable key. Anything not listed compares as an
#: exact string match, which is the only safe default: a fuzzy match on
#: `competitor` would bring back "Acme Digital" for a request about "Acme".
OPERATORS: frozenset[str] = frozenset({"=", "!=", "in", "not_in", "contains"})


class UnknownFilterKeyError(ValueError):
    """A filter named a key that is not part of the metadata contract."""


def normalise_key(key: str) -> str:
    key = key.strip().lower()
    if not key:
        raise UnknownFilterKeyError("filter key is empty")
    if key not in FILTERABLE_KEYS:
        raise UnknownFilterKeyError(
            f"{key!r} is not a filterable metadata key; "
            f"allowed keys are {sorted(FILTERABLE_KEYS)}"
        )
    return key


def chunk_value(chunk, key: str) -> str | None:
    """A chunk's value for one metadata key, or ``None`` if it has none.

    Core fields are read from the chunk itself; anything else comes from
    ``metadata``. ``entity`` is derived, because no caller should have to
    remember which of the two company fields to filter on.
    """
    key = key.strip().lower()
    if key == "entity":
        return chunk.competitor or chunk.company
    if key == "competitor":
        return chunk.competitor
    if key == "company":
        return chunk.company
    if key == "page_type":
        return chunk.page_type.value if chunk.page_type is not None else None
    if key == "source_id":
        return chunk.source_id
    if key == "run_id":
        return chunk.run_id
    if key == "document_id":
        return chunk.document_id
    if key == "url":
        return chunk.url
    value = chunk.metadata.get(key)
    return value if value not in (None, "") else None


def matches(chunk, key: str, expected: str) -> bool:
    """Whether a chunk satisfies one strict equality filter."""
    actual = chunk_value(chunk, key)
    if actual is None:
        # Strict: absent metadata cannot satisfy a filter. See module docstring.
        return False
    return actual.strip().casefold() == expected.strip().casefold()


def validate_filters(filters: Mapping[str, str] | None) -> dict[str, str]:
    """Check a filter mapping up front, so a typo fails loudly at the call."""
    if not filters:
        return {}
    return {normalise_key(key): str(value) for key, value in filters.items()}


def matches_all(chunk, filters: Mapping[str, str]) -> tuple[bool, list[str]]:
    """Whether a chunk passes every filter, and which keys admitted it."""
    matched: list[str] = []
    for key, expected in filters.items():
        if not matches(chunk, key, expected):
            return False, []
        matched.append(key)
    return True, matched
