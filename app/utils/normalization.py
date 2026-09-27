"""Company name normalisation and identity keys."""

from __future__ import annotations

import re

#: Legal suffixes stripped before deriving an identity key. Removing them is what
#: makes "Acme Private Limited" and "Acme" one entity.
LEGAL_SUFFIXES: tuple[str, ...] = (
    "private limited", "pvt ltd", "pvt. ltd", "pvt ltd.", "limited", "ltd.",
    "llp", "llc", "l.l.c", "inc.", "inc", "incorporated", "corp.", "corp",
    "corporation", "gmbh", "ag", "sa", "s.a.", "nv", "b.v.", "bv", "pty", "plc",
    "co.", "company", "srl", "spa", "oy", "ab", "as", "a/s", "kft", "sro",
)

_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_space(value: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends."""
    return _WHITESPACE.sub(" ", value or "").strip()


def normalize_company_name(name: str) -> str:
    """Fold a company name to a comparable key.

    Lowercases, removes punctuation and legal suffixes, and collapses spaces.
    ``"Acme, Inc."`` and ``"Acme Private Limited"`` both become ``"acme"``.
    """
    cleaned = _PUNCTUATION.sub(" ", normalize_space(name).casefold())
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    changed = True
    while changed:
        changed = False
        for suffix in LEGAL_SUFFIXES:
            if cleaned == suffix:
                return ""
            if cleaned.endswith(f" {suffix}"):
                cleaned = cleaned[: -len(suffix) - 1].strip()
                changed = True
    return _WHITESPACE.sub(" ", cleaned).strip()


def company_identity_key(name: str, *, extra_aliases: list[str] | None = None) -> str:
    """A stable key that merges aliases of one company but not distinct names.

    Only *equivalent* forms merge: "Kansbe" and "Kansbe Private Limited" and
    "kansbe.com" share a key, while "Kansbe" and "Kansbe Digital" do not. A
    substring rule would collapse genuinely different competitors, so aliases are
    matched by exact normalised equality only. A bare domain also contributes its
    second-level label, since "kansbe.com" is a spelling of "Kansbe", not of
    "Kansbe Com".
    """
    candidates = [name, *(extra_aliases or [])]
    keys: set[str] = set()
    for candidate in candidates:
        normalized = normalize_company_name(candidate)
        if not normalized:
            continue
        keys.add(normalized)
        if "." in candidate and " " not in candidate.strip():
            label = candidate.strip().split(".")[0]
            if label and (from_label := normalize_company_name(label)):
                keys.add(from_label)
    if not keys:
        return ""
    # Prefer the shortest key: it is the least qualified spelling of the entity.
    return min(keys, key=lambda item: (len(item), item))


def slugify(value: str, *, max_length: int = 80) -> str:
    """A filesystem- and URL-safe slug."""
    cleaned = _PUNCTUATION.sub(" ", normalize_space(value).casefold())
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
    return cleaned[:max_length].strip("-")


def normalize_currency(code: str | None) -> str | None:
    """Uppercase a currency code, or return ``None``."""
    return code.strip().upper() if code and code.strip() else None
