"""Content hashing and stable identifiers.

Hashes are the backbone of deduplication and change detection, so they are
computed over a *composite* key - never over content alone. Two companies that
publish the same boilerplate footer are different sources; hashing only the text
would collapse them into one.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    """SHA-256 of raw bytes, hex encoded."""
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a string's UTF-8 encoding."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def composite_hash(*parts: str | None) -> str:
    """Hash of several parts joined by a separator that cannot appear in a URL.

    The ``\\x00`` separator stops ``("ab", "c")`` and ``("a", "bc")`` from
    colliding, which a plain ``"|".join`` would allow.
    """
    return sha256_text("\x00".join(part or "" for part in parts))


def page_hash(canonical_url: str, content: str) -> str:
    """Content identity for one page: same URL *and* same bytes."""
    return composite_hash(canonical_url, content)


def stable_id(prefix: str, *parts: Any) -> str:
    """A deterministic identifier derived from ``parts``.

    Deterministic so that re-running a research step for the same input produces
    the same ids, which keeps replays and targeted re-runs diffable.
    """
    digest = hashlib.sha256("\x00".join(str(part) for part in parts).encode()).hexdigest()
    return f"{prefix}-{digest[:16]}"


_WHITESPACE = re.compile(r"\s+")


def normalized_hash(text: str) -> str:
    """Hash text after collapsing whitespace, so reformatting is not a change."""
    return sha256_text(_WHITESPACE.sub(" ", text).strip())


def new_run_id() -> str:
    """A fresh random run identifier."""
    return str(uuid.uuid4())
