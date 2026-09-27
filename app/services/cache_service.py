"""Content cache, keyed by URL and validated by content hash.

Exists to serve two jobs at once: not re-fetching a page we already have, and
noticing when a page has changed. The second is why entries carry a content
hash and a timestamp — a cache that could not tell you what changed is just a
cache, and change detection across runs is a spec requirement.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from app.core.logging import get_logger
from app.schemas.provider import FetchResult
from app.utils.hashing import sha256_text
from app.utils.urls import canonicalize_url

log = get_logger(__name__)


@dataclass(frozen=True)
class CacheEntry:
    url: str
    result: FetchResult
    stored_at: float
    content_hash: str | None

    def age(self, *, now: float | None = None) -> float:
        return (now if now is not None else time.monotonic()) - self.stored_at

    def is_stale(self, ttl_seconds: float | None, *, now: float | None = None) -> bool:
        return ttl_seconds is not None and self.age(now=now) > ttl_seconds


@dataclass(frozen=True)
class CacheLookup:
    """What a lookup found, so the caller can distinguish hit from miss."""

    hit: bool
    entry: CacheEntry | None = None
    reason: str = ""


class ContentCache:
    """A bounded LRU cache of fetched pages.

    Bounded because this holds page bodies in memory: an unbounded cache under a
    crawl budget of forty pages per competitor and fifty competitors is a
    memory leak with a slow fuse.
    """

    def __init__(self, *, max_entries: int = 500, ttl_seconds: float | None = 3600.0
                 ) -> None:
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def key_for(url: str) -> str:
        return canonicalize_url(url) or url

    def get(self, url: str, *, ttl_seconds: float | None = None) -> CacheLookup:
        key = self.key_for(url)
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return CacheLookup(hit=False, reason="not cached")
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        if entry.is_stale(ttl):
            del self._entries[key]
            self.misses += 1
            return CacheLookup(hit=False, entry=entry, reason="stale")
        self._entries.move_to_end(key)
        self.hits += 1
        return CacheLookup(hit=True, entry=entry)

    def put(self, result: FetchResult, *, now: float | None = None) -> CacheEntry | None:
        """Store a successful fetch. Failures are never cached.

        Caching a failed retrieval would turn a transient outage into a
        persistent one for the length of the TTL.
        """
        if not result.ok:
            return None
        key = self.key_for(result.final_url or result.url)
        entry = CacheEntry(
            url=key,
            result=result,
            stored_at=now if now is not None else time.monotonic(),
            content_hash=result.content_hash or sha256_text(result.text),
        )
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return entry

    def changed_since(self, previous: CacheEntry, result: FetchResult) -> bool:
        """Whether a fresh fetch differs from what was cached."""
        current = result.content_hash or sha256_text(result.text)
        return current != previous.content_hash

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, int | float | None]:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else None,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
        }
