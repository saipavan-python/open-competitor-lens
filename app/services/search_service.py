"""Search, as a discovery step.

This service exists to enforce one distinction that is easy to lose and very
expensive to lose: **a search result is not a source.** A search hit is a
snippet — somebody else's summary of a page, on a page nobody retrieved. It is
an excellent way to find a URL and a terrible thing to cite, because it cannot
be quoted, it goes stale, and the description of it was written by the search
engine rather than by the company being described.

So every hit this service produces is registered as a
:attr:`~app.core.constants.SourceType.SEARCH_SNIPPET` and carries no page text,
which makes it structurally incapable of backing a high-confidence claim or a
validation signal. It has to be crawled before it can be cited, and the crawl
produces a real source. That is the whole reason the type exists.

The snippet text itself is kept, in ``notes``, because it is what lets a model
decide whether a candidate is worth crawling. It is deliberately *not* in
``text``: ``text`` means "the body of a page we retrieved", and mixing the two
would let an unretrieved description masquerade as retrieved content one
refactor later.

Deduplication is by registrable domain, because a discovery step that returns
the same site forty times has not found forty things — and a candidate pool
padded with one company's forty pages validates to a single competitor while
looking thorough.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.core.constants import SourceType
from app.core.exceptions import SSRFError
from app.core.logging import get_logger
from app.providers.base import SearchProvider
from app.schemas.evidence import Source
from app.schemas.search import SearchHit
from app.utils.hashing import stable_id
from app.utils.urls import canonicalize_url, registrable_domain, validate_url

log = get_logger(__name__)

_SNIPPET_PREFIX = "search snippet (not retrieved, not citable): "


@dataclass
class SearchService:
    """Runs discovery queries and turns hits into non-citable snippets."""

    provider: SearchProvider
    limit: int = 10
    max_queries: int = 6
    _seen_domains: set[str] = field(default_factory=set)

    async def search(self, query: str, *, limit: int | None = None) -> list[SearchHit]:
        """One query, one round of hits. A failing query yields no hits.

        Discovery is best-effort by design: a search backend being down should
        cost this run some candidates, not abort it. An empty result and a
        failed query are deliberately indistinguishable downstream, so a
        partial run cannot present a provider outage as a finding of "nothing
        there".
        """
        try:
            return await self.provider.search(query, limit=limit or self.limit)
        except Exception as exc:  # noqa: BLE001 - discovery must not abort a run
            log.warning(
                "search query failed",
                extra={
                    "query": query,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            return []

    async def discover(
        self, queries: list[str], *, limit: int | None = None
    ) -> list[Source]:
        """Run several queries concurrently and return deduplicated snippets."""
        selected = list(queries[: self.max_queries])
        results = await asyncio.gather(
            *(self.search(query, limit=limit) for query in selected)
        )

        sources: list[Source] = []
        for query, hits in zip(selected, results, strict=True):
            for hit in hits:
                source = self._as_source(hit, query)
                if source is not None:
                    sources.append(source)

        log.info(
            "discovery complete",
            extra={
                "queries": len(selected),
                "snippets": len(sources),
                "unique_domains": len(self._seen_domains),
            },
        )
        return sources

    def reset(self) -> None:
        """Forget which domains this run has seen."""
        self._seen_domains.clear()

    def _as_source(self, hit: SearchHit, query: str) -> Source | None:
        """A hit as a citable-nowhere snippet source, or ``None`` if unusable."""
        raw = (hit.url or "").strip()
        if not raw:
            return None
        try:
            canonical = canonicalize_url(raw)
        except ValueError as exc:
            log.debug("unusable search hit url", extra={"url": raw, "error": str(exc)})
            return None
        if not canonical:
            return None

        # Canonicalisation is permissive by design — it upgrades a bare host to
        # https:// rather than discarding it — so a search result can still come
        # out of it as something no crawler should ever be pointed at: a
        # malformed host, a non-http scheme, or a link to a private address.
        # Discovery results are untrusted input, so the SSRF guard runs here,
        # before the URL reaches anything that fetches.
        try:
            validate_url(canonical, resolve_dns=False)
        except SSRFError as exc:
            log.warning("search hit blocked by url guard", extra={"reason": str(exc)})
            return None

        domain = registrable_domain(canonical) or hit.domain
        if not domain:
            return None
        if domain in self._seen_domains:
            return None
        self._seen_domains.add(domain)

        notes = f"{_SNIPPET_PREFIX}{hit.snippet}" if hit.snippet else None
        return Source(
            id=stable_id("snippet", canonical),
            url=canonical,
            canonical_url=canonical,
            title=hit.title,
            # Intentionally no `text`.
            text=None,
            source_type=SourceType.SEARCH_SNIPPET,
            domain=domain,
            discovery_query=query,
            notes=notes,
        )
