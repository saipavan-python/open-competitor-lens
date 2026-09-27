"""Breadth-limited crawling of one competitor's site.

Three constraints shape this, and all three exist to stop a crawl from becoming
a denial-of-service against someone else's website:

- **One host.** Links are followed within the root domain. A site that links to
  every news article on the internet must not drag the crawler with it.
- **A page budget.** Hard-capped by ``max_pages``, and by depth.
- **A wall-clock budget.** A crawl that has used its time stops taking on new
  work, even if pages remain.

Discovery comes from the fetched HTML, and candidates are ordered by
:func:`app.utils.urls.url_priority` so the commercially decisive pages are
fetched first. When the budget runs out, it should run out on the least useful
page, not a random one.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

from app.core.logging import get_logger
from app.schemas.provider import CrawlBudget, FetchResult
from app.services.cache_service import ContentCache
from app.services.extraction_service import DedupDecision, Deduplicator
from app.utils.concurrency import gather_settled
from app.utils.urls import canonicalize_url, url_priority, validate_url

log = get_logger(__name__)

FetchCallable = Callable[[str], "asyncio.Future[FetchResult] | FetchResult"]

#: Link attributes worth following. ``rel=nofollow`` is deliberately not treated
#: as a prohibition: it is a crawler hint to search engines, and reading a
#: competitor's own sitemap-linked pages is ordinary research.
_LINK_TAGS = ("a", "area", "link")


@dataclass
class CrawlReport:
    """What a crawl did, including what it could not do."""

    root_url: str
    results: list[FetchResult] = field(default_factory=list)
    decisions: list[DedupDecision] = field(default_factory=list)
    pages_fetched: int = 0
    pages_blocked: int = 0
    pages_failed: int = 0
    stopped_because: str = "budget exhausted"

    @property
    def ok_results(self) -> list[FetchResult]:
        return [result for result in self.results if result.ok]

    def summary(self) -> dict[str, int | str]:
        return {
            "pages_fetched": self.pages_fetched,
            "pages_blocked": self.pages_blocked,
            "pages_failed": self.pages_failed,
            "stopped_because": self.stopped_because,
        }


class CrawlingService:
    """Fetches a competitor's site within explicit budgets."""

    def __init__(
        self,
        web: object,
        *,
        budget: CrawlBudget | None = None,
        cache: ContentCache | None = None,
        max_depth: int = 2,
        respect_cache: bool = True,
    ) -> None:
        self.web = web
        self.budget = budget or CrawlBudget()
        self.cache = cache
        self.max_depth = max_depth
        self.respect_cache = respect_cache
        self.max_concurrent = max(1, self.budget.max_concurrent)

    async def crawl_site(
        self, root_url: str, *, budget: CrawlBudget | None = None
    ) -> CrawlReport:
        """Crawl from ``root_url``, returning one report."""
        budget = budget or self.budget
        deduplicator = Deduplicator(scope=canonicalize_url(root_url) or root_url)
        report = CrawlReport(root_url=root_url)
        deadline = time.monotonic() + budget.timeout
        host = _host_of(root_url)

        queue: list[tuple[str, int]] = [(root_url, 0)]
        queued: set[str] = {self._key(root_url)}
        fetched: set[str] = set()

        while queue and report.pages_fetched < budget.max_pages:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                report.stopped_because = "time budget exhausted"
                break
            # Take the most valuable pages still outstanding, so that when the
            # budget runs out it runs out on the least useful page rather than
            # on whichever link happened to appear first in the markup.
            queue.sort(key=lambda pair: (url_priority(pair[0]), pair[1]))
            batch: list[tuple[str, int]] = []
            while queue and report.pages_fetched + len(batch) < budget.max_pages:
                batch.append(queue.pop(0))
            if not batch:
                break

            batch.sort(key=lambda pair: url_priority(pair[0]))
            settled = await gather_settled(
                [self._fetch(url) for url, _ in batch],
                limit=self.max_concurrent,
                deadline=remaining,
            )
            if any(isinstance(error, TimeoutError) for _, error in settled):
                report.stopped_because = "time budget exhausted"

            for (url, depth), outcome in zip(batch, settled, strict=True):
                fetched.add(self._key(url))
                result, error = outcome
                if error is not None or result is None:
                    report.pages_failed += 1
                    result = FetchResult(
                        url=url, retrieval_status="failed", error=str(error)
                    )
                if result.retrieval_status == "blocked":
                    report.pages_blocked += 1
                elif not result.ok:
                    report.pages_failed += 1
                else:
                    report.pages_fetched += 1
                    result.page_type = result.page_type or None

                decision = deduplicator.add(result)
                report.decisions.append(decision)
                if decision.should_keep:
                    report.results.append(result)

                if result.ok and depth < self.max_depth:
                    for link in discover_links(result):
                        if _host_of(link) != host:
                            continue
                        key = self._key(link)
                        if key in queued or key in fetched:
                            continue
                        try:
                            validate_url(link)
                        except Exception:  # noqa: BLE001 - a bad link is not a crawl error
                            continue
                        queued.add(key)
                        queue.append((link, depth + 1))

        if not queue and report.stopped_because == "budget exhausted":
            report.stopped_because = "no further pages to crawl"
        return report

    @staticmethod
    def _key(url: str) -> str:
        return canonicalize_url(url) or url

    async def _fetch(self, url: str) -> FetchResult:
        if self.cache is not None and self.respect_cache:
            cached = self.cache.get(url)
            if cached.hit and cached.entry is not None:
                return cached.entry.result
        result = await self.web.fetch(url)  # type: ignore[attr-defined]
        if self.cache is not None and result.ok:
            self.cache.put(result)
        return result


def _host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def discover_links(result: FetchResult, *, limit: int = 200) -> list[str]:
    """Extract same-page links from a fetched result.

    Falls back to nothing rather than to a regex over raw HTML when the parser
    is unavailable: a sloppy link extractor is how a crawler ends up requesting
    URLs out of JavaScript source.
    """
    html = result.html or ""
    if not html:
        return []
    try:
        from bs4 import BeautifulSoup
    except ImportError:  # pragma: no cover - bs4 is a declared dependency
        log.warning("beautifulsoup unavailable; no links discovered")
        return []

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as exc:  # noqa: BLE001 - malformed markup is common
        log.debug("link discovery failed to parse", extra={"error": str(exc)})
        return []

    base = result.final_url or result.url
    links: list[str] = []
    seen: set[str] = set()
    for tag in soup.find_all(_LINK_TAGS):
        href = tag.get("href")
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        absolute = urljoin(base, href.strip())
        if not absolute.startswith(("http://", "https://")):
            continue
        key = canonicalize_url(absolute) or absolute
        if key in seen:
            continue
        seen.add(key)
        links.append(absolute)
        if len(links) >= limit:
            break
    return links
