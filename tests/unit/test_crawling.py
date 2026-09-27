"""Crawl budgeting and the content cache.

The crawl is the part of this system most able to cause harm if it is wrong, so
these tests are mostly about the limits: the page cap, the single-host rule, the
wall-clock stop, and the guarantee that a competitor's shared boilerplate can
never merge them into one another.
"""

from __future__ import annotations

import asyncio
import time

from app.providers.offline import OfflineWebProvider
from app.schemas.provider import CrawlBudget, FetchResult
from app.services.cache_service import ContentCache
from app.services.crawling_service import CrawlingService, discover_links

HOST = "https://acme.com"


def _page(path: str, text: str = "content", links: str = "") -> FetchResult:
    url = f"{HOST}{path}"
    return FetchResult(
        url=url, final_url=url, status=200, text=text,
        html=f"<html><body>{text}{links}</body></html>", retrieval_status="ok",
    )


def _offline(pages: list[FetchResult]) -> OfflineWebProvider:
    web = OfflineWebProvider()
    for page in pages:
        web.add(page.url, text=page.text, html=page.html)
    return web


class TestCrawlBudget:
    async def test_it_fetches_the_root_when_nothing_is_linked(self) -> None:
        web = _offline([_page("/")])
        report = await CrawlingService(web).crawl_site(HOST)
        assert [r.url for r in report.ok_results] == [f"{HOST}/"]

    async def test_it_follows_links_within_the_page_budget(self) -> None:
        web = _offline([
            _page("/", links='<a href="/pricing">p</a><a href="/about">a</a>'),
            _page("/pricing"), _page("/about"),
        ])
        report = await CrawlingService(web, max_depth=1).crawl_site(HOST)
        assert {r.url for r in report.ok_results} == {
            f"{HOST}/", f"{HOST}/pricing", f"{HOST}/about"
        }

    async def test_it_never_exceeds_max_pages(self) -> None:
        links = "".join(f'<a href="/p{index}">x</a>' for index in range(50))
        pages = [_page("/", links=links)] + [
            _page(f"/p{index}") for index in range(50)
        ]
        report = await CrawlingService(web=_offline(pages), max_depth=1).crawl_site(
            HOST, budget=CrawlBudget(max_pages=5, max_concurrent=3)
        )
        assert report.pages_fetched <= 5

    async def test_it_refuses_to_leave_the_root_host(self) -> None:
        """A site linking to every news site must not drag the crawl along."""
        web = _offline([
            _page("/", links='<a href="https://other.com/x">x</a>'),
            _page("/p", links='<a href="https://other.com/y">y</a>'),
        ])
        report = await CrawlingService(web, max_depth=2).crawl_site(HOST)
        assert all("other.com" not in result.url for result in report.results)

    async def test_it_stops_when_the_time_budget_is_spent(self) -> None:
        """The deadline must bound work in flight, not just decide the next batch.

        Uses real async I/O. A blocking call would stall the event loop, and no
        timeout mechanism could fire — which is a property of blocking code, not
        of the budget.
        """

        class SlowWeb(OfflineWebProvider):
            async def fetch(self, url: str) -> FetchResult:
                await asyncio.sleep(0.05)
                return await super().fetch(url)

        web = SlowWeb()
        for index in range(20):
            web.add(f"{HOST}/p{index}", text="x", html="<html>x</html>")
        web.add(f"{HOST}/", html='<html><body>' + "".join(
            f'<a href="/p{i}">x</a>' for i in range(20)) + "</body></html>", text="root")
        report = await CrawlingService(web, max_depth=1).crawl_site(
            HOST, budget=CrawlBudget(max_pages=20, max_concurrent=1, timeout=0.2)
        )
        assert report.stopped_because == "time budget exhausted"
        assert report.pages_fetched < 20

    async def test_it_does_not_revisit_a_page(self) -> None:
        web = _offline([
            _page("/", links='<a href="/a">1</a><a href="/a">2</a><a href="/b">3</a>'),
            _page("/a", links='<a href="/">home</a>'), _page("/b"),
        ])
        report = await CrawlingService(web, max_depth=2).crawl_site(HOST)
        urls = [result.url for result in report.ok_results]
        assert len(urls) == len(set(urls))

    async def test_a_failing_page_does_not_abort_the_crawl(self) -> None:
        class FlakyWeb(OfflineWebProvider):
            async def fetch(self, url: str) -> FetchResult:
                if url.endswith("/bad"):
                    return FetchResult(url=url, retrieval_status="failed", error="boom")
                return await super().fetch(url)

        web = FlakyWeb()
        web.add(f"{HOST}/", html='<html><body><a href="/bad">b</a>'
                              '<a href="/good">g</a></body></html>', text="root")
        web.add(f"{HOST}/good", text="fine")
        report = await CrawlingService(web, max_depth=1).crawl_site(HOST)
        assert report.pages_failed >= 1
        assert f"{HOST}/good" in {r.url for r in report.ok_results}

    async def test_it_reports_a_blocked_page_without_failing(self) -> None:
        class BlockedWeb(OfflineWebProvider):
            async def fetch(self, url: str) -> FetchResult:
                return FetchResult(url=url, retrieval_status="blocked", error="SSRF")

        report = await CrawlingService(BlockedWeb()).crawl_site(HOST)
        assert report.pages_blocked >= 1
        assert report.ok_results == []


class TestCrawlOrdering:
    async def test_high_priority_pages_are_fetched_first(self) -> None:
        """When the budget runs out it must run out on the least useful page."""
        links = ('<a href="/blog/a">1</a><a href="/blog/b">2</a>'
                 '<a href="/blog/c">3</a><a href="/pricing">4</a>')
        web = _offline([_page("/", links=links), _page("/blog/a"), _page("/blog/b"),
                        _page("/blog/c"), _page("/pricing")])
        report = await CrawlingService(web, max_depth=1).crawl_site(
            HOST, budget=CrawlBudget(max_pages=2, max_concurrent=1)
        )
        assert f"{HOST}/pricing" in {r.url for r in report.ok_results}


class TestDeduplicationAcrossCompetitors:
    async def test_shared_boilerplate_yields_two_sources_per_competitor(self) -> None:
        boiler = "Copyright 2026 Acme Corp. All rights reserved."
        web = _offline([
            _page("/", text=boiler, links='<a href="/legal">l</a>'),
            _page("/legal", text=boiler),
        ])
        report = await CrawlingService(web, max_depth=1).crawl_site(HOST)
        assert len(report.ok_results) == 2

    async def test_one_company_gets_one_entry_for_a_repeated_page(self) -> None:
        web = _offline([_page("/", links='<a href="/a">1</a><a href="/a">2</a>'), _page("/a")])
        report = await CrawlingService(web, max_depth=1).crawl_site(HOST)
        assert len(report.ok_results) == 2


class TestLinkDiscovery:
    def test_it_ignores_non_http_and_duplicate_links(self) -> None:
        result = _page("/", links=(
            '<a href="/a">1</a><a href="/a">2</a><a href="mailto:x@y.com">m</a>'
            '<a href="#top">t</a><a href="javascript:void(0)">j</a>'
            '<a href="https://acme.com/b">3</a>'
        ))
        found = discover_links(result)
        assert found == [f"{HOST}/a", f"{HOST}/b"]

    def test_it_resolves_relative_links_against_the_final_url(self) -> None:
        result = _page("/docs/index.html", links='<a href="../pricing">p</a>')
        assert discover_links(result) == [f"{HOST}/pricing"]

    def test_it_returns_nothing_without_html(self) -> None:
        assert discover_links(FetchResult(url=HOST, text="no html")) == []

    def test_it_survives_malformed_markup(self) -> None:
        result = FetchResult(url=HOST, html="<html><body><a href=/a>unclosed")
        assert isinstance(discover_links(result), list)


class TestContentCache:
    def _result(self, url: str, text: str) -> FetchResult:
        return FetchResult(url=url, final_url=url, status=200, text=text,
                           html=f"<html>{text}</html>")

    def test_a_stored_page_is_a_hit(self) -> None:
        cache = ContentCache()
        cache.put(self._result(f"{HOST}/a", "one"))
        assert cache.get(f"{HOST}/a").hit

    def test_a_missing_page_is_a_miss(self) -> None:
        assert not ContentCache().get(f"{HOST}/a").hit

    def test_a_failed_fetch_is_never_cached(self) -> None:
        cache = ContentCache()
        cache.put(FetchResult(url=f"{HOST}/a", retrieval_status="failed"))
        assert not cache.get(f"{HOST}/a").hit

    def test_tracking_parameters_hit_the_same_entry(self) -> None:
        cache = ContentCache()
        cache.put(self._result(f"{HOST}/a", "one"))
        assert cache.get(f"{HOST}/a?utm_source=x").hit

    def test_a_stale_entry_is_evicted_and_reported(self) -> None:
        cache = ContentCache(ttl_seconds=10)
        cache.put(self._result(f"{HOST}/a", "one"), now=time.monotonic() - 100)
        lookup = cache.get(f"{HOST}/a")
        assert not lookup.hit
        assert lookup.reason == "stale"

    def test_it_detects_changed_content(self) -> None:
        cache = ContentCache()
        entry = cache.put(self._result(f"{HOST}/a", "one"))
        assert entry is not None
        assert cache.changed_since(entry, self._result(f"{HOST}/a", "two"))
        assert not cache.changed_since(entry, self._result(f"{HOST}/a", "one"))

    def test_it_evicts_the_least_recently_used_entry(self) -> None:
        cache = ContentCache(max_entries=2)
        cache.put(self._result(f"{HOST}/a", "1"))
        cache.put(self._result(f"{HOST}/b", "2"))
        cache.get(f"{HOST}/a")  # refresh a
        cache.put(self._result(f"{HOST}/c", "3"))
        assert len(cache) == 2
        assert cache.get(f"{HOST}/b").hit is False
        assert cache.get(f"{HOST}/a").hit
        assert cache.get(f"{HOST}/c").hit

    async def test_a_crawl_reuses_cached_pages(self) -> None:
        calls: list[str] = []

        class CountingWeb(OfflineWebProvider):
            async def fetch(self, url: str) -> FetchResult:
                calls.append(url)
                return await super().fetch(url)

        web = CountingWeb()
        web.add(f"{HOST}/", text="root")
        cache = ContentCache()
        service = CrawlingService(web, cache=cache, max_depth=0)
        await service.crawl_site(HOST)
        await service.crawl_site(HOST)
        assert len(calls) == 1
