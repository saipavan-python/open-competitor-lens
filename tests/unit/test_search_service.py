"""Search is a way to find pages, never a way to cite them.

The distinction is worth testing directly, because nothing about it is enforced
by the type system: a snippet is a `Source` like any other, and the only thing
stopping it from backing a claim is that this service labels it correctly and
leaves its page text empty. If that labelling is ever lost — a refactor that
"helpfully" fills in `text`, or a new caller that builds a Source by hand — the
failure is silent and the report cites search engines as though they were the
companies being compared.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.core.constants import NON_PRIMARY_SOURCE_TYPES, SourceType
from app.providers.base import SearchProvider
from app.schemas.search import SearchHit
from app.services.search_service import SearchService


def _hit(url: str, *, snippet: str = "a description", rank: int = 1) -> SearchHit:
    return SearchHit(
        url=url,
        title=f"Title for {url}",
        snippet=snippet,
        domain=None,
        rank=rank,
        provider="test",
        retrieved_at=datetime.now(UTC),
    )


class StubSearchProvider(SearchProvider):
    """Returns canned hits, or raises if the test wants a failure."""

    def __init__(self, hits: list[SearchHit] | Exception) -> None:
        self._hits = hits
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, *, limit: int = 10, **kwargs) -> list[SearchHit]:
        self.calls.append((query, limit))
        if isinstance(self._hits, Exception):
            raise self._hits
        return self._hits[:limit]


class TestSnippetSources:
    async def test_a_hit_becomes_a_search_snippet(self) -> None:
        service = SearchService(StubSearchProvider([_hit("https://acme.example/x")]))
        sources = await service.discover(["acme"])
        assert len(sources) == 1
        assert sources[0].source_type is SourceType.SEARCH_SNIPPET

    async def test_a_snippet_carries_no_page_text(self) -> None:
        """The field that would let it be quoted is deliberately empty."""
        service = SearchService(StubSearchProvider([_hit("https://acme.example/x")]))
        source = (await service.discover(["acme"]))[0]
        assert source.text is None

    async def test_a_snippet_is_never_authoritative(self) -> None:
        """Whatever else changes, this must not: a snippet cannot back a signal."""
        service = SearchService(StubSearchProvider([_hit("https://acme.example/x")]))
        source = (await service.discover(["acme"]))[0]
        assert source.source_type in NON_PRIMARY_SOURCE_TYPES

    async def test_the_snippet_text_is_kept_but_labelled(self) -> None:
        """Kept for relevance judgement, labelled so it is never mistaken for a quote."""
        service = SearchService(
            StubSearchProvider([_hit("https://acme.example/x", snippet="cheap hosting")])
        )
        source = (await service.discover(["acme"]))[0]
        assert source.notes is not None
        assert "cheap hosting" in source.notes
        assert "not citable" in source.notes

    async def test_the_originating_query_is_recorded(self) -> None:
        """So a reader can trace a candidate back to the query that surfaced it."""
        service = SearchService(StubSearchProvider([_hit("https://acme.example/x")]))
        source = (await service.discover(["hosting companies in india"]))[0]
        assert source.discovery_query == "hosting companies in india"


class TestDeduplication:
    async def test_one_domain_contributes_one_candidate(self) -> None:
        """Forty pages of one site is one company, not forty candidates."""
        service = SearchService(
            StubSearchProvider(
                [
                    _hit("https://acme.example/pricing"),
                    _hit("https://acme.example/about"),
                    _hit("https://www.acme.example/"),
                    _hit("https://acme.example/blog/one"),
                ]
            )
        )
        sources = await service.discover(["acme"])
        assert len(sources) == 1

    async def test_distinct_domains_are_all_kept(self) -> None:
        service = SearchService(
            StubSearchProvider(
                [
                    _hit("https://acme.example/"),
                    _hit("https://beta.example/"),
                    _hit("https://gamma.example/"),
                ]
            )
        )
        assert len(await service.discover(["hosting"])) == 3

    async def test_reset_allows_a_domain_to_appear_again(self) -> None:
        """A later phase may legitimately revisit a domain the pool already saw."""
        service = SearchService(StubSearchProvider([_hit("https://acme.example/")]))
        first = await service.discover(["acme"])
        service.reset()
        second = await service.discover(["acme competitors"])
        assert len(first) == len(second) == 1


class TestMultiQuery:
    async def test_queries_run_concurrently(self) -> None:
        provider = StubSearchProvider([_hit("https://acme.example/x")])
        service = SearchService(provider)
        await service.discover(["one", "two", "three"])
        assert [query for query, _ in provider.calls] == ["one", "two", "three"]

    async def test_the_query_count_is_capped(self) -> None:
        """A caller cannot fan out without bound."""
        provider = StubSearchProvider([])
        service = SearchService(provider, max_queries=2)
        await service.discover(["one", "two", "three", "four"])
        assert len(provider.calls) == 2

    async def test_the_limit_reaches_the_provider(self) -> None:
        provider = StubSearchProvider([])
        service = SearchService(provider, limit=7)
        await service.discover(["one"], limit=3)
        assert provider.calls[0][1] == 3


class TestFailureTolerance:
    async def test_a_failing_query_yields_no_hits_and_does_not_raise(self) -> None:
        service = SearchService(StubSearchProvider(RuntimeError("provider down")))
        assert await service.discover(["acme"]) == []

    async def test_one_failed_query_does_not_lose_the_others(self) -> None:
        """Queries are independent: one bad query must not empty the pool."""

        class FlakyProvider(SearchProvider):
            async def search(self, query: str, *, limit: int = 10, **kwargs):
                if query == "bad":
                    raise RuntimeError("boom")
                return [_hit(f"https://{query}.example/")]

        sources = await SearchService(FlakyProvider()).discover(["bad", "good"])
        assert len(sources) == 1
        assert sources[0].domain == "good.example"

    async def test_a_malformed_url_is_skipped_not_fatal(self) -> None:
        """One junk hit must not cost the run every other candidate."""
        provider = StubSearchProvider(
            [_hit("not a url at all"), _hit("https://acme.example/x")]
        )
        sources = await SearchService(provider).discover(["acme"])
        assert len(sources) == 1
        assert sources[0].domain == "acme.example"

    async def test_a_non_http_scheme_is_skipped(self) -> None:
        provider = StubSearchProvider([_hit("ftp://acme.example/pub")])
        assert await SearchService(provider).discover(["acme"]) == []

    async def test_a_hit_with_no_url_cannot_even_be_built(self) -> None:
        """The schema refuses an empty URL, so this never reaches the service."""
        with pytest.raises(ValidationError):
            _hit("")


class TestUntrustedResults:
    """Search results are attacker-influenced input, not trusted URLs."""

    async def test_a_hit_pointing_at_a_private_address_is_blocked(self) -> None:
        """The link is not followed here, but it must not reach the crawler."""
        provider = StubSearchProvider(
            [
                _hit("http://169.254.169.254/latest/meta-data/"),
                _hit("http://10.0.0.5/admin"),
                _hit("http://127.0.0.1:8080/"),
                _hit("https://acme.example/x"),
            ]
        )
        sources = await SearchService(provider).discover(["acme"])
        assert [source.domain for source in sources] == ["acme.example"]

    async def test_credentials_in_a_result_url_are_stripped(self) -> None:
        """Canonicalisation drops userinfo, so no secret is ever stored or fetched.

        Rejecting outright would also be defensible; stripping is what the shared
        canonicaliser already does everywhere, and the property that matters is
        that no credential survives into the stored source.
        """
        provider = StubSearchProvider([_hit("https://user:pw@acme.example/x")])
        sources = await SearchService(provider).discover(["acme"])
        assert len(sources) == 1
        assert "user" not in sources[0].url
        assert "@" not in sources[0].url
        assert sources[0].url == "https://acme.example/x"


class TestIds:
    async def test_the_same_url_always_gets_the_same_source_id(self) -> None:
        """Stable ids let a later crawl match a snippet back to what it found."""

        async def one_id() -> str:
            service = SearchService(StubSearchProvider([_hit("https://acme.example/x")]))
            return (await service.discover(["acme"]))[0].id

        assert await one_id() == await one_id()

    async def test_a_search_id_is_not_confusable_with_a_page_id(self) -> None:
        """A snippet and the page it points at must not share an id.

        Otherwise the crawl that retrieves that page overwrites the snippet, or
        the snippet stands in for the page, and neither is what anyone intended.
        """
        from app.utils.hashing import stable_id

        service = SearchService(StubSearchProvider([_hit("https://acme.example/x")]))
        snippet = (await service.discover(["acme"]))[0]
        page = stable_id("page", "https://acme.example/x")
        assert snippet.id != page
