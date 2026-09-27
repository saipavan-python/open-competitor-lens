"""Content de-duplication rules.

The three cases below are the whole specification, and they pull in opposite
directions. A crawler that only keys on content hash collapses two competitors
that share a legal boilerplate footer into one competitor. A crawler that only
keys on URL records the same page twice every time it is re-fetched. A crawler
that keys on content alone also silently merges genuinely different pages that
happen to render the same text.

The resolution: identity is the canonical URL, content identity is a hash of
that URL *plus* the content, and the scope is per company.
"""

from __future__ import annotations

import pytest

from app.providers.base import WebProvider
from app.providers.offline import OfflineWebProvider
from app.schemas.provider import FetchResult
from app.services.extraction_service import Deduplicator

BOILERPLATE = "Copyright 2026. All rights reserved. Terms Privacy."


def _page(url: str, text: str) -> FetchResult:
    return FetchResult(
        url=url,
        final_url=url,
        status=200,
        text=text,
        html=f"<html><body>{text}</body></html>",
        retrieval_status="ok",
    )


class TestSourceIdentity:
    def test_same_url_fetched_twice_is_one_source(self) -> None:
        dedupe = Deduplicator()
        first = dedupe.add(_page("https://acme.com/pricing", "Starter $10"))
        second = dedupe.add(_page("https://acme.com/pricing", "Starter $10"))
        assert first.is_new
        assert second.is_duplicate
        assert not second.is_new
        assert dedupe.count == 1

    def test_two_urls_with_identical_content_are_two_sources(self) -> None:
        """Same company, same text, different URLs: both are real sources."""
        dedupe = Deduplicator()
        assert dedupe.add(_page("https://acme.com/about", BOILERPLATE)).is_new
        assert dedupe.add(_page("https://acme.com/about-us", BOILERPLATE)).is_new
        assert dedupe.count == 2

    def test_identical_boilerplate_across_competitors_is_never_merged(self) -> None:
        """The failure mode this whole class exists to prevent."""
        acme = Deduplicator(scope="acme")
        beta = Deduplicator(scope="beta")
        assert acme.add(_page("https://acme.com/legal", BOILERPLATE)).is_new
        assert beta.add(_page("https://beta.com/legal", BOILERPLATE)).is_new
        assert acme.count == 1
        assert beta.count == 1

    def test_scopes_are_independent_of_construction_order(self) -> None:
        first = Deduplicator(scope="acme")
        first.add(_page("https://acme.com/a", "one"))
        second = Deduplicator(scope="beta")
        assert second.add(_page("beta.com/a", "one")).is_new

    def test_query_tracking_parameters_do_not_create_a_second_source(self) -> None:
        dedupe = Deduplicator()
        assert dedupe.add(_page("https://acme.com/p?utm_source=x", "Price $10")).is_new
        assert dedupe.add(_page("https://acme.com/p", "Price $10")).is_duplicate
        assert dedupe.count == 1

    def test_a_url_with_a_fragment_is_the_same_source(self) -> None:
        dedupe = Deduplicator()
        dedupe.add(_page("https://acme.com/docs#install", "Install here"))
        assert dedupe.add(_page("https://acme.com/docs", "Install here")).is_duplicate


class TestChangeDetection:
    def test_same_url_with_new_content_is_an_update(self) -> None:
        """Identity holds, but the content changed — that is a change event."""
        dedupe = Deduplicator()
        dedupe.add(_page("https://acme.com/pricing", "Starter $10"))
        decision = dedupe.add(_page("https://acme.com/pricing", "Starter $12"))
        assert decision.is_duplicate
        assert decision.changed
        assert dedupe.count == 1

    def test_unchanged_content_is_not_reported_as_changed(self) -> None:
        dedupe = Deduplicator()
        dedupe.add(_page("https://acme.com/pricing", "Starter $10"))
        assert not dedupe.add(_page("https://acme.com/pricing", "Starter $10")).changed

    def test_content_key_depends_on_the_url_not_just_the_bytes(self) -> None:
        """Identical bytes at two URLs must not share a content key."""
        dedupe = Deduplicator()
        one = dedupe.add(_page("https://acme.com/a", "same"))
        two = dedupe.add(_page("https://acme.com/b", "same"))
        assert one.content_key != two.content_key

    def test_a_known_url_can_be_asked_for_without_recording_it(self) -> None:
        dedupe = Deduplicator()
        dedupe.add(_page("https://acme.com/pricing", "x"))
        assert dedupe.seen("https://acme.com/pricing")
        assert dedupe.seen("https://acme.com/pricing?utm_campaign=q")
        assert not dedupe.seen("https://acme.com/enterprise")


class TestFailedRetrievals:
    def test_a_failed_fetch_does_not_claim_a_source(self) -> None:
        """A page we could not read is not a source, and must not shadow a retry."""
        dedupe = Deduplicator()
        failed = FetchResult(
            url="https://acme.com/pricing", status=503,
            retrieval_status="failed", error="upstream 503",
        )
        assert not dedupe.add(failed).recorded
        assert dedupe.count == 0
        assert dedupe.add(_page("https://acme.com/pricing", "Starter $10")).is_new
        assert dedupe.count == 1

    def test_a_missing_page_is_not_recorded(self) -> None:
        dedupe = Deduplicator()
        dedupe.add(FetchResult(url="https://acme.com/gone", status=404,
                              retrieval_status="not_found"))
        assert dedupe.count == 0

    def test_a_blocked_url_is_not_recorded(self) -> None:
        dedupe = Deduplicator()
        dedupe.add(FetchResult(url="http://169.254.169.254/", status=None,
                              retrieval_status="blocked", error="SSRF"))
        assert dedupe.count == 0


class TestDeduplicateMany:
    def test_bulk_dedupe_preserves_first_occurrence_order(self) -> None:
        dedupe = Deduplicator()
        pages = [
            _page("https://acme.com/a", "one"),
            _page("https://acme.com/a", "one"),
            _page("https://acme.com/b", "two"),
            _page("https://acme.com/a", "one"),
        ]
        kept = dedupe.deduplicate(pages)
        assert [page.url for page in kept] == [
            "https://acme.com/a", "https://acme.com/b"
        ]

    def test_duplicate_decisions_are_reported_alongside_the_kept_pages(self) -> None:
        dedupe = Deduplicator()
        pages = [
            _page("https://acme.com/a", "one"),
            _page("https://acme.com/a", "one"),
        ]
        kept, decisions = dedupe.deduplicate(pages, return_decisions=True)
        assert len(kept) == 1
        assert [decision.is_duplicate for decision in decisions] == [False, True]

    def test_reset_clears_the_scope(self) -> None:
        dedupe = Deduplicator()
        dedupe.add(_page("https://acme.com/a", "one"))
        dedupe.reset()
        assert dedupe.count == 0
        assert dedupe.add(_page("https://acme.com/a", "one")).is_new


class TestWebProviderProtocolConformance:
    @pytest.mark.parametrize(
        "cls", [OfflineWebProvider], ids=lambda c: c.__name__
    )
    def test_offline_provider_satisfies_the_protocol(self, cls: type) -> None:
        assert isinstance(cls(), WebProvider)
