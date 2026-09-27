"""Web retrieval behaviour, driven entirely through an injected fetcher.

No test here touches the network. ``HttpWebProvider`` takes its transport as a
callable, so the redirect chain, the size cap, the SSRF guard and the HTML
extraction can all be exercised deterministically — and so can the failure
paths, which are the ones that matter and the ones nobody can reproduce against
a live site.
"""

from __future__ import annotations

import pytest

from app.core.constants import PageType
from app.providers.offline import OfflineWebProvider
from app.providers.web.browser import should_escalate_to_browser
from app.providers.web.http import HttpPayload, HttpWebProvider
from app.schemas.provider import FetchResult
from app.utils.classification import classify_page

LONG_PAGE = "Detailed competitor information. " * 500  # comfortably over 20k chars


def _html(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><title>"
        f"{title}</title></head><body><main><h1>{title}</h1>"
        f"<p>{body}</p></main><nav>nav noise</nav></body></html>"
    )


def payload(
    url: str, *, body: str = "", status: int = 200, content_type: str = "text/html",
    location: str | None = None,
) -> HttpPayload:
    headers = {"content-type": content_type, "content-length": str(len(body))}
    if location:
        headers["location"] = location
    return HttpPayload(status=status, headers=headers, body=body.encode("utf-8"), url=url)


@pytest.fixture()
def provider() -> HttpWebProvider:
    return HttpWebProvider(timeout=5, max_bytes=50_000, user_agent="test-agent/1.0")


class TestHappyPath:
    async def test_fetches_and_extracts_readable_text(self, provider: HttpWebProvider) -> None:
        async def fetcher(url: str) -> HttpPayload:
            return payload(url, body=_html("Pricing", "Starter $10, Pro $50."))

        result = await provider.fetch("https://acme.com/pricing", fetcher=fetcher)
        assert result.ok
        assert result.status == 200
        assert "Starter $10" in result.text
        assert result.title == "Pricing"
        assert result.content_hash
        assert result.provider == "http"

    async def test_a_plain_text_response_is_kept_verbatim(
        self, provider: HttpWebProvider
    ) -> None:
        async def fetcher(url: str) -> HttpPayload:
            return payload(url, body="just words", content_type="text/plain")

        result = await provider.fetch("https://acme.com/robots.txt", fetcher=fetcher)
        assert result.text.strip() == "just words"

    async def test_content_length_measures_extracted_text(
        self, provider: HttpWebProvider
    ) -> None:
        async def fetcher(url: str) -> HttpPayload:
            return payload(url, body=_html("A", "x" * 500))

        result = await provider.fetch("https://acme.com/a", fetcher=fetcher)
        assert result.content_length == len(result.text)


class TestSSRF:
    async def test_a_private_address_is_blocked_before_any_request(self) -> None:
        called: list[str] = []

        async def fetcher(url: str) -> HttpPayload:
            called.append(url)
            return payload(url, body="secret")

        result = await HttpWebProvider().fetch("http://169.254.169.254/latest/meta-data",
                                              fetcher=fetcher)
        assert result.retrieval_status == "blocked"
        assert not called, "a blocked URL must never reach the transport"

    async def test_a_redirect_into_a_private_address_is_blocked(self) -> None:
        async def fetcher(url: str) -> HttpPayload:
            if url.startswith("https://acme.com"):
                return payload(url, status=302, location="http://10.0.0.5/secrets")
            return payload(url, body="internal data")

        result = await HttpWebProvider().fetch("https://acme.com/go", fetcher=fetcher)
        assert result.retrieval_status == "blocked"
        assert "10.0.0.5" in (result.error or "")

    async def test_a_redirect_loop_stops_at_the_limit(self) -> None:
        async def fetcher(url: str) -> HttpPayload:
            return payload(url, status=302, location="https://acme.com/loop")

        result = await HttpWebProvider(max_redirects=3).fetch(
            "https://acme.com/loop", fetcher=fetcher
        )
        assert result.retrieval_status == "failed"
        assert "redirect" in (result.error or "").lower()


class TestLimits:
    async def test_a_response_over_the_cap_is_refused(self) -> None:
        async def fetcher(url: str) -> HttpPayload:
            return payload(url, body="<html><body>" + "A" * 100_000 + "</body></html>")

        result = await HttpWebProvider(max_bytes=1_000).fetch(
            "https://acme.com/big", fetcher=fetcher
        )
        assert result.retrieval_status == "failed"
        assert "too large" in (result.error or "").lower()

    async def test_a_transport_error_becomes_a_failed_result(
        self, provider: HttpWebProvider
    ) -> None:
        async def fetcher(url: str) -> HttpPayload:
            raise OSError("connection reset")

        result = await provider.fetch("https://acme.com/a", fetcher=fetcher)
        assert result.retrieval_status == "failed"
        assert "connection reset" in (result.error or "")

    async def test_a_non_html_content_type_is_not_parsed(self, provider: HttpWebProvider) -> None:
        async def fetcher(url: str) -> HttpPayload:
            return payload(url, body="%PDF-1.4 binary", content_type="application/pdf")

        result = await provider.fetch("https://acme.com/d.pdf", fetcher=fetcher)
        assert result.ok
        assert result.text == ""


class TestStatusHandling:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [(404, "not_found"), (410, "not_found"), (500, "failed"),
         (503, "failed"), (403, "failed")],
    )
    async def test_error_statuses_map_to_a_retrieval_status(
        self, provider: HttpWebProvider, status: int, expected: str
    ) -> None:
        async def fetcher(url: str) -> HttpPayload:
            return payload(url, body="", status=status)

        result = await provider.fetch("https://acme.com/x", fetcher=fetcher)
        assert result.retrieval_status == expected
        assert result.status == status


class TestOfflineWebProvider:
    async def test_serves_a_registered_fixture(self) -> None:
        web = OfflineWebProvider()
        web.add("https://acme.com/pricing", text="Starter $10", title="Pricing")
        result = await web.fetch("https://acme.com/pricing")
        assert result.ok
        assert "Starter $10" in result.text

    async def test_an_unregistered_url_is_not_found_not_invented(self) -> None:
        result = await OfflineWebProvider().fetch("https://acme.com/nope")
        assert result.retrieval_status == "not_found"
        assert result.text == ""

    async def test_offline_fetch_is_reproducible(self) -> None:
        web = OfflineWebProvider()
        web.add("https://acme.com/a", text="stable")
        first = await web.fetch("https://acme.com/a")
        second = await web.fetch("https://acme.com/a")
        assert first.content_hash == second.content_hash

    async def test_offline_browser_delegates_to_the_web_provider(self) -> None:
        from app.providers.offline import OfflineBrowserProvider

        web = OfflineWebProvider()
        web.add("https://acme.com/app", text="rendered")
        browser = OfflineBrowserProvider(web)
        await browser.open("https://acme.com/app")
        assert (await browser.navigate("https://acme.com/app")).ok
        await browser.close()


class TestRealTransportIsNotUsedByDefault:
    def test_the_default_provider_has_no_transport_configured_yet(self) -> None:
        """A provider constructed with no fetcher must not silently do nothing."""
        web = HttpWebProvider()
        assert web.name == "http"


class TestPageClassification:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://example.com/", PageType.HOMEPAGE),
            ("https://example.com/pricing", PageType.PRICING),
            ("https://example.com/blog/post-1", PageType.BLOG),
            ("https://example.com/case-studies/acme", PageType.CASE_STUDY),
            ("https://example.com/customers", PageType.CUSTOMER),
            ("https://example.com/about/team", PageType.ABOUT),
            ("https://example.com/products/widget", PageType.PRODUCT),
            ("https://example.com/features", PageType.FEATURE),
            ("https://example.com/solutions/retail", PageType.SOLUTION),
            ("https://example.com/industries/healthcare", PageType.INDUSTRY),
            ("https://example.com/docs/installation", PageType.DOCUMENTATION),
            ("https://example.com/faq", PageType.FAQ),
            ("https://example.com/changelog/2026-01", PageType.NEWS),
            ("https://example.com/totally/unmapped/thing", PageType.OTHER),
        ],
    )
    def test_path_drives_the_classification(
        self, url: str, expected: PageType
    ) -> None:
        assert classify_page(url) is expected

    def test_a_trailing_slash_does_not_change_the_classification(self) -> None:
        assert classify_page("https://example.com/pricing/") is PageType.PRICING

    def test_host_beats_path_on_a_review_platform(self) -> None:
        assert classify_page("https://www.g2.com/products/thing") is PageType.REVIEW

    def test_a_github_url_is_recognised_by_host(self) -> None:
        assert classify_page("https://github.com/acme/widget") is PageType.GITHUB

    def test_a_social_host_is_a_fallback_not_an_override(self) -> None:
        """A recognised path still classifies; an unmapped one falls to SOCIAL."""
        assert classify_page("https://x.com/docs") is PageType.DOCUMENTATION
        assert classify_page("https://x.com/someaccount") is PageType.SOCIAL

    def test_structured_data_classifies_an_opaque_path(self) -> None:
        html = ('<html><body><div itemscope itemtype="https://schema.org/Review">'
                "<h1>Acme</h1></div></body></html>")
        assert classify_page("https://example.com/post/4821", html) is PageType.REVIEW

    def test_the_title_classifies_when_the_path_does_not(self) -> None:
        html = "<html><head><title>Pricing and Plans - Acme</title></head><body>x</body></html>"
        assert classify_page("https://example.com/p", html) is PageType.PRICING

    def test_classification_never_raises_on_junk(self) -> None:
        for url in ("", "not a url", "https://", "ftp://x.com/a", "///"):
            assert isinstance(classify_page(url, "<html", ""), PageType)


class TestBrowserEscalation:
    def test_an_spa_shell_escalates(self) -> None:
        result = FetchResult(url="https://acme.com/", status=200, text="",
                             html="<html><body><div id=root></div></body></html>")
        assert should_escalate_to_browser(result)

    def test_a_long_text_page_does_not_escalate(self) -> None:
        result = FetchResult(url="https://acme.com/", status=200,
                             text=LONG_PAGE, html="<html><body>x</body></html>")
        assert not should_escalate_to_browser(result)

    def test_a_short_high_value_page_escalates(self) -> None:
        result = FetchResult(url="https://acme.com/pricing", status=200, text="",
                             html="<html><body></body></html>")
        assert should_escalate_to_browser(result)

    def test_a_404_does_not_escalate(self) -> None:
        result = FetchResult(url="https://acme.com/pricing", status=404, text="",
                             retrieval_status="not_found")
        assert not should_escalate_to_browser(result)

    def test_a_bot_protection_refusal_escalates(self) -> None:
        """A 403 is usually a user-agent gate, and a real browser gets through."""
        result = FetchResult(url="https://acme.com/", status=403, text="",
                             retrieval_status="failed")
        assert should_escalate_to_browser(result)

    def test_a_server_error_does_not_escalate(self) -> None:
        """A browser cannot repair a broken server, and a retry adds load."""
        result = FetchResult(url="https://acme.com/", status=503, text="",
                             retrieval_status="failed")
        assert not should_escalate_to_browser(result)

    def test_a_rate_limit_does_not_escalate(self) -> None:
        result = FetchResult(url="https://acme.com/", status=429, text="",
                             retrieval_status="failed")
        assert not should_escalate_to_browser(result)

    def test_an_interactive_path_escalates_even_with_plenty_of_text(self) -> None:
        result = FetchResult(url="https://acme.com/app/dashboard", status=200,
                             text=LONG_PAGE, html="<html><body>x</body></html>")
        assert should_escalate_to_browser(result)

    def test_a_blocked_page_never_escalates(self) -> None:
        result = FetchResult(url="http://10.0.0.1/", retrieval_status="blocked",
                             text="")
        assert not should_escalate_to_browser(result)


class TestUnavailableBrowser:
    async def test_it_explains_why_it_cannot_render(self) -> None:
        from app.providers.browser.unavailable import (
            BrowserUnavailableError,
            UnavailableBrowserProvider,
        )

        provider = UnavailableBrowserProvider("playwright is not installed")
        assert not provider.available()
        with pytest.raises(BrowserUnavailableError) as excinfo:
            await provider.navigate("https://acme.com/")
        assert "playwright" in str(excinfo.value)
        assert "ENABLE_BROWSER_USE" in str(excinfo.value)

