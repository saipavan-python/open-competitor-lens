"""The browser provider, driven with fakes that stand in for Playwright.

Found while reviewing the branch before publishing, and the provider had no tests
at all. It carried two defects that a browser that is never launched cannot show
you, so these tests fake the Playwright surface rather than requiring a browser
install.

The one that matters is the redirect. ``validate_url`` was called on the URL passed
to ``goto()`` and nowhere else, while Playwright follows redirects itself. A page
that answers ``302`` with a location of ``http://169.254.169.254/latest/meta-data/``
had its response body placed in ``FetchResult.text`` - which is citable report
content. The docstring claimed "the same URL guard is applied here as at the HTTP
boundary"; it was not. The guard now sits on the route, and these tests hold it
there.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.providers.browser.browser_use import BrowserUseProvider

METADATA_URL = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class FakeResponse:
    def __init__(self, status: int = 200) -> None:
        self.status = status


class FakeRequest:
    def __init__(self, url: str) -> None:
        self.url = url


class FakeRoute:
    def __init__(self, url: str) -> None:
        self.request = FakeRequest(url)
        self.aborted = False
        self.continued = False

    async def abort(self) -> None:
        self.aborted = True

    async def continue_(self) -> None:
        self.continued = True


class FakePage:
    """Just the surface ``navigate`` touches."""

    def __init__(
        self,
        *,
        final_url: str = "https://acme.com/",
        status: int | None = 200,
        text: str = "Acme sells a CRM. Starter is $10 per user per month.",
        html: str = "<html><body>Acme</body></html>",
        title: str = "Acme",
        redirects_to: str | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.url = final_url
        self._status = status
        self._text = text
        self._html = html
        self._title = title
        self._redirects_to = redirects_to
        self._raises = raises
        self.goto_calls: list[str] = []
        self.text_reads = 0

    async def goto(self, url: str, **kwargs: Any) -> FakeResponse:
        self.goto_calls.append(url)
        if self._raises is not None:
            raise self._raises
        if self._redirects_to is not None:
            self.url = self._redirects_to
        return FakeResponse(self._status) if self._status is not None else None

    async def title(self) -> str:
        return self._title

    async def inner_text(self, selector: str) -> str:
        self.text_reads += 1
        return self._text

    async def content(self) -> str:
        return self._html


class FakeContext:
    def __init__(self) -> None:
        self.routes: list[tuple[str, Any]] = []
        self.page = FakePage()
        self.closed = False

    async def route(self, pattern: str, handler: Any) -> None:
        self.routes.append((pattern, handler))

    async def new_page(self) -> FakePage:
        return self.page

    async def close(self) -> None:
        self.closed = True


class FakeBrowser:
    def __init__(self) -> None:
        self.contexts: list[FakeContext] = []

    async def new_context(self, **kwargs: Any) -> FakeContext:
        context = FakeContext()
        self.contexts.append(context)
        return context


def provider_with(page: FakePage) -> BrowserUseProvider:
    """A provider mid-session: a page is already open, no browser needed."""
    provider = BrowserUseProvider()
    provider._page = page
    provider._context = FakeContext()
    return provider


class TestTheRouteGuardRefusesInternalHosts:
    async def test_a_request_to_the_metadata_endpoint_is_aborted(self) -> None:
        """The credential-theft case: aborted, and not continued."""
        provider = BrowserUseProvider()
        route = FakeRoute(METADATA_URL)
        await provider._guard_route(route)
        assert route.aborted
        assert not route.continued

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1/admin",
            "http://10.0.0.5/internal",
            "http://[::1]/",
            "file:///etc/passwd",
        ],
    )
    async def test_internal_schemes_and_hosts_are_aborted(self, url: str) -> None:
        provider = BrowserUseProvider()
        route = FakeRoute(url)
        await provider._guard_route(route)
        assert route.aborted, f"{url} should have been refused"

    async def test_a_public_request_is_allowed_through(self) -> None:
        provider = BrowserUseProvider()
        route = FakeRoute("https://acme.com/pricing")
        await provider._guard_route(route)
        assert route.continued
        assert not route.aborted

    async def test_the_guard_is_installed_on_the_context(self) -> None:
        """Without the registration, the handler above is never called."""
        provider = BrowserUseProvider()
        provider._browser = FakeBrowser()
        await provider._replace_context()
        patterns = [pattern for pattern, _ in provider._browser.contexts[0].routes]
        assert patterns == ["**/*"]

    async def test_every_new_context_gets_the_guard(self) -> None:
        """``open()`` runs per escalated page, so each context needs it."""
        provider = BrowserUseProvider()
        provider._browser = FakeBrowser()
        await provider._replace_context()
        await provider._replace_context()
        for context in provider._browser.contexts:
            assert context.routes, "a context was created without the SSRF guard"


class TestNavigationRefusesWhatItReaches:
    def test_a_page_that_redirects_to_the_metadata_endpoint_is_refused(self) -> None:
        """The body must not be read, let alone hashed into the report."""
        page = FakePage(redirects_to=METADATA_URL)
        result = run(provider_with(page).navigate("https://acme.com/"))
        assert result.retrieval_status == "blocked"
        assert not result.ok
        assert page.text_reads == 0, "the body was read despite the block"

    def test_a_landing_on_a_blocked_host_is_blocked_not_ok(self) -> None:
        result = run(provider_with(FakePage(final_url=METADATA_URL)).navigate("https://acme.com/"))
        assert result.retrieval_status == "blocked"
        assert result.error

    def test_a_blocked_entry_url_is_refused_before_any_navigation(self) -> None:
        page = FakePage()
        result = run(provider_with(page).navigate("http://127.0.0.1/admin"))
        assert result.retrieval_status == "blocked"
        assert page.goto_calls == []

    def test_a_navigation_error_is_returned_rather_than_raised(self) -> None:
        """A dead page is a finding for the report, not a crashed run."""
        page = FakePage(raises=RuntimeError("net::ERR_CONNECTION_REFUSED"))
        result = run(provider_with(page).navigate("https://acme.com/"))
        assert result.retrieval_status == "failed"
        assert "ERR_CONNECTION_REFUSED" in (result.error or "")

    def test_a_read_failure_is_returned_rather_than_raised(self) -> None:
        provider = provider_with(FakePage())

        async def boom(selector: str) -> str:
            raise RuntimeError("target closed")

        provider._page.inner_text = boom  # type: ignore[method-assign]
        result = run(provider.navigate("https://acme.com/"))
        assert result.retrieval_status == "failed"


class TestTheRealStatusIsRecorded:
    def test_a_404_is_not_recorded_as_a_successful_retrieval(self) -> None:
        """Hardcoded 200 put a Playwright error page into the content hash."""
        result = run(provider_with(FakePage(status=404)).navigate("https://acme.com/gone"))
        assert result.status == 404
        assert result.retrieval_status == "not_found"
        assert not result.ok
        assert result.content_hash is None

    def test_a_403_is_recorded_as_failed(self) -> None:
        result = run(provider_with(FakePage(status=403)).navigate("https://acme.com/"))
        assert result.status == 403
        assert result.retrieval_status == "failed"

    def test_a_server_error_is_recorded_as_failed(self) -> None:
        result = run(provider_with(FakePage(status=500)).navigate("https://acme.com/"))
        assert result.status == 500
        assert result.retrieval_status == "failed"

    def test_a_successful_page_carries_its_own_status_and_hash(self) -> None:
        result = run(provider_with(FakePage(status=200)).navigate("https://acme.com/"))
        assert result.ok
        assert result.status == 200
        assert result.content_hash
        assert "Acme sells a CRM" in result.text


class TestRenderedPagesAreBounded:
    def test_an_oversized_page_is_refused(self) -> None:
        """The HTTP provider caps the body; the browser had no cap at all."""
        provider = provider_with(FakePage(text="x" * 5_000))
        provider.max_bytes = 1_000
        result = run(provider.navigate("https://acme.com/"))
        assert result.retrieval_status == "failed"
        assert "too large" in (result.error or "")
        assert not result.text


class TestContextsAreNotLeaked:
    async def test_a_previous_context_is_closed_when_the_next_is_opened(self) -> None:
        """``open()`` runs per page, so this leaked one Chromium context each."""
        provider = BrowserUseProvider()
        provider._browser = FakeBrowser()
        await provider._replace_context()
        first = provider._browser.contexts[0]
        await provider._replace_context()
        assert first.closed
        assert not provider._browser.contexts[1].closed
