"""A real browser, for pages that server-side retrieval cannot reach.

Playwright is imported lazily, inside the methods that need it. Making it a
module-scope import would turn an optional integration into a hard startup
requirement for every deployment, including the ones that never escalate.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.logging import get_logger
from app.providers.browser.unavailable import BrowserUnavailableError
from app.schemas.provider import FetchResult
from app.utils.classification import classify_page
from app.utils.hashing import sha256_text
from app.utils.urls import SSRFError, validate_url

log = get_logger(__name__)

#: A crawler identifies itself honestly, because a site operator seeing this in
#: their logs is entitled to know what is visiting them. An earlier version
#: advertised "respects robots.txt" and nothing in this file or in the HTTP
#: provider ever fetched or checked robots.txt - a claim made to third parties
#: that was simply untrue, in a string they log and in terms-of decisions they
#: may take on the strength of it. Say what the thing is.
DEFAULT_USER_AGENT = "open-competitor-lens/0.1 (+competitive research)"


class BrowserUseProvider:
    """Playwright-backed browser following the plan's browser_use integration."""

    name = "browser_use"

    def __init__(self, *, timeout: int = 60, headless: bool = True,
                 user_agent: str | None = None, max_bytes: int = 5_000_000) -> None:
        self.timeout = timeout
        self.headless = headless
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        #: Cap on a rendered page. The HTTP provider enforces this while reading
        #: the body; without it here, a page that renders to 200MB of DOM is read
        #: into memory whole and then into the report.
        self.max_bytes = max_bytes
        self._playwright: Any = None
        self._context: Any = None
        self._browser: Any = None
        self._page: Any = None

    def available(self) -> bool:
        try:
            import playwright  # noqa: F401
        except ImportError:
            return False
        return True

    async def _guard_route(self, route: Any) -> None:
        """Refuse any request the SSRF guard would refuse.

        The guard has to sit on the route, not on the entry point. Playwright
        follows redirects itself and loads subresources, so validating only the
        URL passed to ``goto()`` validates nothing: a page that 302s to
        ``http://169.254.169.254/...``, or fetches it in-page, reaches it either
        way, and the response body ends up in ``FetchResult.text`` - that is, in
        citable report content. The HTTP provider re-validates every hop for the
        same reason; this is the browser equivalent.
        """
        try:
            validate_url(route.request.url)
        except SSRFError as exc:
            log.warning(
                "blocked browser request",
                extra={"url": route.request.url, "reason": str(exc)},
            )
            await route.abort()
            return
        await route.continue_()

    async def open(self, url: str) -> None:
        """Open ``url`` in a fresh context, guarding every request it makes.

        A browser can reach internal hosts that a plain fetch would never try —
        redirects, in-page fetches, and internal links all become reachable — so
        :meth:`_guard_route` is installed on the context before anything loads.
        """
        validate_url(url)
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise BrowserUnavailableError("playwright is not installed") from exc

        if self._browser is None:
            self._playwright = await async_playwright().start()
            try:
                self._browser = await self._playwright.chromium.launch(
                    headless=self.headless
                )
            except Exception as exc:  # noqa: BLE001 - browser binaries are optional
                await self.close()
                raise BrowserUnavailableError(
                    f"could not launch chromium: {exc}"
                ) from exc
        await self._replace_context()
        await self._page.goto(url, timeout=self.timeout * 1000, wait_until="domcontentloaded")

    async def _replace_context(self) -> None:
        """Start a new context, closing the previous one first.

        ``open()`` runs per escalated page, so without this every escalation
        leaked a Chromium context - the old one was never closed and
        :meth:`close` only ever tore down the most recent one.
        """
        if self._context is not None:
            try:
                await self._context.close()
            except Exception as exc:  # noqa: BLE001 - teardown is best effort
                log.debug("browser context close failed", extra={"error": str(exc)})
        self._context = await self._browser.new_context(user_agent=self.user_agent)
        await self._context.route("**/*", self._guard_route)
        self._page = await self._context.new_page()

    def _failure(self, url: str, started: float, status: str, error: str) -> FetchResult:
        return FetchResult(
            url=url, final_url=url, status=None, retrieval_status=status, error=error,
            provider=self.name,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    async def navigate(self, url: str) -> FetchResult:
        started = time.monotonic()
        try:
            if self._page is None:
                await self.open(url)
                response = None
            else:
                validate_url(url)
                response = await self._page.goto(
                    url, timeout=self.timeout * 1000, wait_until="domcontentloaded"
                )
        except SSRFError as exc:
            log.warning("blocked browser navigation",
                        extra={"url": url, "reason": str(exc)})
            return self._failure(url, started, "blocked", str(exc))
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed result
            log.warning("browser navigation failed",
                        extra={"url": url, "error": str(exc)})
            return self._failure(url, started, "failed", str(exc))

        assert self._page is not None  # noqa: S101 - open()/navigate() sets it
        final_url = self._page.url

        # The route guard stops an internal *request*. This is the second check,
        # for the case where a navigation somehow lands elsewhere anyway: if the
        # page we ended up on is not a public one, its body is not read.
        try:
            validate_url(final_url)
        except SSRFError as exc:
            log.warning("browser landed on a blocked host",
                        extra={"url": final_url, "reason": str(exc)})
            return self._failure(url, started, "blocked", str(exc))

        # The real status, not a hardcoded 200. Recording a 404 or a Playwright
        # error page as a successful retrieval poisons the content hash, the
        # fetch cache and `changed_since` with content that was never the page.
        status = getattr(response, "status", None)
        if status is not None and status >= 400:
            return FetchResult(
                url=url, final_url=final_url, status=status, provider=self.name,
                retrieval_status="not_found" if status == 404 else "failed",
                error=f"the page responded {status}",
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        try:
            title = await self._page.title()
            text = (await self._page.inner_text("body")).strip()
            html = await self._page.content()
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed result
            log.warning("browser read failed",
                        extra={"url": final_url, "error": str(exc)})
            return self._failure(url, started, "failed", str(exc))

        if len(text) > self.max_bytes:
            return self._failure(
                url, started, "failed",
                f"rendered page is too large: {len(text)} characters exceeds "
                f"the {self.max_bytes} byte limit",
            )

        return FetchResult(
            url=url, final_url=final_url, status=status, content_type="text/html",
            text=text, html=html, title=title, content_hash=sha256_text(text),
            provider=self.name, content_length=len(text),
            page_type=classify_page(final_url, html, text).value,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    async def extract(self, schema: type[Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "structured extraction is the ExtractionService's job; "
            "browser providers return text only"
        )

    async def close(self) -> None:
        """Tear down everything, in order, tolerating partial failure."""
        for attribute in ("_page", "_context", "_browser"):
            resource = getattr(self, attribute, None)
            if resource is None:
                continue
            try:
                await resource.close()
            except Exception as exc:  # noqa: BLE001 - teardown is best effort
                log.debug("browser teardown failed",
                          extra={"resource": attribute, "error": str(exc)})
            setattr(self, attribute, None)
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as exc:  # noqa: BLE001
                log.debug("playwright stop failed", extra={"error": str(exc)})
            self._playwright = None
