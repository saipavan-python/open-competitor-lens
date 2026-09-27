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
from app.utils.urls import validate_url

log = get_logger(__name__)

DEFAULT_USER_AGENT = (
    "competitor-analysis-agent/0.1 (+research; respects robots.txt)"
)


class BrowserUseProvider:
    """Playwright-backed browser following the plan's browser_use integration."""

    name = "browser_use"

    def __init__(self, *, timeout: int = 60, headless: bool = True,
                 user_agent: str | None = None) -> None:
        self.timeout = timeout
        self.headless = headless
        self.user_agent = user_agent or DEFAULT_USER_AGENT
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

    async def open(self, url: str) -> None:
        """Open ``url`` in a fresh context, validating it first.

        A browser can reach internal hosts that a plain fetch would never try —
        redirects, in-page fetches, and internal links all become reachable — so
        the same URL guard is applied here as at the HTTP boundary.
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
        self._context = await self._browser.new_context(user_agent=self.user_agent)
        self._page = await self._context.new_page()
        await self._page.goto(url, timeout=self.timeout * 1000, wait_until="domcontentloaded")

    async def navigate(self, url: str) -> FetchResult:
        started = time.monotonic()
        if self._page is None:
            await self.open(url)
        else:
            validate_url(url)
            await self._page.goto(url, timeout=self.timeout * 1000,
                                  wait_until="domcontentloaded")
        assert self._page is not None  # noqa: S101 - open()/navigate() sets it
        title = await self._page.title()
        text = (await self._page.inner_text("body")).strip()
        html = await self._page.content()
        final_url = self._page.url
        return FetchResult(
            url=url, final_url=final_url, status=200, content_type="text/html",
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
