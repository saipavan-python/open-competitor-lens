"""The unavailable-browser stand-in.

Kept apart from :mod:`app.providers.browser.browser_use` so that the "no browser
in this deployment" case can be imported, and tested, without pulling in
Playwright at all.
"""

from __future__ import annotations

import os
from typing import Any

from app.core.exceptions import ProviderError
from app.schemas.provider import FetchResult


class BrowserUnavailableError(ProviderError):
    """No browser is available in this deployment."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            f"No browser is available: {detail}. Either install Playwright "
            f"(playwright install chromium) or set ENABLE_BROWSER_USE=false.",
            retryable=False,
            provider="browser",
        )
        self.detail = detail


class UnavailableBrowserProvider:
    """Stands in when no browser exists, so escalation can fail informatively.

    A deployment without a browser must still be able to produce a report that
    says "this page could not be rendered", rather than one that silently treats
    the unrendered page as empty.
    """

    name = "unavailable"

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason or "browser support is disabled (ENABLE_BROWSER_USE=false)"

    def available(self) -> bool:
        return False

    async def open(self, url: str) -> None:
        raise BrowserUnavailableError(self.reason)

    async def navigate(self, url: str) -> FetchResult:
        raise BrowserUnavailableError(self.reason)

    async def extract(self, schema: type[Any]) -> dict[str, Any]:
        raise BrowserUnavailableError(self.reason)

    async def close(self) -> None:
        return None


def browser_from_settings(*, enabled: bool | None = None, timeout: int = 60
                          ) -> UnavailableBrowserProvider | Any:
    """Pick a browser implementation for this deployment.

    Imports the real implementation lazily, so a deployment that will never open
    a browser does not need Playwright importable.
    """
    if enabled is None:
        enabled = os.environ.get("ENABLE_BROWSER_USE", "true").lower() in ("1", "true", "yes")
    if not enabled:
        return UnavailableBrowserProvider()
    try:
        import playwright  # noqa: F401
    except ImportError:
        return UnavailableBrowserProvider("playwright is not installed")
    from app.providers.browser.browser_use import BrowserUseProvider

    return BrowserUseProvider(timeout=timeout)
