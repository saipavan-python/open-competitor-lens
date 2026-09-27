"""Firecrawl-backed web retrieval.

A hosted alternative to running our own fetcher. It is a real API, so it is
constructed only when ``FIRECRAWL_API_KEY`` is set, and it is never imported at
module scope by the registry — an unset optional integration must not be able to
break startup.
"""

from __future__ import annotations

import os
import time
from typing import Any

from app.core.exceptions import ProviderError, ProviderNotConfiguredError
from app.core.logging import get_logger
from app.schemas.provider import FetchResult
from app.utils.classification import classify_page
from app.utils.hashing import sha256_text
from app.utils.urls import SSRFError, validate_url

log = get_logger(__name__)


class FirecrawlWebProvider:
    """Retrieves and cleans pages through the Firecrawl API."""

    name = "firecrawl"

    def __init__(self, api_key: str | None = None, *, timeout: int = 60,
                 max_bytes: int = 5_000_000) -> None:
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ProviderNotConfiguredError("firecrawl", "FIRECRAWL_API_KEY")
        self.timeout = timeout
        self.max_bytes = max_bytes

    async def fetch(self, url: str) -> FetchResult:
        started = time.monotonic()
        try:
            validate_url(url)
        except Exception as exc:  # noqa: BLE001 - reported as blocked, not raised
            return FetchResult(url=url, status=None, retrieval_status="blocked",
                               error=str(exc), provider=self.name,
                               duration_ms=int((time.monotonic() - started) * 1000))
        try:
            document = await self._scrape(url)
        except Exception as exc:  # noqa: BLE001 - a failure is data, not a crash
            log.warning("firecrawl request failed", extra={"url": url, "error": str(exc)})
            return FetchResult(url=url, status=None, retrieval_status="failed",
                               error=str(exc), provider=self.name,
                               duration_ms=int((time.monotonic() - started) * 1000))

        text = (document.get("markdown") or document.get("content") or "").strip()
        metadata = document.get("metadata") or {}
        html = document.get("html") or ""
        if len(text) > self.max_bytes:
            text = text[: self.max_bytes]
        # The cap above trimmed ``text`` only. ``html`` is what gets classified
        # and stored, so an oversized document was bounded in one field and not
        # the other, and the limit was not enforced on the wire either.
        html = html[: self.max_bytes]

        # Firecrawl reports the URL it actually fetched in ``sourceURL``. That is
        # attacker-influenced - it is whatever the scraped page resolved to - and
        # it is recorded as ``final_url`` and then used to classify the page, so
        # it has to clear the same guard as the URL we asked for. A page that
        # redirects to a metadata endpoint gets a 200 from Firecrawl's network
        # and would otherwise be recorded as a successful retrieval of it.
        final_url = url
        reported = metadata.get("sourceURL")
        if reported and reported != url:
            try:
                validate_url(reported)
            except SSRFError as exc:
                log.warning(
                    "blocked a scraped page that resolved to an internal host",
                    extra={"url": url, "reported_url": reported, "reason": str(exc)},
                )
                return FetchResult(
                    url=url, final_url=None, status=None, provider=self.name,
                    retrieval_status="blocked", error=str(exc),
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            final_url = reported

        return FetchResult(
            url=url,
            final_url=final_url,
            status=200,
            content_type="text/markdown",
            text=text,
            html=html,
            title=metadata.get("title"),
            content_hash=sha256_text(text),
            provider=self.name,
            page_type=classify_page(final_url, html, text).value,
            content_length=len(text),
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    async def _scrape(self, url: str) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"url": url, "formats": ["markdown", "html"]},
            )
            if response.status_code >= 400:
                raise ProviderError(
                    f"firecrawl returned {response.status_code}: {response.text[:200]}",
                    provider=self.name, retryable=response.status_code >= 500,
                )
            return response.json().get("data") or {}

    async def extract(self, url: str, schema: type[Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "structured extraction is the ExtractionService's job; "
            "FirecrawlWebProvider returns text only"
        )
