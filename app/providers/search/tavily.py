"""Tavily search.

Discovery only. A :class:`~app.schemas.search.SearchHit` from this provider is a
lead, not a source: the snippet is not retrieved page text, and nothing in this
system may be cited from it. The RAG layer in Task 10 enforces that, and
:class:`app.providers.offline.OfflineSearchProvider` exists so the tests exercise
the same path.

Keeping that distinction here as well is deliberate: the adapter does not fetch
the pages it finds, and it does not pretend its snippet is the page. A provider
that returned a populated ``snippet`` and quietly implied it was quotable would
push the burden of not-citing-it onto every caller.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.schemas.search import SearchHit

log = get_logger(__name__)


class TavilySearchProvider:
    """Web search through the Tavily API."""

    name = "tavily"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: int = 30,
        max_results: int = 20,
        client: Any = None,
    ) -> None:
        if not api_key.strip():
            raise ProviderError(
                "the Tavily provider needs an API key; set TAVILY_API_KEY",
            )
        self._timeout = timeout
        self._max_results = max_results
        self._client = client or self._build_client(api_key)

    @staticmethod
    def _build_client(api_key: str) -> Any:
        """The SDK client, imported here so the dependency is paid only on use."""
        try:
            from tavily import AsyncTavilyClient
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderError(
                "the tavily-python package is required; pip install tavily-python",
            ) from exc
        return AsyncTavilyClient(api_key=api_key)

    async def search(
        self, query: str, *, limit: int = 10, **kwargs: Any
    ) -> list[SearchHit]:
        """Search, returning leads.

        ``include_raw_content`` is deliberately left off. Raw content here is
        Tavily's cached copy of somebody else's page, not a page this run
        retrieved, and a run that started citing it would be citing a cache with
        no retrieval record behind it.
        """
        started = time.monotonic()
        try:
            response = await self._client.search(
                query=query,
                max_results=min(max(1, limit), self._max_results),
                include_raw_content=False,
                **kwargs,
            )
            if hasattr(response, "__await__"):
                response = await response
        except Exception as exc:  # noqa: BLE001 - re-raised as ProviderError below
            raise ProviderError(
                f"the Tavily search failed for {query!r}: {exc}", retryable=True
            ) from exc

        raw = (response or {}).get("results") or []
        retrieved_at = datetime.now(UTC)
        hits: list[SearchHit] = []
        for index, item in enumerate(raw):
            url = item.get("url")
            if not url:
                continue
            hits.append(
                SearchHit(
                    url=url,
                    title=item.get("title"),
                    snippet=item.get("content"),
                    domain=item.get("url") and _domain_of(url),
                    rank=index + 1,
                    provider=self.name,
                    retrieved_at=retrieved_at,
                    raw_score=item.get("score"),
                )
            )
        log.info(
            "tavily search",
            extra={
                "query": query,
                "hits": len(hits),
                "duration_ms": int((time.monotonic() - started) * 1000),
            },
        )
        return hits


def _domain_of(url: str) -> str | None:
    """The host of a URL, for the hit's ``domain``."""
    from urllib.parse import urlparse

    try:
        host = urlparse(url).hostname
    except ValueError:
        return None
    return host
