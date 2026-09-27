"""Web providers, and the re-exports that keep documented paths valid."""

from app.providers.web.firecrawl import FirecrawlWebProvider
from app.providers.web.http import Fetcher, HttpPayload, HttpWebProvider, extract_text

__all__ = [
    "Fetcher",
    "FirecrawlWebProvider",
    "HttpPayload",
    "HttpWebProvider",
    "extract_text",
]
