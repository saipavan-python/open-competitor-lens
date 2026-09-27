"""Application services: the layer between providers and the orchestrator.

Services own decisions that need more than one provider and no knowledge of
LangGraph. Keeping them here is what lets a service be unit-tested with an
offline provider and no graph, and lets the graph stay a thin orchestration of
work it does not itself perform.
"""

from app.services.cache_service import CacheEntry, CacheLookup, ContentCache
from app.services.crawling_service import CrawlingService, CrawlReport, discover_links
from app.services.extraction_service import (
    DedupDecision,
    Deduplicator,
    ExtractionResult,
    ExtractionService,
)

__all__ = [
    "CacheEntry",
    "CacheLookup",
    "ContentCache",
    "CrawlReport",
    "CrawlingService",
    "DedupDecision",
    "Deduplicator",
    "ExtractionResult",
    "ExtractionService",
    "discover_links",
]
