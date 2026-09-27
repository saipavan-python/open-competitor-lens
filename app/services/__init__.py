"""Application services: the layer between providers and the orchestrator.

Services own decisions that need more than one provider and no knowledge of
LangGraph. Keeping them here is what lets a service be unit-tested with an
offline provider and no graph, and lets the graph stay a thin orchestration of
work it does not itself perform.
"""

from app.services.cache_service import CacheEntry, CacheLookup, ContentCache
from app.services.change_detection import ChangeDetector, CompetitorSnapshot
from app.services.citation_service import CitationResolution, CitationService
from app.services.comparison_service import ComparisonService, RankedDimension
from app.services.crawling_service import CrawlingService, CrawlReport, discover_links
from app.services.evidence_service import EvidenceService
from app.services.extraction_service import (
    DedupDecision,
    Deduplicator,
    ExtractionResult,
    ExtractionService,
)
from app.services.model_router import ModelRouter
from app.services.quality_service import MAX_REPORT_REPAIR_ROUNDS, QualityService
from app.services.rag_service import RagService
from app.services.report_service import ReportService
from app.services.research_service import ResearchService
from app.services.search_service import SearchService

__all__ = [
    "MAX_REPORT_REPAIR_ROUNDS",
    "CacheEntry",
    "CacheLookup",
    "ChangeDetector",
    "CitationResolution",
    "CitationService",
    "ComparisonService",
    "CompetitorSnapshot",
    "ContentCache",
    "CrawlReport",
    "CrawlingService",
    "DedupDecision",
    "Deduplicator",
    "EvidenceService",
    "ExtractionResult",
    "ExtractionService",
    "ModelRouter",
    "QualityService",
    "RagService",
    "RankedDimension",
    "ReportService",
    "ResearchService",
    "SearchService",
    "discover_links",
]
