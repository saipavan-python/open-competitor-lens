"""Domain-wide constants and enumerations shared by schemas, agents and nodes."""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    """Lifecycle of a research run (spec section 51)."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressStage(StrEnum):
    """User-visible pipeline stages (spec section 51)."""

    INITIALIZING = "initializing"
    COMPANY_DISCOVERY = "company_discovery"
    COMPETITOR_DISCOVERY = "competitor_discovery"
    PLANNING = "planning"
    RESEARCHING = "researching"
    INDEXING = "indexing"
    ANALYZING = "analyzing"
    FACT_CHECKING = "fact_checking"
    REPORTING = "reporting"
    COMPLETED = "completed"


#: Ordered stage -> cumulative progress percentage. The status endpoint reports
#: the percentage of the last stage entered, so a run never reads 0% or 100%
#: while still working.
STAGE_PROGRESS: dict[ProgressStage, int] = {
    ProgressStage.INITIALIZING: 2,
    ProgressStage.COMPANY_DISCOVERY: 12,
    ProgressStage.COMPETITOR_DISCOVERY: 22,
    ProgressStage.PLANNING: 28,
    ProgressStage.RESEARCHING: 62,
    ProgressStage.INDEXING: 70,
    ProgressStage.ANALYZING: 82,
    ProgressStage.FACT_CHECKING: 90,
    ProgressStage.REPORTING: 97,
    ProgressStage.COMPLETED: 100,
}


class ClaimType(StrEnum):
    """How a claim came to be known (spec section 15)."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    REPORTED = "reported"
    UNKNOWN = "unknown"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceType(StrEnum):
    """Source authority tiers, most authoritative first (spec section 13)."""

    OFFICIAL_SITE = "official_site"
    OFFICIAL_DOCS = "official_docs"
    OFFICIAL_PROFILE = "official_profile"
    PRIMARY_PUBLIC = "primary_public"
    INDEPENDENT = "independent"
    REVIEW_PLATFORM = "review_platform"
    NEWS = "news"
    DIRECTORY = "directory"
    SEARCH_SNIPPET = "search_snippet"
    UNKNOWN = "unknown"


#: Numeric authority weight, used to break ties between conflicting sources.
#: A snippet is a discovery aid and can never by itself raise a claim to ``high``.
SOURCE_AUTHORITY: dict[SourceType, int] = {
    SourceType.OFFICIAL_SITE: 100,
    SourceType.OFFICIAL_DOCS: 95,
    SourceType.OFFICIAL_PROFILE: 90,
    SourceType.PRIMARY_PUBLIC: 85,
    SourceType.INDEPENDENT: 70,
    SourceType.REVIEW_PLATFORM: 60,
    SourceType.NEWS: 50,
    SourceType.DIRECTORY: 40,
    SourceType.SEARCH_SNIPPET: 20,
    SourceType.UNKNOWN: 10,
}

#: Sources that can never support a ``high`` confidence claim on their own.
NON_PRIMARY_SOURCE_TYPES = frozenset({SourceType.SEARCH_SNIPPET, SourceType.UNKNOWN})


class PageType(StrEnum):
    """Classified page types (spec section 90)."""

    HOMEPAGE = "homepage"
    ABOUT = "about"
    PRODUCT = "product"
    SERVICE = "service"
    PRICING = "pricing"
    FEATURE = "feature"
    SOLUTION = "solution"
    INDUSTRY = "industry"
    CUSTOMER = "customer"
    CASE_STUDY = "case-study"
    FAQ = "faq"
    BLOG = "blog"
    DOCUMENTATION = "documentation"
    REVIEW = "review"
    SOCIAL = "social"
    NEWS = "news"
    GITHUB = "github"
    OTHER = "other"


class AnalysisCategory(StrEnum):
    """The specialized analyst categories, matching report sections."""

    WEBSITE = "website"
    PRODUCT = "product"
    PRICING = "pricing"
    MARKETING = "marketing"
    SEO = "seo"
    SOCIAL = "social"
    REVIEW = "review"
    TECHNOLOGY = "technology"
    MARKET = "market"


class InsightType(StrEnum):
    """Structured insight categories (spec section 38)."""

    PRODUCT = "product"
    PRICING = "pricing"
    MARKETING = "marketing"
    SEO = "seo"
    SOCIAL = "social"
    REVIEW = "review"
    TECHNOLOGY = "technology"
    CUSTOMER = "customer"
    MARKET = "market"
    POSITIONING = "positioning"


class ClaimVerdict(StrEnum):
    """Outcome of validating one claim against the evidence index (spec section 39)."""

    SUPPORTED = "supported"
    OVERSTATEMENT = "overstatement"
    UNSUPPORTED = "unsupported"
    CONTRADICTORY = "contradictory"
    STALE = "stale"


class RetrievalStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    DUPLICATE = "duplicate"
    BROWSER = "browser"


class EventType(StrEnum):
    """Internal domain events (spec section 102)."""

    RESEARCH_STARTED = "research_started"
    COMPANY_DISCOVERED = "company_discovered"
    COMPETITORS_FOUND = "competitors_found"
    SOURCE_FETCHED = "source_fetched"
    SOURCE_FAILED = "source_failed"
    EVIDENCE_EXTRACTED = "evidence_extracted"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    FACT_CHECK_FAILED = "fact_check_failed"
    REPORT_GENERATED = "report_generated"
    RESEARCH_COMPLETED = "research_completed"


class ChangeType(StrEnum):
    """Detected competitor changes (spec section 61)."""

    PRICING = "pricing"
    PRODUCT = "product"
    WEBSITE = "website"
    MARKETING = "marketing"
    REVIEW = "review"
    TECHNOLOGY = "technology"


#: Literal strings used whenever information could not be established. Having a
#: single vocabulary for "we looked and found nothing" keeps the report honest
#: and stops an empty field from being rendered as a fabricated value.
UNKNOWN_SENTINELS = (
    "Not publicly available",
    "Not found during research",
    "Unable to verify",
    "Conflicting public information",
)

#: Pages crawled first, in priority order, before the rest of a site (spec 20).
PAGE_PRIORITY_PATHS: tuple[str, ...] = (
    "/", "/about", "/about-us", "/products", "/services", "/features",
    "/pricing", "/plans", "/solutions", "/industries", "/customers",
    "/case-studies", "/case-studies/", "/customers/", "/faq", "/blog", "/contact",
)

#: Public social platforms the social analyst looks for (spec section 32).
SOCIAL_PLATFORMS: dict[str, str] = {
    "linkedin": "linkedin.com",
    "youtube": "youtube.com",
    "instagram": "instagram.com",
    "facebook": "facebook.com",
    "x": "x.com",
    "tiktok": "tiktok.com",
    "github": "github.com",
}
