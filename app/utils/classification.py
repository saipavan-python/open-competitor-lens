"""Page classification.

A pure function of (url, html, text) with no I/O, which is what lets the HTTP
provider stamp a page type onto its result and the crawl service filter on it
without either reaching upward in the dependency graph.

The URL path is the primary signal because it is what a site author chose
deliberately, and it is stable across redesigns. Document structure is the
fallback for pages whose path says nothing — a review page at ``/post/4821`` is
only identifiable from its content. Content-type markup and schema.org hints
are consulted before free text, because they are authored rather than inferred.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from app.core.constants import PageType

#: Path segment to page type, longest and most specific first. Order matters:
#: ``/case-studies/acme`` must classify as CASE_STUDY, not fall through to
#: CUSTOMER because it contains the word "customers" further along the path.
_PATH_RULES: tuple[tuple[str, PageType], ...] = (
    ("/case-studies", PageType.CASE_STUDY),
    ("/case_studies", PageType.CASE_STUDY),
    ("/customer-stories", PageType.CASE_STUDY),
    ("/success-stories", PageType.CASE_STUDY),
    ("/customers", PageType.CUSTOMER),
    ("/customer", PageType.CUSTOMER),
    ("/testimonials", PageType.CUSTOMER),
    ("/pricing", PageType.PRICING),
    ("/plans", PageType.PRICING),
    ("/plans-and-pricing", PageType.PRICING),
    ("/billing", PageType.PRICING),
    ("/features", PageType.FEATURE),
    ("/integrations", PageType.FEATURE),
    ("/solutions", PageType.SOLUTION),
    ("/industries", PageType.INDUSTRY),
    ("/verticals", PageType.INDUSTRY),
    ("/use-cases", PageType.SOLUTION),
    ("/products", PageType.PRODUCT),
    ("/product", PageType.PRODUCT),
    ("/platform", PageType.PRODUCT),
    ("/services", PageType.SERVICE),
    ("/service", PageType.SERVICE),
    ("/about", PageType.ABOUT),
    ("/company", PageType.ABOUT),
    ("/team", PageType.ABOUT),
    ("/careers", PageType.ABOUT),
    ("/faq", PageType.FAQ),
    ("/help", PageType.FAQ),
    ("/support", PageType.FAQ),
    ("/docs", PageType.DOCUMENTATION),
    ("/documentation", PageType.DOCUMENTATION),
    ("/guides", PageType.DOCUMENTATION),
    ("/blog", PageType.BLOG),
    ("/news", PageType.NEWS),
    ("/press", PageType.NEWS),
    ("/changelog", PageType.NEWS),
    ("/reviews", PageType.REVIEW),
)

#: Host-level rules, checked before path rules. A page on g2.com is a review
#: whatever its path says, because that is what the host *is*.
_HOST_RULES: tuple[tuple[str, PageType], ...] = (
    ("github.com", PageType.GITHUB),
    ("gitlab.com", PageType.GITHUB),
    ("docs.github.com", PageType.DOCUMENTATION),
    ("g2.com", PageType.REVIEW),
    ("capterra.com", PageType.REVIEW),
    ("trustradius.com", PageType.REVIEW),
    ("gertel.com", PageType.REVIEW),
    ("producthunt.com", PageType.REVIEW),
    ("reddit.com", PageType.REVIEW),
    ("news.ycombinator.com", PageType.NEWS),
)

#: Ordered content signals: markup and structured data before free text, since
#: they were authored rather than guessed.
_HTML_SIGNALS: tuple[tuple[str, PageType], ...] = (
    (r'itemscope[^>]+itemtype=["\']https?://schema\.org/Review', PageType.REVIEW),
    (r'itemtype=["\']https?://schema\.org/(?:Product|Offer|SoftwareApplication)\b',
     PageType.PRODUCT),
    (r'itemtype=["\']https?://schema\.org/FAQPage', PageType.FAQ),
    (r'itemtype=["\']https?://schema\.org/(?:Article|BlogPosting|NewsArticle)',
     PageType.BLOG),
    (r'itemtype=["\']https?://schema\.org/(?:Organization|Corporation)', PageType.ABOUT),
    (r'class=["\'][^"\']*\b(?:review|testimonial|rating)\b', PageType.REVIEW),
    (r'class=["\'][^"\']*\b(?:pricing|price-table|plans?)\b', PageType.PRICING),
    (r'class=["\'][^"\']*\b(?:docs|documentation)\b', PageType.DOCUMENTATION),
)

#: Free-text signals, used only when nothing else matched. The document title
#: carries more weight than the body, so it is matched first and alone.
_TITLE_SIGNALS: tuple[tuple[re.Pattern[str], PageType], ...] = (
    (re.compile(r"\bpricing\b|\bplans?\s+and\s+pricing\b|\bhow much\b", re.I), PageType.PRICING),
    (re.compile(r"\bcase stud(?:y|ies)\b|\bcustomer stor(?:y|ies)\b", re.I), PageType.CASE_STUDY),
    (re.compile(r"\bproduct\s+overview\b|\bfeatures?\b", re.I), PageType.FEATURE),
    (re.compile(r"\babout\s+us\b|\bour\s+story\b|\bcompany\b", re.I), PageType.ABOUT),
    (re.compile(r"\bfrequently asked\b|\bfaq\b", re.I), PageType.FAQ),
    (re.compile(r"\b(?:reviews?|ratings?|testimonials?)\b", re.I), PageType.REVIEW),
    (re.compile(r"\bblog\b|\barticle\b", re.I), PageType.BLOG),
    (re.compile(r"\bdocumentation\b|\bapi\s+reference\b", re.I), PageType.DOCUMENTATION),
)

_SOCIAL_HOSTS = frozenset({
    "x.com", "twitter.com", "linkedin.com", "facebook.com", "instagram.com",
    "youtube.com", "mastodon.social", "bsky.app", "tiktok.com",
})


def _normalised_path(url: str) -> str:
    path = (urlsplit(url).path or "/").rstrip("/").lower() or "/"
    return path


def _host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    return host


def _title(html: str) -> str:
    if not html:
        return ""
    if match := re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S):
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def classify_page(url: str, html: str = "", text: str = "") -> PageType:
    """Classify a retrieved page into one of the 18 spec page types.

    Falls through progressively: host, then path, then structured markup, then
    the document title, then the body, and finally :attr:`PageType.OTHER`. It
    never raises and never guesses beyond that last step — an unrecognised page
    is reported as OTHER rather than being forced into a category that would
    then be cited as fact.
    """
    host = _host(url)
    # Purpose-defining hosts first: a page on a review platform is a review
    # whatever its path says, because that is what the site *is*.
    for pattern, page_type in _HOST_RULES:
        if host == pattern or host.endswith(f".{pattern}"):
            return page_type

    path = _normalised_path(url)
    if path == "/":
        return PageType.HOMEPAGE
    # Longest matching prefix wins, so /case-studies beats /customers.
    for prefix, page_type in _PATH_RULES:
        if path == prefix or path.startswith(f"{prefix}/"):
            return page_type
    for part in path.strip("/").split("/"):
        if part in {"pricing", "plans"}:
            return PageType.PRICING
        if part in {"case-studies", "case_studies", "customers", "testimonials"}:
            return PageType.CASE_STUDY if "case" in part else PageType.CUSTOMER
        if part in {"blog", "news", "press", "changelog"}:
            return PageType.BLOG if part == "blog" else PageType.NEWS
        if part in {"docs", "documentation", "guides"}:
            return PageType.DOCUMENTATION
        if part in {"reviews", "testimonials"}:
            return PageType.REVIEW
        if part in {"features", "integrations"}:
            return PageType.FEATURE
        if part in {"solutions", "use-cases"}:
            return PageType.SOLUTION
        if part in {"industries", "verticals"}:
            return PageType.INDUSTRY
        if part in {"about", "company", "team"}:
            return PageType.ABOUT
        if part in {"faq", "help", "support"}:
            return PageType.FAQ

    # A social host is a fallback, not an override: x.com/premium is a pricing
    # page, and x.com/someaccount is a social page.
    if host in _SOCIAL_HOSTS:
        return PageType.SOCIAL

    lowered_html = (html or "").lower()
    for pattern, page_type in _HTML_SIGNALS:
        if re.search(pattern, lowered_html, re.IGNORECASE):
            return page_type

    title = _title(html or "")
    for pattern, page_type in _TITLE_SIGNALS:
        if pattern.search(title):
            return page_type
    haystack = title or (text or "")[:2000]
    for pattern, page_type in _TITLE_SIGNALS:
        if pattern.search(haystack):
            return page_type
    return PageType.OTHER
