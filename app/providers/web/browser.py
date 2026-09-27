"""Browser escalation.

``should_escalate_to_browser`` is asked once per fetched page whether
server-side retrieval was good enough. Answering yes costs a browser — seconds,
memory, and often a CAPTCHA — so the default is no, and the reasons to say yes
are enumerated rather than heuristic. A page that genuinely needs a browser is
one that returns no text, refuses the request, errors, or is an application
shell that renders client-side.
"""

from __future__ import annotations

from app.schemas.provider import FetchResult
from app.utils.urls import BROWSER_REQUIRED_PATH_HINTS, url_priority
from app.utils.urls import should_escalate_to_browser as _should_escalate

#: Below this many characters of text, a high-priority page is treated as
#: unextracted rather than genuinely empty.
_TINY_BODY = 200

#: Markup fingerprints of a client-rendered shell.
_SPA_MARKERS = (
    'id="root"', "id='root'", 'id="app"', "id='app'", "__NEXT_DATA__",
    "ng-version", "data-reactroot", "id=\"__nuxt\"", "v-cloak",
)


def should_escalate_to_browser(result: FetchResult) -> bool:
    """Whether a browser is needed to read this page.

    Wraps the URL-based rules in :mod:`app.utils.urls` with the facts only the
    fetch result knows: the status code and how much text actually came back.
    A blocked URL is never escalated — it was refused on purpose, and asking
    again by a different route is exactly what the guard exists to prevent.
    """
    if result.retrieval_status == "blocked":
        return False
    return _should_escalate(
        result.final_url or result.url,
        status=result.status,
        text=result.text or "",
        html=result.html or "",
        min_text_chars=_TINY_BODY,
    )


def is_client_rendered_shell(html: str) -> bool:
    """Whether the markup is an application shell with no server-rendered text."""
    lowered = (html or "").lower()
    return any(marker.lower() in lowered for marker in _SPA_MARKERS)


def requires_browser_path(url: str) -> bool:
    """Whether the URL's path implies an interactive application."""
    lowered = (url or "").lower()
    return any(hint in lowered for hint in BROWSER_REQUIRED_PATH_HINTS)


def is_high_value_page(url: str) -> bool:
    """Whether this is one of the page types worth spending a browser on."""
    return url_priority(url) < 200
