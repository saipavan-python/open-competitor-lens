"""HTTP retrieval with per-hop SSRF validation.

Redirects are followed manually rather than by handing ``follow_redirects=True``
to the client. That is the whole point of this module: an automatic redirect
follower validates the URL it was given and then obeys whatever the server says
next, so ``https://evil.example`` can answer ``302`` and land the request on
``169.254.169.254``. Walking the chain here means every hop goes through
:func:`app.utils.urls.validate_url` before a socket is opened.

The transport is injected as a callable so that all of this — the redirect
chain, the byte cap, the blocked addresses, the extraction — is testable without
a network, and so the failure paths can be exercised at all.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

from app.core.exceptions import SSRFError
from app.core.logging import get_logger
from app.schemas.provider import FetchResult
from app.utils.classification import classify_page
from app.utils.hashing import sha256_text
from app.utils.urls import validate_url

log = get_logger(__name__)


@dataclass(frozen=True)
class HttpPayload:
    """One HTTP response, decoupled from the client library.

    Keeping this a plain dataclass rather than an ``httpx.Response`` is what
    lets a test drive the whole provider with a five-line fake.
    """

    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    url: str = ""

    def header(self, name: str) -> str | None:
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return None


#: A transport: given a URL, return one response. Never follows redirects.
Fetcher = Callable[[str], Awaitable[HttpPayload]]

_TEXTUAL_TYPES = ("text/", "application/json", "application/xml", "application/ld+json")
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class HttpWebProvider:
    """Fetches pages over HTTP(S), with the SSRF guard on every hop."""

    name = "http"

    def __init__(
        self,
        *,
        safe_fetcher: Fetcher | None = None,
        timeout: int = 30,
        max_bytes: int = 5_000_000,
        user_agent: str = "competitor-analysis-agent/0.1",
        max_redirects: int = 3,
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.user_agent = user_agent
        self.max_redirects = max_redirects
        self._safe_fetcher = safe_fetcher or httpx_fetcher(timeout, max_bytes, user_agent)

    async def fetch(self, url: str, *, fetcher: Fetcher | None = None) -> FetchResult:
        """Retrieve one page. Never raises for a retrieval failure.

        A page we could not read is a finding the report has to be able to
        state, not a crash that ends the run, so every failure below is
        returned as a ``FetchResult`` carrying a ``retrieval_status``.
        """
        get = fetcher or self._safe_fetcher
        started = time.monotonic()
        chain = [url]
        current = url

        for _ in range(self.max_redirects + 1):
            try:
                validate_url(current)
            except SSRFError as exc:
                log.warning("blocked outbound request",
                            extra={"url": current, "reason": str(exc)})
                return _failure(url, chain, "blocked", str(exc), started)
            try:
                response = await get(current)
            except SSRFError as exc:
                return _failure(url, chain, "blocked", str(exc), started)
            except Exception as exc:  # noqa: BLE001 - surfaced as a failed result
                log.warning("transport error", extra={"url": current, "error": str(exc)})
                return _failure(url, chain, "failed", str(exc), started)

            if len(response.body) > self.max_bytes:
                return _failure(
                    url, chain, "failed",
                    f"response is too large: {len(response.body)} bytes exceeds "
                    f"the {self.max_bytes} byte limit",
                    started,
                )

            if response.status in _REDIRECT_STATUSES:
                location = response.header("location")
                if not location:
                    return _failure(url, chain, "failed",
                                    f"status {response.status} with no Location header",
                                    started)
                current = urljoin(current, location.strip())
                chain.append(current)
                if len(chain) - 1 > self.max_redirects:
                    return _failure(
                        url, chain, "failed",
                        f"redirect chain exceeded {self.max_redirects} hops", started)
                continue

            return _interpret(url, chain, response, started)

        return _failure(url, chain, "failed",
                        f"redirect chain exceeded {self.max_redirects} hops", started)

    async def extract(self, url: str, schema: type[Any]) -> dict[str, Any]:
        """Not implemented here, deliberately.

        Structured extraction needs a model router, which sits above this layer
        in the dependency graph. This provider's contract is text in,
        ``FetchResult`` out; pretending otherwise is how a web adapter ends up
        importing the orchestrator.
        """
        raise NotImplementedError(
            "structured extraction is the ExtractionService's job; "
            "HttpWebProvider returns text only"
        )


def _interpret(url: str, chain: list[str], response: HttpPayload,
               started: float) -> FetchResult:
    """Turn a final (non-redirect) response into a ``FetchResult``."""
    content_type = response.header("content-type") or ""
    final = chain[-1]

    if response.status in (404, 410):
        return _failure(url, chain, "not_found", f"status {response.status}", started,
                        status=response.status, content_type=content_type)
    if response.status >= 400:
        return _failure(url, chain, "failed", f"status {response.status}", started,
                        status=response.status, content_type=content_type)

    if not any(kind in content_type.lower() for kind in _TEXTUAL_TYPES):
        # A PDF or an image is not an error, it is simply not text we can quote.
        # Recorded as a successful fetch with empty text so the report can say
        # the page exists but was not readable as text.
        return _build(url=url, final_url=final, status=response.status,
                      content_type=content_type, text="", html="",
                      content_length=0,
                      content_hash=sha256_text(_decode(response.body, content_type)),
                      page_type=classify_page(final).value,
                      provider="http", started=started)

    raw = _decode(response.body, content_type)
    title, text = extract_text(raw, response.url or final)
    return _build(url=url, final_url=final, status=response.status,
                  content_type=content_type, text=text, html=raw, title=title,
                  content_hash=sha256_text(text), content_length=len(text),
                  page_type=classify_page(final, raw, text).value,
                  provider="http", started=started)


def _decode(body: bytes, content_type: str) -> str:
    return body.decode(_charset_of(content_type), errors="replace")


def _charset_of(content_type: str) -> str:
    for part in content_type.split(";")[1:]:
        if "charset=" in part.lower():
            return part.split("=", 1)[1].strip().strip('"') or "utf-8"
    return "utf-8"


def _build(*, started: float, **kwargs: Any) -> FetchResult:
    return FetchResult(duration_ms=int((time.monotonic() - started) * 1000), **kwargs)


def _failure(url: str, chain: list[str], retrieval_status: str, error: str,
             started: float, **kwargs: Any) -> FetchResult:
    return _build(url=url, final_url=chain[-1] if chain else url, text="", html="",
                  retrieval_status=retrieval_status, error=error, provider="http",
                  started=started, **kwargs)


def extract_text(html: str, url: str = "") -> tuple[str, str]:
    """Return ``(title, readable_text)`` for an HTML document.

    Trafilatura first, because boilerplate stripping is the difference between
    a quotable paragraph and forty thousand characters of navigation. BeautifulSoup
    is the fallback when trafilatura cannot parse, and stripping tags is the last
    resort — messy text the caller can truncate beats no text at all, since no
    text means the claim goes unsourced.
    """
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=True,
            favor_precision=True,
        )
        if extracted and extracted.strip():
            metadata = trafilatura.extract_metadata(html, default_url=url or None)
            return ((metadata.title if metadata else "") or "").strip(), extracted.strip()
    except Exception as exc:  # noqa: BLE001 - a parser failure must not end a fetch
        log.debug("trafilatura extraction failed", extra={"url": url, "error": str(exc)})

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        title = str(soup.title.string).strip() if soup.title and soup.title.string else ""
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
        if text:
            return title, text
    except Exception as exc:  # noqa: BLE001
        log.debug("beautifulsoup extraction failed", extra={"url": url, "error": str(exc)})

    return "", re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def httpx_fetcher(timeout: int, max_bytes: int, user_agent: str) -> Fetcher:
    """The default transport: a real client, streaming with a hard byte cap.

    The cap is enforced while reading, not afterwards. Trusting
    ``Content-Length`` and then reading the body is how a server streams a
    gigabyte past a limit that was nominally in force.
    """

    async def fetch(url: str) -> HttpPayload:
        import httpx

        async with (
            httpx.AsyncClient(
                timeout=timeout,
                # Hops are walked and validated by the caller, not followed
                # blindly: an automatic follower would obey a 302 into 169.254.
                follow_redirects=False,
                headers={"user-agent": user_agent},
            ) as client,
            client.stream("GET", url) as response,
        ):
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise ValueError(
                    f"response is too large: {declared} bytes exceeds "
                    f"the {max_bytes} byte limit"
                )
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(
                        f"response is too large: exceeded the {max_bytes} byte limit"
                    )
                chunks.append(chunk)
            return HttpPayload(
                status=response.status_code,
                headers=dict(response.headers),
                body=b"".join(chunks),
                url=str(response.url),
            )

    return fetch
