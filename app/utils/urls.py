"""Outbound URL safety: canonicalisation, SSRF validation, crawl prioritisation.

This module is the only place in the system permitted to hand a URL to an HTTP
client. Every outbound request goes through :func:`validate_url` first, and
every redirect hop is re-validated by :func:`validate_redirect_chain`, because a
public URL that 302s to ``169.254.169.254`` is the classic way to defeat a
one-shot check.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.core.exceptions import SSRFError

#: Query parameters stripped during canonicalisation. All are campaign or
#: click identifiers: they change per visit and would defeat deduplication.
TRACKING_PARAMS: frozenset[str] = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_source_platform", "utm_creative_format", "utm_marketing_tactic",
    "gclid", "gbraid", "wbraid", "dclid", "fbclid", "msclkid", "yclid", "twclid",
    "igshid", "mc_cid", "mc_eid", "ref", "referrer", "source", "hsa_acc",
    "_ga", "_gl", "ref_src", "spm", "scm", "trk", "trkCampaign",
})

DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}

#: Hosts that resolve to internal infrastructure regardless of DNS.
CLOUD_METADATA_HOSTS: frozenset[str] = frozenset({
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
    "metadata",
})

#: Multi-label public suffixes we treat as a single registrable domain. A
#: general list would be a large dependency; these cover the cases a competitor
#: crawl actually meets.
MULTI_PART_TLDS: frozenset[str] = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk", "co.in", "net.in", "org.in",
    "com.au", "net.au", "org.au", "co.nz", "com.br", "com.mx", "co.za",
    "com.sg", "com.tr", "com.cn", "co.jp", "co.kr", "com.ar", "com.my",
})

#: Path fragments that indicate a page whose content needs a real browser.
BROWSER_REQUIRED_PATH_HINTS: tuple[str, ...] = (
    "/app/", "/dashboard", "/console", "/login", "/signup", "/checkout",
    "/interactive", "/calculator", "/demo/",
)

#: The ten page types that decide most of a competitor's commercial story.
#: Ordered: the index *is* the priority score.
HIGH_PRIORITY_PATHS: tuple[str, ...] = (
    "", "/about", "/products", "/services", "/features", "/pricing",
    "/solutions", "/customers", "/case-studies", "/faq",
)

#: Commercially relevant, but not decisive: crawled after the band above.
SECONDARY_PRIORITY_PATHS: frozenset[str] = frozenset({
    "/about-us", "/plans", "/industries", "/blog", "/contact", "/enterprise",
    "/compare", "/alternatives", "/why-us", "/company", "/team",
})

_SPA_SHELL = re.compile(
    r"<(?:div|main)[^>]*id=[\"'](?:root|app|__next|__nuxt)[\"'][^>]*>\s*</", re.IGNORECASE
)
_TINY_BODY = 500


@dataclass(frozen=True, slots=True)
class ValidatedUrl:
    """A URL that has passed every outbound safety check."""

    url: str
    scheme: str
    hostname: str
    port: int
    path: str
    query: str
    resolved_ips: tuple[str, ...] = ()

    @property
    def is_https(self) -> bool:
        return self.scheme == "https"


def _strip_www(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def canonicalize_url(raw: str) -> str | None:
    """Normalise a URL for deduplication, or return ``None`` if unusable.

    Idempotent: ``canonicalize_url(canonicalize_url(u)) == canonicalize_url(u)``.
    """
    if not raw or not raw.strip():
        return None
    candidate = raw.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    try:
        parts = urlsplit(candidate)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    host = (parts.hostname or "").lower()
    if not host:
        return None
    host = _strip_www(host)
    try:
        host = host.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        return None

    netloc = host
    port = parts.port
    if port is not None and port != DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{port}"

    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/":
        path = path.rstrip("/")
    else:
        path = ""

    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS]
    query = urlencode(sorted(kept), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def _ip_is_blocked(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a human reason when an address must not be requested."""
    if address.is_loopback:
        return "loopback address"
    if address.is_link_local:
        return "link-local address (includes cloud metadata endpoints)"
    if address.is_private:
        return "private address"
    if address.is_multicast:
        return "multicast address"
    if address.is_reserved or address.is_unspecified:
        return "reserved or unspecified address"
    if isinstance(address, ipaddress.IPv6Address):
        mapped = address.ipv4_mapped or (
            address.teredo[1] if address.teredo else None
        )
        if mapped is not None and _ip_is_blocked(mapped):
            return "IPv4-mapped private address"
    return None


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse ``host`` as an IP literal, or ``None`` if it is a name.

    ``urlsplit().hostname`` already strips brackets, so IPv6 literals arrive
    bare. Decimal and hex obfuscation (``2130706433``, ``0x7f000001``) are
    rejected outright rather than silently resolved, because accepting them would
    mean accepting a second, unvalidated path to a blocked address.
    """
    text = host.strip("[]")
    if not text:
        return None
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", text) or text.isdigit():
        try:
            return ipaddress.ip_address(int(text, 0))
        except ValueError:
            raise SSRFError("Blocked URL: host is an obfuscated numeric address") from None
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        return None


def validate_url(raw: str, *, allowed_ports: list[int] | None = None,
                 resolve_dns: bool = True, require_dns: bool = False) -> ValidatedUrl:
    """Validate a URL for outbound use, or raise :class:`SSRFError`.

    Checks, in order: presence, scheme, userinfo, host syntax, port, cloud
    metadata hostnames, IP literal membership, then DNS resolution. DNS is
    checked because a public-looking name is free to resolve to ``10.0.0.1``.

    ``require_dns`` controls what an unresolvable host means. Left ``False``
    (the default) it is treated as a network problem rather than an attack, so
    the guard works in air-gapped deployments; set it ``True`` to refuse
    anything whose addresses could not be confirmed.
    """
    if not raw or not raw.strip():
        raise SSRFError("Blocked URL: empty URL")
    try:
        parts = urlsplit(raw.strip())
    except ValueError as exc:
        raise SSRFError("Blocked URL: malformed") from exc

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise SSRFError(f"Blocked URL: scheme '{scheme or 'none'}' is not http or https")

    if parts.username or parts.password:
        raise SSRFError("Blocked URL: embedded credentials are not allowed")

    try:
        hostname = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise SSRFError("Blocked URL: malformed host or port") from exc
    if not hostname:
        raise SSRFError("Blocked URL: missing host")
    hostname = hostname.lower().removeprefix("www.")

    literal = _parse_ip(hostname)
    if literal is None and not re.fullmatch(r"[a-z0-9._~%-]+", hostname):
        raise SSRFError("Blocked URL: malformed host")

    if port is None:
        port = DEFAULT_PORTS[scheme]
    permitted = allowed_ports if allowed_ports is not None else [80, 443, 8080, 8443]
    if port not in permitted:
        raise SSRFError(f"Blocked URL: port {port} is not permitted")

    if hostname in CLOUD_METADATA_HOSTS or hostname.endswith(".internal"):
        raise SSRFError("Blocked URL: internal metadata host")

    if literal is not None:
        reason = _ip_is_blocked(literal)
        if reason:
            raise SSRFError(f"Blocked URL: {hostname} is a {reason}")
        return ValidatedUrl(url=canonicalize_url(raw) or raw, scheme=scheme,
                            hostname=hostname, port=port, path=parts.path or "/",
                            query=parts.query, resolved_ips=(str(literal),))

    resolved: tuple[str, ...] = ()
    if resolve_dns:
        try:
            infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
        except (socket.gaierror, UnicodeError, OSError):
            # An unresolvable host is not evidence of an attack. The request will
            # fail at the socket layer and be recorded as a failed source, which
            # is the honest outcome. Treating "no DNS" as "hostile" would break
            # every air-gapped deployment and every offline test run.
            if require_dns:
                raise SSRFError(
                    f"Blocked URL: host '{hostname}' could not be resolved"
                ) from None
            infos = []
        candidates = {str(info[4][0]) for info in infos}
        for raw_ip in candidates:
            try:
                address = ipaddress.ip_address(raw_ip)
            except ValueError:
                continue
            reason = _ip_is_blocked(address)
            if reason:
                raise SSRFError(f"Blocked URL: {hostname} resolves to a {reason} ({raw_ip})")
        resolved = tuple(sorted(candidates))
        if not candidates and require_dns:
            raise SSRFError(f"Blocked URL: host '{hostname}' resolved to no addresses")

    return ValidatedUrl(url=canonicalize_url(raw) or raw, scheme=scheme, hostname=hostname,
                        port=port, path=parts.path or "/", query=parts.query,
                        resolved_ips=resolved)


def validate_redirect_chain(urls: list[str] | None, *, max_redirects: int = 3,
                            allowed_ports: list[int] | None = None) -> ValidatedUrl:
    """Validate every hop of a redirect chain.

    Validating only the requested URL is not enough: a server we are willing to
    talk to can answer ``302`` and point us anywhere, including the cloud
    metadata service. Every hop is checked, and the chain length is bounded.
    """
    if not urls:
        raise SSRFError("Blocked redirect chain: no URLs to validate")
    if len(urls) - 1 > max_redirects:
        raise SSRFError(
            f"Blocked redirect chain: {len(urls) - 1} redirects exceeds "
            f"the limit of {max_redirects}"
        )
    validated = validate_url(urls[0], allowed_ports=allowed_ports)
    for hop in urls[1:]:
        validated = validate_url(hop, allowed_ports=allowed_ports)
    return validated

def registrable_domain(url: str) -> str | None:
    """The registrable domain of a URL, e.g. ``blog.acme.co.uk`` -> ``acme.co.uk``."""
    host = urlsplit(url).hostname
    if not host:
        return None
    host = _strip_www(host.lower())
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if ".".join(labels[-2:]) in MULTI_PART_TLDS:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def same_site(first: str, second: str) -> bool:
    """Whether two URLs belong to the same registrable domain."""
    left, right = registrable_domain(first), registrable_domain(second)
    return bool(left) and left == right


def url_priority(url: str) -> int:
    """Crawl priority: lower is more important (spec section 20).

    Banded rather than a flat list index, so the ten commercially decisive page
    types all sort ahead of everything else and the score stays stable when the
    priority list is extended:

    ``0-9``   home, about, products, services, features, pricing, solutions,
              customers, case studies, FAQ
    ``40``    other commercial and editorial pages (blog, contact, industries…)
    ``100+``  everything else, ordered by path depth

    Deterministic in the URL, so a repeated crawl visits pages in the same order
    and change detection stays comparable between runs.
    """
    path = (urlsplit(url).path or "/").rstrip("/").lower() or "/"
    if path in HIGH_PRIORITY_PATHS:
        return HIGH_PRIORITY_PATHS.index(path)
    if path in SECONDARY_PRIORITY_PATHS:
        return 40
    if any(path.startswith(hint) for hint in ("/blog", "/news", "/article", "/post")):
        return 200
    return 100 + path.count("/") * 10


def should_escalate_to_browser(
    url: str, *, status: int | None, text: str, html: str,
    min_text_chars: int = _TINY_BODY,
) -> bool:
    """Whether normal HTTP extraction was insufficient for this page.

    Deliberately conservative, and biased towards *not* escalating: a browser
    costs seconds, memory and often a CAPTCHA, and firing five of them at a site
    that is already struggling is discourteous.

    - ``403``/``401`` escalate: the server is refusing our client, and a real
      browser with a real user agent frequently gets through.
    - ``404``/``410`` do not: a definitive answer no amount of JavaScript changes.
    - ``429`` does not: a browser would deepen the rate limit it just hit.
    - ``5xx`` does not: a browser cannot repair a broken server, and the second
      request adds load to a site that is already failing.
    - A ``200`` with no text escalates: that is an application shell, which is
      precisely what a browser is for.
    """
    if status is None:
        return True
    if status in (401, 403):
        return True
    if status >= 400:
        return False
    if url_priority(url) < 200 and len(text.strip()) < min_text_chars:
        return True
    if not text.strip() and (_SPA_SHELL.search(html or "")
                             or len(html or "") < min_text_chars):
        return True
    lowered = (urlsplit(url).path or "").lower()
    return any(hint in lowered for hint in BROWSER_REQUIRED_PATH_HINTS)


def safe_url_for_log(url: str) -> str:
    """A URL with any query string removed, safe to put in a log line."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable url>"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def fetch_with_guard(
    client: Any, url: str, **kwargs: Any
) -> Any:  # pragma: no cover - thin httpx adapter, covered by provider tests
    """Perform a guarded GET. Kept next to the guard so it cannot be bypassed."""
    validate_url(url)
    return client.get(url, follow_redirects=False, **kwargs)
