"""RED: URL canonicalisation, SSRF validation, and the utility layer.

The SSRF guard is the single control between user-supplied URLs and the
outbound request path, so it is tested directly against every address class the
platform must refuse.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Iterable

import pytest

from app.core.exceptions import SSRFError
from app.utils.concurrency import gather_limited
from app.utils.dates import parse_date, utcnow
from app.utils.hashing import composite_hash, page_hash, sha256_text, stable_id
from app.utils.json import safe_json_loads
from app.utils.normalization import (
    company_identity_key,
    normalize_company_name,
    normalize_space,
    slugify,
)
from app.utils.retry import RetryExhaustedError, with_retry
from app.utils.text import chunk_text, collapse_whitespace, truncate, word_count
from app.utils.urls import (
    canonicalize_url,
    registrable_domain,
    same_site,
    should_escalate_to_browser,
    url_priority,
    validate_redirect_chain,
    validate_url,
)

BLOCKED_URLS = [
    "http://127.0.0.1/x",
    "http://127.0.0.1:8080/x",
    "https://[::1]/x",
    "http://[fd00::1]/x",
    "http://10.0.0.1/x",
    "http://10.255.255.254/x",
    "http://172.16.0.1/x",
    "http://172.31.255.255/x",
    "http://192.168.1.1/x",
    "http://169.254.169.254/latest/meta-data/",
    "http://169.254.1.1/x",
    # RFC 6598 shared address space. ``is_private`` does not cover it, so the
    # guard used to allow it - and it is precisely the range that reaches
    # internal service and proxy endpoints in cloud and container networks.
    "http://100.64.0.1/x",
    "http://100.127.255.254/x",
    "http://0.0.0.0/x",
    "http://255.255.255.255/x",
    "http://localhost/x",
    "http://localhost:3000/x",
    "http://metadata.google.internal/computeMetadata/v1/",
    "ftp://example.com/x",
    "file:///etc/passwd",
    "javascript:alert(1)",
    "http://user:password@example.com/x",
    "http://example.com:22/x",
    "http://example.com:3306/x",
    "http://[::ffff:127.0.0.1]/x",
    "http://0x7f000001/x",
    "http://2130706433/x",
    "",
    "not a url",
    "http://",
]


class TestCanonicalizeUrl:
    def test_lowercases_scheme_and_host_but_not_path(self) -> None:
        assert canonicalize_url("HTTPS://Example.COM/Path") == "https://example.com/Path"

    def test_strips_www(self) -> None:
        assert canonicalize_url("https://www.acme.com/a") == "https://acme.com/a"

    def test_drops_fragment(self) -> None:
        assert canonicalize_url("https://acme.com/a#section") == "https://acme.com/a"

    def test_drops_tracking_params(self) -> None:
        result = canonicalize_url("https://acme.com/a?utm_source=x&utm_medium=y&id=3&fbclid=z")
        assert result == "https://acme.com/a?id=3"

    def test_sorts_remaining_query_params(self) -> None:
        assert canonicalize_url("https://acme.com/a?b=2&a=1") == "https://acme.com/a?a=1&b=2"

    def test_drops_default_port(self) -> None:
        assert canonicalize_url("https://acme.com:443/a") == "https://acme.com/a"
        assert canonicalize_url("http://acme.com:80/a") == "http://acme.com/a"

    def test_keeps_non_default_port(self) -> None:
        assert canonicalize_url("https://acme.com:8443/a") == "https://acme.com:8443/a"

    def test_strips_trailing_slash_on_root(self) -> None:
        assert canonicalize_url("https://acme.com/") == "https://acme.com"

    def test_adds_scheme_when_missing(self) -> None:
        assert canonicalize_url("acme.com/pricing") == "https://acme.com/pricing"

    def test_returns_none_for_unparseable(self) -> None:
        assert canonicalize_url("http://") is None
        assert canonicalize_url("") is None

    def test_idempotent(self) -> None:
        once = canonicalize_url("HTTPS://WWW.Acme.com:443/a/?b=2&utm_source=q#f")
        assert canonicalize_url(once) == once

    def test_distinguishes_different_paths(self) -> None:
        assert canonicalize_url("https://acme.com/a") != canonicalize_url("https://acme.com/b")


class TestValidateUrl:
    @pytest.mark.parametrize("url", BLOCKED_URLS)
    def test_rejects_dangerous_urls(self, url: str) -> None:
        with pytest.raises(SSRFError):
            validate_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://acme.com",
            "https://www.acme.com/pricing",
            "https://sub.domain.acme.co.uk/a?b=1",
            "https://acme.com:8443/a",
            "https://xn--80ak6aa92e.com/",
            "https://8.8.8.8/",
            "https://[2606:4700:4700::1111]/",
        ],
    )
    def test_accepts_public_urls(self, url: str) -> None:
        assert validate_url(url).hostname

    def test_returns_normalised_url(self) -> None:
        result = validate_url("HTTPS://WWW.Acme.com/pricing?utm_source=x")
        assert result.url == "https://acme.com/pricing"
        assert result.hostname == "acme.com"
        assert result.scheme == "https"
        assert result.port == 443

    def test_error_message_names_the_reason(self) -> None:
        with pytest.raises(SSRFError, match="loopback"):
            validate_url("http://127.0.0.1/x")

    def test_rejects_dns_name_resolving_to_private_address(self, monkeypatch) -> None:
        """A public-looking hostname that resolves inward is still SSRF."""

        def fake_getaddrinfo(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 80))]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        with pytest.raises(SSRFError):
            validate_url("https://sneaky.example.com/")

    def test_rejects_malformed_host(self) -> None:
        with pytest.raises(SSRFError):
            validate_url("https://exa mple.com/")


class TestValidateRedirectChain:
    def test_accepts_a_clean_chain(self) -> None:
        validate_redirect_chain(["https://a.com/1", "https://a.com/2", "https://a.com/3"])

    def test_rejects_a_chain_that_redirects_to_metadata(self) -> None:
        """Review Focus: validation must be re-applied to every hop."""
        with pytest.raises(SSRFError):
            validate_redirect_chain(["https://a.com/1", "http://169.254.169.254/latest/"])

    def test_rejects_a_chain_that_redirects_to_loopback(self) -> None:
        with pytest.raises(SSRFError):
            validate_redirect_chain(["https://a.com/1", "http://127.0.0.1/admin"])

    def test_rejects_when_first_hop_is_already_bad(self) -> None:
        with pytest.raises(SSRFError):
            validate_redirect_chain(["http://192.168.0.1/"])

    def test_rejects_an_over_long_chain(self) -> None:
        with pytest.raises(SSRFError):
            validate_redirect_chain([f"https://a.com/{i}" for i in range(10)], max_redirects=3)

    def test_empty_chain_is_rejected(self) -> None:
        with pytest.raises(SSRFError):
            validate_redirect_chain([])


class TestDomainHelpers:
    def test_registrable_domain_strips_subdomain(self) -> None:
        assert registrable_domain("https://blog.acme.com/a") == "acme.com"

    def test_registrable_domain_of_bare_domain(self) -> None:
        assert registrable_domain("https://acme.com") == "acme.com"

    def test_registrable_domain_handles_multi_part_tld(self) -> None:
        assert registrable_domain("https://a.acme.co.uk/x") == "acme.co.uk"

    def test_same_site_is_true_for_subdomains(self) -> None:
        assert same_site("https://acme.com/a", "https://blog.acme.com/b") is True

    def test_same_site_is_false_across_domains(self) -> None:
        assert same_site("https://acme.com/a", "https://beta.com/b") is False

    def test_same_site_tolerates_a_bad_url(self) -> None:
        assert same_site("https://acme.com", "not a url") is False


class TestCrawlPriority:
    def test_homepage_outranks_a_blog_post(self) -> None:
        assert url_priority("https://acme.com/") < url_priority("https://acme.com/blog/post-1")

    def test_pricing_outranks_blog(self) -> None:
        assert url_priority("https://acme.com/pricing") < url_priority("https://acme.com/blog/x")

    def test_about_and_products_are_high_priority(self) -> None:
        for path in ("/about", "/products", "/services", "/customers"):
            assert url_priority(f"https://acme.com{path}") < 20

    def test_deep_paths_score_worse(self) -> None:
        assert url_priority("https://acme.com/a/b/c/d/e") > url_priority("https://acme.com/a")


class TestBrowserEscalation:
    def test_escalates_on_empty_body(self) -> None:
        assert should_escalate_to_browser(
            "https://acme.com/x", status=200, text="", html="<html><body><div id=root></div></body></html>"
        ) is True

    def test_escalates_on_non_2xx(self) -> None:
        assert should_escalate_to_browser("https://acme.com/x", status=403, text="Forbidden",
                                           html="<html></html>") is True

    def test_escalates_on_spa_shell(self) -> None:
        assert should_escalate_to_browser(
            "https://acme.com/x", status=200, text="", html="<div id=app></div>"
        ) is True

    def test_does_not_escalate_for_rich_static_content(self) -> None:
        body = "<html><body>" + ("real content " * 500) + "</body></html>"
        assert should_escalate_to_browser("https://acme.com/x", status=200,
                                           text="real content " * 500, html=body) is False

    def test_does_not_escalate_for_a_plain_404(self) -> None:
        assert should_escalate_to_browser("https://acme.com/x", status=404, text="Not found",
                                           html="<html>Not found</html>") is False

    def test_known_js_heavy_path_escalates_even_with_content(self) -> None:
        assert should_escalate_to_browser("https://acme.com/app/dashboard", status=200,
                                           text="Loading", html="<html>Loading</html>") is True


class TestHashing:
    def test_composite_hash_separator_prevents_collision(self) -> None:
        assert composite_hash("ab", "c") != composite_hash("a", "bc")

    def test_page_hash_includes_the_url(self) -> None:
        """Review Focus: identical boilerplate on two sites stays two sources."""
        assert page_hash("https://a.com", "same footer") != page_hash("https://b.com", "same footer")

    def test_page_hash_is_stable(self) -> None:
        assert page_hash("https://a.com", "x") == page_hash("https://a.com", "x")

    def test_stable_id_is_deterministic(self) -> None:
        assert stable_id("ev", "acme", "pricing") == stable_id("ev", "acme", "pricing")

    def test_stable_id_differs_on_input(self) -> None:
        assert stable_id("ev", "acme") != stable_id("ev", "beta")

    def test_stable_id_has_the_prefix(self) -> None:
        assert stable_id("src", "x").startswith("src-")

    def test_sha256_text_matches_known_value(self) -> None:
        assert sha256_text("") == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )


class TestNormalization:
    def test_normalize_company_name_strips_legal_suffix(self) -> None:
        assert normalize_company_name("Acme Private Limited") == "acme"

    def test_normalize_company_name_strips_punctuation(self) -> None:
        assert normalize_company_name("Acme, Inc.") == "acme"

    def test_normalize_company_name_folds_case(self) -> None:
        assert normalize_company_name("ACME") == "acme"

    def test_identity_key_merges_aliases(self) -> None:
        """Review Focus: 'Kansbe' and 'Kansbe Digital' must not collapse blindly."""
        assert company_identity_key("Kansbe") == company_identity_key("Kansbe Private Limited")
        assert company_identity_key("Kansbe") == company_identity_key("kansbe.com")

    def test_identity_key_separates_genuinely_different_names(self) -> None:
        assert company_identity_key("Kansbe") != company_identity_key("Kansbe Digital")

    def test_normalize_space_collapses_runs(self) -> None:
        assert normalize_space("  a \n b\t c  ") == "a b c"

    def test_slugify(self) -> None:
        assert slugify("Acme Digital Services!") == "acme-digital-services"


class TestText:
    def test_chunk_respects_max_size(self) -> None:
        text = "word " * 2000
        chunks = chunk_text(text, max_chars=500)
        assert all(len(chunk) <= 700 for chunk in chunks)
        assert len(chunks) > 1

    def test_chunk_never_splits_a_paragraph(self) -> None:
        paragraphs = [f"Paragraph {i} " + "A" * 100 for i in range(10)]
        chunks = chunk_text("\n\n".join(paragraphs), max_chars=400)
        joined = "\n\n".join(chunks)
        for paragraph in paragraphs:
            assert paragraph in joined

    def test_chunk_keeps_heading_with_its_section(self) -> None:
        text = "## Pricing\nStarter is $10.\n\n## Enterprise\nTalk to sales."
        chunks = chunk_text(text, max_chars=200)
        assert any("## Pricing" in chunk for chunk in chunks)
        assert all(chunk.strip() for chunk in chunks)

    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []

    def test_collapse_whitespace(self) -> None:
        assert collapse_whitespace("a \n\tb   c") == "a b c"

    def test_truncate_adds_ellipsis_only_when_shortened(self) -> None:
        assert truncate("hello", 10) == "hello"
        assert truncate("hello world", 5).endswith("...")
        assert len(truncate("hello world", 5)) <= 5

    def test_word_count(self) -> None:
        assert word_count("one two three") == 3


class TestSafeJson:
    def test_parses_plain_json(self) -> None:
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fences(self) -> None:
        assert safe_json_loads('```json\n{"a": 1}\n```') == {"a": 1}

    def test_extracts_object_from_surrounding_prose(self) -> None:
        assert safe_json_loads('Here you go: {"a": 1} hope that helps') == {"a": 1}

    def test_returns_none_on_garbage(self) -> None:
        assert safe_json_loads("not json at all") is None

    def test_handles_nested_braces(self) -> None:
        assert safe_json_loads('x {"a": {"b": [1, 2]}} y') == {"a": {"b": [1, 2]}}


class TestRetry:
    def test_returns_on_first_success(self) -> None:
        async def run() -> str:
            return "ok"

        assert asyncio.run(with_retry(run, attempts=3, base_delay=0)) == "ok"

    def test_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        async def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("boom")
            return "ok"

        assert asyncio.run(with_retry(flaky, attempts=5, base_delay=0)) == "ok"
        assert calls["n"] == 3

    def test_raises_after_exhausting_attempts(self) -> None:
        async def always() -> None:
            raise ConnectionError("boom")

        with pytest.raises(RetryExhaustedError):
            asyncio.run(with_retry(always, attempts=2, base_delay=0))

    def test_does_not_retry_a_non_retryable_error(self) -> None:
        calls = {"n": 0}

        async def bad() -> None:
            calls["n"] += 1
            raise ValueError("permanent")

        with pytest.raises(ValueError):
            asyncio.run(with_retry(bad, attempts=5, base_delay=0,
                                   retry_on=lambda exc: isinstance(exc, ConnectionError)))
        assert calls["n"] == 1


class TestGatherLimited:
    def test_preserves_input_order(self) -> None:
        async def value(n: int) -> int:
            await asyncio.sleep(0)
            return n

        result = asyncio.run(gather_limited([value(i) for i in range(10)], limit=3))
        assert result == list(range(10))

    def test_never_exceeds_the_limit(self) -> None:
        active = {"now": 0, "peak": 0}

        async def tracked(n: int) -> int:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            await asyncio.sleep(0.01)
            active["now"] -= 1
            return n

        asyncio.run(gather_limited([tracked(i) for i in range(12)], limit=3))
        assert active["peak"] <= 3

    def test_empty_input(self) -> None:
        assert asyncio.run(gather_limited([], limit=3)) == []


class TestDates:
    def test_parse_iso_date(self) -> None:
        assert parse_date("2026-01-15").year == 2026

    def test_parse_returns_none_for_garbage(self) -> None:
        assert parse_date("sometime last year") is None

    def test_parse_returns_none_for_none(self) -> None:
        assert parse_date(None) is None

    def test_utcnow_is_aware(self) -> None:
        assert utcnow().tzinfo is not None


class TestIterables:
    def test_dedupe_preserves_order(self) -> None:
        from app.utils.text import dedupe_preserving_order

        items: Iterable[str] = ["b", "a", "b", "c", "a"]
        assert dedupe_preserving_order(items) == ["b", "a", "c"]
