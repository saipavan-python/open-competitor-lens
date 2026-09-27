"""Redaction at the logging boundary.

Found while reviewing the branch before publishing: the redaction code had no
tests at all, and the part that was untested had a hole. The message and the
traceback were passed through :func:`redact`; the structured context fields were
not. ``error`` is one of those fields, and every provider's failure path fills it
with ``str(provider_exception)`` - so a single Tavily or Firecrawl timeout printed
the key to stdout. The module docstring claimed "anything that looks like a
credential is redacted before it reaches a handler", which was true of the parts
nobody reads and false of the parts that carry the failure.

These tests use realistic provider error strings rather than a synthetic token,
because the leak was in the *shape* of the message: the key arrived embedded in a
URL and a header dict, which is how httpx and the OpenAI SDK actually report it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from app.core.exceptions import AppError, ProviderError
from app.core.logging import CONTEXT_FIELDS, JsonFormatter, TextFormatter, redact

FAKE_KEY = "tvly-ABCDEFGHIJKLMNOP1234"


def record(**fields: Any) -> logging.LogRecord:
    entry = logging.LogRecord(
        name="app.providers.web.http", level=logging.WARNING, pathname=__file__,
        lineno=42, msg="transport error", args=None, exc_info=None,
    )
    for name, value in fields.items():
        setattr(entry, name, value)
    return entry


class TestRedact:
    @pytest.mark.parametrize(
        "text",
        [
            f"Authorization: Bearer {FAKE_KEY}",
            f"api_key={FAKE_KEY}",
            f"api-key: {FAKE_KEY}",
            f'password="{FAKE_KEY}"',
            f"sk-{'a' * 32}",
            f"fc-{'b' * 32}",
            f"secret: {FAKE_KEY}",
        ],
    )
    def test_credential_shaped_text_is_redacted(self, text: str) -> None:
        out = redact(text)
        assert "[redacted]" in out

    def test_ordinary_text_is_left_alone(self) -> None:
        """Over-eager redaction makes logs useless, which is its own failure."""
        text = "blocked 169.254.169.254 is a link-local address"
        assert redact(text) == text

    def test_a_clean_provider_error_survives(self) -> None:
        text = "net::ERR_CONNECTION_REFUSED at https://acme.com/pricing"
        assert redact(text) == text


class TestContextFieldsAreRedacted:
    """The regression: context fields went to the handler unredacted."""

    TAVILY_ERROR = (
        "HTTPStatusError: 401 for url "
        "https://api.tavily.com/search?api_key=" + FAKE_KEY
    )
    OPENAI_ERROR = (
        "AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect "
        "API key provided: sk-" + "z" * 40 + "'}}"
    )

    @pytest.mark.parametrize("formatter", [JsonFormatter(), TextFormatter()])
    def test_an_error_context_field_is_redacted(self, formatter: Any) -> None:
        out = formatter.format(record(error=self.TAVILY_ERROR))
        assert FAKE_KEY not in out
        assert "[redacted]" in out

    @pytest.mark.parametrize("formatter", [JsonFormatter(), TextFormatter()])
    def test_a_secret_in_an_openai_error_is_redacted(self, formatter: Any) -> None:
        out = formatter.format(record(error=self.OPENAI_ERROR))
        assert "z" * 40 not in out

    def test_the_message_is_still_redacted(self) -> None:
        out = JsonFormatter().format(record())
        entry = json.loads(out)
        assert entry["message"] == "transport error"

    def test_the_traceback_is_still_redacted(self) -> None:
        try:
            raise RuntimeError(f"request failed with api_key={FAKE_KEY}")
        except RuntimeError:
            import sys

            entry = logging.LogRecord(
                name="app.x", level=logging.ERROR, pathname=__file__, lineno=1,
                msg="boom", args=None, exc_info=sys.exc_info(),
            )
        out = JsonFormatter().format(entry)
        assert FAKE_KEY not in out

    def test_a_clean_error_is_not_mangled(self) -> None:
        """Redaction must not destroy the diagnostics it is protecting."""
        clean = "net::ERR_NAME_NOT_RESOLVED at https://acme.com"
        out = JsonFormatter().format(record(error=clean))
        assert clean in json.loads(out)["error"]

    def test_durations_stay_numeric(self) -> None:
        """``duration_ms=1204`` cannot hold a secret; stringifying it is noise."""
        entry = json.loads(JsonFormatter().format(record(duration_ms=1204)))
        assert entry["duration_ms"] == 1204

    def test_run_and_company_context_survives(self) -> None:
        entry = json.loads(
            JsonFormatter().format(record(run_id="r1", company="Acme", agent="discovery"))
        )
        assert entry["run_id"] == "r1"
        assert entry["company"] == "Acme"
        assert entry["agent"] == "discovery"


class TestThePayloadShape:
    def test_every_context_field_is_emitted_when_present(self) -> None:
        entry = json.loads(
            JsonFormatter().format(record(**{name: f"v-{name}" for name in CONTEXT_FIELDS}))
        )
        for name in CONTEXT_FIELDS:
            assert entry[name] == f"v-{name}"

    def test_a_key_outside_the_context_whitelist_is_dropped(self) -> None:
        """Not a fix, a documented decision: the payload has a fixed shape."""
        entry = json.loads(JsonFormatter().format(record(an_unlisted_field="surprise")))
        assert "an_unlisted_field" not in entry

    def test_the_line_is_one_json_object(self) -> None:
        out = JsonFormatter().format(record())
        assert "\n" not in out
        assert json.loads(out)["logger"] == "app.providers.web.http"


class TestLocalPathsAreScrubbed:
    @pytest.mark.parametrize(
        "text",
        [
            r"config at C:\Users\somebody\project\.env",
            r"C:\Users\somebody\AppData\Local\Temp\x",
            "failed reading /home/deployer/app/config.py",
            "traceback in /Users/bob/repo/app/main.py",
            "wrote /root/.config/app.json",
        ],
    )
    def test_an_absolute_local_path_is_scrubbed(self, text: str) -> None:
        out = redact(text)
        assert "[path]" in out
        assert "somebody" not in out and "deployer" not in out and "bob" not in out

    def test_a_url_is_not_mistaken_for_a_path(self) -> None:
        """``//`` must not make every URL look like a local path."""
        text = "failed fetching https://acme.com/pricing"
        assert redact(text) == text


class TestTheApiErrorBodyIsSafe:
    """``AppError.to_dict`` is what the API hands a client.

    Its docstring promised "never a credential, never an internal filesystem
    path" and returned ``self.message`` verbatim. The LLM adapters build that
    message from ``str(provider_exception)``, so the errors most likely to carry
    a secret were the ones that skipped redaction entirely.
    """

    def test_a_credential_in_the_message_does_not_reach_the_client(self) -> None:
        error = ProviderError(
            f"401 from https://api.openai.com/v1/chat/completions?key={FAKE_KEY}",
            provider="openai",
        )
        assert FAKE_KEY not in json.dumps(error.to_dict())

    def test_a_local_path_in_the_message_does_not_reach_the_client(self) -> None:
        error = ProviderError(r"could not read C:\Users\somebody\project\.env")
        assert "somebody" not in json.dumps(error.to_dict())

    def test_the_routing_metadata_survives(self) -> None:
        error = ProviderError("upstream is down", provider="tavily", step="search", retryable=True)
        body = error.to_dict()
        assert body["type"] == "ProviderError"
        assert body["provider"] == "tavily"
        assert body["step"] == "search"
        assert body["retryable"] is True

    def test_an_ordinary_message_is_unchanged(self) -> None:
        """Redaction must not cost the client its actual diagnosis."""
        error = AppError("Provider 'tavily' is not configured. Set TAVILY_API_KEY.")
        assert error.to_dict()["message"] == error.message

    def test_no_traceback_is_exposed(self) -> None:
        body = ProviderError("boom").to_dict()
        assert set(body) == {"type", "message", "retryable", "provider", "step"}

