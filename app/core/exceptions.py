"""Typed exception hierarchy.

Every failure the system raises on purpose is an :class:`AppError` carrying a
``retryable`` flag. Anything else escaping a node is a bug and is reported as
``UnknownError`` with a generic message so no stack trace reaches an API client.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for every deliberate failure in the application."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        provider: str | None = None,
        step: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.provider = provider
        self.step = step

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe representation safe to hand to an API client.

        Contains the message and routing metadata only - never a traceback,
        never a credential, never an internal filesystem path.
        """
        return {
            "type": type(self).__name__,
            "message": self.message,
            "retryable": self.retryable,
            "provider": self.provider,
            "step": self.step,
        }


class ConfigurationError(AppError):
    """A required configuration value is missing or invalid."""


class ProviderNotConfiguredError(AppError):
    """A provider was requested but its credentials are absent.

    Raised instead of silently degrading, so an unconfigured integration is
    always visible rather than pretending to work.
    """

    def __init__(self, provider: str, missing_env_var: str, *, message: str | None = None
                 ) -> None:
        super().__init__(
            message
            or f"Provider '{provider}' is not configured. Set {missing_env_var}.",
            retryable=False,
            provider=provider,
        )
        self.missing_env_var = missing_env_var


class ProviderError(AppError):
    """A provider call failed in a way that is the caller's problem."""


class RateLimitError(ProviderError):
    """HTTP 429 or a provider quota exhaustion."""

    def __init__(self, message: str, *, provider: str | None = None,
                 retry_after: float | None = None) -> None:
        super().__init__(message, retryable=True, provider=provider)
        self.retry_after = retry_after


class TimeoutError_(ProviderError):
    """An outbound call exceeded its configured timeout."""

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message, retryable=True, provider=provider)


class SSRFError(AppError):
    """A URL was rejected by the outbound-request guard."""


class ValidationError(AppError):
    """User input failed validation."""


class BudgetExceededError(AppError):
    """A research budget (queries, pages, tokens, browser steps) is exhausted."""


class EvidenceNotSupportedError(AppError):
    """A claim tried to cite a source the run never actually retrieved."""

    def __init__(self, message: str, *, claim: str | None = None,
                 source_id: str | None = None) -> None:
        super().__init__(message, retryable=False, step="evidence")
        self.claim = claim
        self.source_id = source_id


class ExtractionError(AppError):
    """A web page could not be turned into usable text."""


class UnknownError(AppError):
    """An unexpected failure, reported without internal detail."""
