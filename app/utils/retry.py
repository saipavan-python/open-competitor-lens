"""Retry with exponential backoff and jitter.

Only genuinely transient failures are retried. Retrying a 404 or a validation
error just burns the budget and delays the report, so the retry predicate is
explicit and defaults to the transient classes.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from app.core.exceptions import AppError


class RetryExhaustedError(AppError):
    """Every attempt failed. Wraps the final error as ``__cause__``."""

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        super().__init__(
            f"All {attempts} attempts failed; last error: {type(last_error).__name__}: "
            f"{last_error}",
            retryable=False,
        )
        self.attempts = attempts
        self.last_error = last_error


def default_retry_on(exc: BaseException) -> bool:
    """Transient by default: rate limits, timeouts and connection errors."""
    if isinstance(exc, AppError):
        return exc.retryable
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError)):
        return True
    return type(exc).__name__ in {
        "APITimeoutError", "APIConnectionError", "InternalServerError",
        "RateLimitError", "ServiceUnavailableError", "OverloadedError",
    }


async def with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.25,
    retry_on: Callable[[BaseException], bool] = default_retry_on,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Call ``fn`` with exponential backoff.

    ``base_delay=0`` skips sleeping, which is what the tests use so a retry
    assertion costs milliseconds instead of seconds.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - the predicate decides what is retryable
            last_error = exc
            if attempt == attempts or not retry_on(exc):
                if attempt == attempts and retry_on(exc):
                    raise RetryExhaustedError(attempts, exc) from exc
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            if base_delay > 0:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                await asyncio.sleep(delay * (1 + random.uniform(-jitter, jitter)))
    assert last_error is not None  # noqa: S101 - loop always sets it before exiting
    raise RetryExhaustedError(attempts, last_error)
