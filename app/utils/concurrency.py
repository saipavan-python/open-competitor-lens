"""Bounded concurrency helpers.

Every fan-out in the system goes through :func:`gather_limited`. An unbounded
``asyncio.gather`` over a candidate list is how a research run takes down a
search provider's quota and gets the whole run rate-limited.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from typing import Any


class _BoundedSemaphore:
    """Reusable semaphore holder, so a limit can be attached to a session."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self._semaphore = asyncio.Semaphore(self.limit)

    async def __aenter__(self) -> _BoundedSemaphore:
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        self._semaphore.release()


async def gather_limited[T](
    coros: Sequence[Coroutine[Any, Any, T]], *, limit: int = 5
) -> list[T]:
    """Run coroutines with at most ``limit`` in flight, preserving input order.

    A coroutine that raises does not cancel its siblings; the exception surfaces
    at the end. One failed competitor must not abort the other four.
    """
    if not coros:
        return []
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return list(await asyncio.gather(*(run(coro) for coro in coros)))


async def gather_settled[T](
    coros: Sequence[Coroutine[Any, Any, T]], *, limit: int = 5
) -> list[tuple[T | None, BaseException | None]]:
    """Like :func:`gather_limited` but returns ``(value, error)`` pairs.

    Used wherever partial success is the desired behaviour: research collection
    reports what worked and what did not, rather than failing wholesale.
    """
    if not coros:
        return []

    async def run(coro: Coroutine[Any, Any, T]) -> tuple[T | None, BaseException | None]:
        try:
            return await coro, None
        except Exception as exc:  # noqa: BLE001 - the error is returned, not raised
            return None, exc

    return await gather_limited([run(coro) for coro in coros], limit=limit)


async def run_with_semaphore[T](
    coro: Coroutine[Any, Any, T], semaphore: asyncio.Semaphore
) -> T:
    """Await one coroutine while holding ``semaphore``."""
    async with semaphore:
        return await coro


def make_limiter(limit: int) -> Callable[[], _BoundedSemaphore]:
    """A factory for a shared, reusable concurrency limiter."""
    return lambda: _BoundedSemaphore(limit)
