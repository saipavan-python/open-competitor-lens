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
    coros: Sequence[Coroutine[Any, Any, T]], *, limit: int = 5,
    deadline: float | None = None
) -> list[T]:
    """Run coroutines with at most ``limit`` in flight, preserving input order.

    A coroutine that raises does not cancel its siblings; the first exception
    encountered, in input order, is re-raised once every task has settled. One
    failed competitor must not abort the other four.

    ``deadline`` is the total seconds the whole batch may take, not a per-item
    timeout. See :func:`gather_settled`.
    """
    results = await gather_settled(coros, limit=limit, deadline=deadline)
    for _, error in results:
        if error is not None:
            raise error
    return [value for value, _ in results]


async def gather_settled[T](
    coros: Sequence[Coroutine[Any, Any, T]], *, limit: int = 5,
    deadline: float | None = None
) -> list[tuple[T | None, BaseException | None]]:
    """Like :func:`gather_limited` but returns ``(value, error)`` pairs.

    Used wherever partial success is the desired behaviour: research collection
    reports what worked and what did not, rather than failing wholesale.

    ``deadline`` bounds the batch as a whole, not each item. Work still in
    flight when it expires is cancelled and reported as a :class:`TimeoutError`
    in its slot, so a batch cannot overrun its budget by starting work it had no
    time to finish. Naming it ``deadline`` rather than ``timeout`` is deliberate:
    a per-item timeout would be a different contract, and would be easy to
    misread as one.
    """
    if not coros:
        return []
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    tasks = [asyncio.ensure_future(run(coro)) for coro in coros]
    try:
        await asyncio.wait(tasks, timeout=deadline)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    results: list[tuple[T | None, BaseException | None]] = []
    for task, coro in zip(tasks, coros, strict=True):
        if task.cancelled():
            # A coroutine cancelled before its first step is never awaited, and
            # Python warns about each one. Close the caller's coroutine: the
            # task's own coroutine is a wrapper, so closing that would leave the
            # inner one dangling.
            close = getattr(coro, "close", None)
            if close is not None:
                close()
            results.append((None, TimeoutError("batch deadline expired")))
        else:
            error = task.exception()
            results.append((None, error) if error else (task.result(), None))
    return results


async def run_with_semaphore[T](
    coro: Coroutine[Any, Any, T], semaphore: asyncio.Semaphore
) -> T:
    """Await one coroutine while holding ``semaphore``."""
    async with semaphore:
        return await coro


def make_limiter(limit: int) -> Callable[[], _BoundedSemaphore]:
    """A factory for a shared, reusable concurrency limiter."""
    return lambda: _BoundedSemaphore(limit)
