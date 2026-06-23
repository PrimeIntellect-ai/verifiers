"""Asyncio helpers shared by the v1 execution path."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def gather_cancel_on_error(*awaitables: Awaitable[T]) -> list[T]:
    """Gather in order, cancelling and draining siblings before re-raising a failure."""
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        # Failed siblings can retain live traces and runtimes after gather returns.
        # Cancel and drain them before preserving the original flat exception.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
