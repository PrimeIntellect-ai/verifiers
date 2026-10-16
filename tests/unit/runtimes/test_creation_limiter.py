"""A creation limiter's reservation never waits on the loop's default executor: a
pipeline's burst of blocking threaded work there must not hold a box create's slot."""

import asyncio
import threading

from verifiers.v1.runtimes import limiters
from verifiers.v1.runtimes.limiters import CreationLimiter


async def test_a_reservation_goes_through_while_the_default_executor_is_full(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(limiters, "LIMITER_DIR", tmp_path)
    busy = threading.Event()
    burst = [asyncio.ensure_future(asyncio.to_thread(busy.wait, 10)) for _ in range(64)]
    try:
        async with asyncio.timeout(2.0):
            async with CreationLimiter("prime-sandbox", "run", per_sec=100.0):
                pass
    finally:
        busy.set()
        await asyncio.gather(*burst)
