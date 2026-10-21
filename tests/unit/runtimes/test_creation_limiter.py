"""A creation limiter's reservation never waits on the loop's default executor: a
pipeline's burst of blocking threaded work there must not hold a box create's slot."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time

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


def _exited_pid() -> int:
    child = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        check=True,
        text=True,
    )
    return int(child.stdout)


async def test_a_killed_processs_reservations_do_not_delay_the_next_launch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(limiters, "LIMITER_DIR", tmp_path)
    gone = _exited_pid()
    bucket = tmp_path / "prime-sandbox-run.bucket"
    # A killed launch reserved an hour of creates; a live process holds the next second.
    bucket.write_text(json.dumps({str(gone): time.time() + 3600}))
    async with asyncio.timeout(2.0):
        async with CreationLimiter("prime-sandbox", "run", per_sec=100.0):
            pass
    ends = json.loads(bucket.read_text())
    assert list(ends) == [str(os.getpid())], (
        "the dead launch's reservations are dropped"
    )
    # A live process's reservations still pace the next create.
    limiter = CreationLimiter("prime-sandbox", "run", per_sec=1.0)
    assert limiter._reserve() > 0.0
