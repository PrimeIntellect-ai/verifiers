"""Creation-rate limiters for the remote runtimes.

A leaky bucket backed by a lock file under the user cache (``~/.cache/verifiers``, falling
back to the temp dir when no home is resolvable), so a provider's creation rate (Modal
sandboxes, Prime sandboxes and tunnels) is enforced across EVERY process that shares the
bucket — the eval process and all the elastically-spawned env-server worker processes alike —
not just within one process. A bucket is named by the limiter's name and a scope the caller
picks; the runtimes pass the run id, so one run's backlog (or the reservations a killed run
left behind) never delays another run.
"""

import asyncio
import fcntl
import os
import time
from typing import Self

from verifiers.v1.utils.paths import CACHE_DIR

LIMITER_DIR = CACHE_DIR / "limiter"


class CreationLimiter:
    """An async leaky bucket shared across processes via a lock file: each `async with`
    reserves the next `1/per_sec`-spaced slot (advancing the on-disk cursor under an exclusive
    flock) and sleeps until it, so the aggregate creation rate across every process sharing
    the bucket stays at `per_sec`. The reservation runs off the event loop; the wait does not
    hold the lock. Reservations are never released, so a cancelled waiter still holds its
    slot; the backlog drains at `per_sec` regardless."""

    def __init__(self, name: str, scope: str, per_sec: float) -> None:
        self._interval = 1 / per_sec
        self._path = LIMITER_DIR / f"{name}-{scope.replace('/', '--')}.bucket"

    def _reserve(self) -> float:
        os.makedirs(LIMITER_DIR, exist_ok=True)
        # Shared buckets require a clock comparable across the run's hosts.
        with open(self._path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                data = f.read().strip()
                now = time.time()
                slot = max(now, float(data) if data else 0.0)
                f.seek(0)
                f.truncate()
                f.write(repr(slot + self._interval))
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return slot - now

    async def __aenter__(self) -> Self:
        wait = await asyncio.to_thread(self._reserve)
        if wait > 0:
            await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


def creation_limiter(
    per_sec: float | None, name: str, scope: str
) -> CreationLimiter | None:
    """A limiter pacing `name`'s creation to `per_sec`/s within `scope` (None/<= 0
    disables). All callers (and processes) sharing a name and scope share one bucket, so
    use one rate per name."""
    if not per_sec or per_sec <= 0:
        return None
    return CreationLimiter(name, scope, per_sec)
