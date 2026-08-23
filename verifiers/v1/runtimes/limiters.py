"""User-global creation-rate limiters for the remote runtimes.

A leaky bucket backed by a lock file under the user cache (``~/.cache/verifiers``, falling
back to the temp dir when no home is resolvable), so a provider's per-account creation rate
(Modal sandboxes, Prime tunnels) is enforced across EVERY process for the user —
the single-process eval and all the elastically-spawned env-server worker processes alike — not
just within one process. Keyed by name: one bucket file per name, shared by every process (and
run) for the user.
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
    flock) and sleeps until it, so the aggregate creation rate across all of the user's
    processes stays at `per_sec`. The reservation runs off the event loop; the wait does not
    hold the lock. Backlogs over five minutes fail rather than silently stalling creation."""

    def __init__(self, name: str, per_sec: float) -> None:
        self._interval = 1 / per_sec
        self._path = LIMITER_DIR / f"{name}.bucket"

    def _reserve(self) -> float:
        os.makedirs(LIMITER_DIR, exist_ok=True)
        # Shared buckets require a clock comparable across hosts and boots.
        with open(self._path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                data = f.read().strip()
                now = time.time()
                # flock does not interlock reliably on every shared filesystem
                # (NFS), so a torn read can surface here. A corrupted cursor
                # resets the bucket instead of failing every creation.
                try:
                    cursor = float(data) if data else 0.0
                except ValueError:
                    cursor = 0.0
                slot = max(now, cursor)
                wait = slot - now
                if wait > 5 * 60:
                    raise TimeoutError(
                        f"{self._path.stem} creation limiter backlog of {wait:.1f}s "
                        f"exceeds 300s ({self._path})"
                    )
                f.seek(0)
                f.truncate()
                f.write(repr(slot + self._interval))
                f.flush()
                return wait
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    async def __aenter__(self) -> Self:
        wait = await asyncio.to_thread(self._reserve)
        if wait > 0:
            await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


_creation_limiters: dict[str, CreationLimiter] = {}


def creation_limiter(per_sec: float | None, name: str) -> CreationLimiter | None:
    """A user-global limiter pacing `name`'s creation to `per_sec`/s (None/<= 0 disables).

    All callers (and processes) sharing a `name` share one bucket, so use one rate per name."""
    if not per_sec or per_sec <= 0:
        return None
    limiter = _creation_limiters.get(name)
    if limiter is None:
        limiter = _creation_limiters[name] = CreationLimiter(name, per_sec)
    return limiter
