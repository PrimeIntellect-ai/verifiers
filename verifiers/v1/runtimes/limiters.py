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
import logging
import os
import time
from typing import Self

from verifiers.v1.utils.paths import CACHE_DIR

LIMITER_DIR = CACHE_DIR / "limiter"

logger = logging.getLogger(__name__)

BACKLOG_WARN_SECONDS = 10 * 60
"""Backlog above which a reservation logs the bucket for diagnosis. A backlog this deep
usually means admission far outpaces the configured rate, or a killed run left its
reservations behind in the bucket file."""


class CreationLimiter:
    """An async leaky bucket shared across processes via a lock file: each `async with`
    reserves the next `1/per_sec`-spaced slot (advancing the on-disk cursor under an exclusive
    flock) and sleeps until it, so the aggregate creation rate across all of the user's
    processes stays at `per_sec`. The reservation runs off the event loop; the wait does not
    hold the lock. Reservations are never released, so a cancelled waiter still holds its
    slot; the backlog drains at `per_sec` regardless."""

    def __init__(self, name: str, per_sec: float) -> None:
        self._interval = 1 / per_sec
        self._path = LIMITER_DIR / f"{name}.bucket"
        self._last_warned = 0.0

    def _reserve(self) -> float:
        os.makedirs(LIMITER_DIR, exist_ok=True)
        # Shared buckets require a clock comparable across hosts and boots.
        with open(self._path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                data = f.read().strip()
                now = time.time()
                slot = max(now, float(data) if data else 0.0)
                wait = slot - now
                f.seek(0)
                f.truncate()
                f.write(repr(slot + self._interval))
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        # Every queued rollout reserves at once, so warn once a minute, not once per slot.
        if wait > BACKLOG_WARN_SECONDS and now - self._last_warned > 60:
            self._last_warned = now
            logger.warning(
                "%s creation limiter backlog is %.0fs (%d queued at %.2f/s); "
                "delete %s to discard reservations left by a killed run",
                self._path.stem,
                wait,
                wait / self._interval,
                1 / self._interval,
                self._path,
            )
        return wait

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
