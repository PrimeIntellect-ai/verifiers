"""Run-scoped creation-rate limiters for the remote runtimes.

A leaky bucket backed by a lock file under the user cache (``~/.cache/verifiers``, falling
back to the temp dir when no home is resolvable), so a provider's creation rate (Modal
sandboxes, Prime sandboxes and tunnels) is enforced across EVERY process of a run — the
eval process and all the elastically-spawned env-server worker processes alike — not just
within one process. One bucket file per name and run: the run is ``$VF_RUN_ID`` when the
launcher sets it, else the process group, so a run's backlog (or the reservations a killed
run left behind) never delays another run.
"""

import asyncio
import fcntl
import logging
import os
import time
from pathlib import Path
from typing import Self

from verifiers.v1.utils.paths import CACHE_DIR

LIMITER_DIR = CACHE_DIR / "limiter"

logger = logging.getLogger(__name__)

BACKLOG_WARN_SECONDS = 10 * 60
"""Backlog above which a reservation logs the bucket for diagnosis. A backlog this deep
usually means admission far outpaces the configured rate, or a killed run left its
reservations behind in the bucket file."""

STALE_BUCKET_SECONDS = 24 * 60 * 60
"""Bucket files untouched this long belong to finished runs and are swept."""


def run_scope() -> str:
    """The key that groups one run's processes: the launcher's ``$VF_RUN_ID``, else the
    process group, which spawned env servers and pool workers inherit."""
    return os.environ.get("VF_RUN_ID") or f"pg{os.getpgid(0)}"


class CreationLimiter:
    """An async leaky bucket shared across processes via a lock file: each `async with`
    reserves the next `1/per_sec`-spaced slot (advancing the on-disk cursor under an exclusive
    flock) and sleeps until it, so the aggregate creation rate across all of the user's
    processes stays at `per_sec`. The reservation runs off the event loop; the wait does not
    hold the lock. Reservations are never released, so a cancelled waiter still holds its
    slot; the backlog drains at `per_sec` regardless."""

    def __init__(self, name: str, per_sec: float) -> None:
        self._interval = 1 / per_sec
        self._name = name
        self._path: Path | None = None
        self._last_warned = 0.0

    def _open(self) -> Path:
        """Resolve the bucket on first use, after the launcher has set the run identity,
        and sweep buckets of long-finished runs while at it."""
        os.makedirs(LIMITER_DIR, exist_ok=True)
        now = time.time()
        for stale in LIMITER_DIR.glob("*.bucket"):
            try:
                if now - stale.stat().st_mtime > STALE_BUCKET_SECONDS:
                    stale.unlink()
            except FileNotFoundError:
                pass
        return LIMITER_DIR / f"{self._name}-{run_scope()}.bucket"

    def _reserve(self) -> float:
        if self._path is None:
            self._path = self._open()
        # Shared buckets require a clock comparable across the run's hosts.
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
                self._name,
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
    """A run-scoped limiter pacing `name`'s creation to `per_sec`/s (None/<= 0 disables).

    All callers (and processes) of a run sharing a `name` share one bucket, so use one rate
    per name."""
    if not per_sec or per_sec <= 0:
        return None
    limiter = _creation_limiters.get(name)
    if limiter is None:
        limiter = _creation_limiters[name] = CreationLimiter(name, per_sec)
    return limiter
