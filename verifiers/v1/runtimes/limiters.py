"""Host-global creation-rate limiters for the remote runtimes.

A leaky bucket backed by a lock file, so a provider's per-account creation rate (Modal
sandboxes, Prime tunnels) is enforced across EVERY process on the host — the single-process
eval and all the elastically-spawned env-server worker processes alike — not just within one
process. So the configured rate is the actual account-wide rate (assuming one env-server
host). Keyed by name: one bucket file per name, shared by every process (and run) on the host.
"""

import asyncio
import fcntl
import os
import tempfile
import time

_LIMITER_DIR = os.path.join(tempfile.gettempdir(), "vf-rate-limiters")


class CreationLimiter:
    """An async leaky bucket shared across processes via a lock file: each `async with`
    reserves the next `1/per_sec`-spaced slot (advancing the on-disk cursor under an exclusive
    flock) and sleeps until it, so the aggregate creation rate across all host processes stays
    at `per_sec`. The reservation runs off the event loop; the wait does not hold the lock."""

    def __init__(self, name: str, per_sec: float) -> None:
        self._interval = 1 / per_sec
        self._path = os.path.join(_LIMITER_DIR, f"{name}.bucket")

    def _reserve(self) -> float:
        os.makedirs(_LIMITER_DIR, exist_ok=True)
        # monotonic is host-wide on Linux, so the cursor is comparable across processes.
        with open(self._path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                data = f.read().strip()
                now = time.monotonic()
                slot = max(now, float(data) if data else 0.0)
                f.seek(0)
                f.truncate()
                f.write(repr(slot + self._interval))
                f.flush()
                return slot - now
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    async def __aenter__(self) -> "CreationLimiter":
        wait = await asyncio.to_thread(self._reserve)
        if wait > 0:
            await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


_creation_limiters: dict[str, CreationLimiter] = {}


def creation_limiter(per_sec: float | None, name: str) -> CreationLimiter | None:
    """A host-global limiter pacing `name`'s creation to `per_sec`/s (None/<= 0 disables).

    All callers (and processes) sharing a `name` share one bucket, so use one rate per name."""
    if not per_sec or per_sec <= 0:
        return None
    limiter = _creation_limiters.get(name)
    if limiter is None:
        limiter = _creation_limiters[name] = CreationLimiter(name, per_sec)
    return limiter


# The prime_tunnel service caps tunnel starts at 512/min per API token — a property of the
# tunnel service, shared by every runtime that opens a prime_tunnel (prime AND modal). One
# host-global limiter, not a per-runtime config knob.
_TUNNELS_PER_MIN = 512
TUNNEL_LIMITER = creation_limiter(_TUNNELS_PER_MIN / 60, "prime-tunnel")
