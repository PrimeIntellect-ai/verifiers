"""Small shared helpers."""

from __future__ import annotations

import asyncio
import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.v1.trace import Trace


# Resolved once on first use: glibc's malloc_trim, or False where it isn't available.
_malloc_trim: object = None

# A rollout parses large request bodies (base64 images) per turn and frees them, but glibc
# keeps the freed arenas, so a long-lived worker's resting RSS climbs and never drops. An
# occasional malloc_trim hands those arenas back to the OS; trimming every Nth rollout (not
# every one) keeps the per-trim arena walk off the hot path.
_TRIM_EVERY_ROLLOUTS = 16
_rollouts_since_trim = 0


def trim_memory() -> None:
    """Best-effort `malloc_trim(0)` — return glibc's freed arenas to the OS. A no-op off
    glibc (musl, macOS), resolved once and then cached. Walks every arena's free lists, so it
    can block for tens of ms on a large heap — call it off the event loop (see
    `trim_memory_periodically`)."""
    global _malloc_trim
    if _malloc_trim is None:
        try:
            _malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
        except (OSError, AttributeError):
            _malloc_trim = False
    if _malloc_trim:
        _malloc_trim(0)


async def trim_memory_periodically() -> None:
    """Call `trim_memory` once every `_TRIM_EVERY_ROLLOUTS` invocations. Invoked per finished
    rollout so a worker's resting RSS is bounded without trimming on every single one. The
    trim runs in a worker thread — `ctypes` releases the GIL during the call, so the heap walk
    doesn't block the event loop (and every other in-flight rollout with it)."""
    global _rollouts_since_trim
    _rollouts_since_trim += 1
    if _rollouts_since_trim >= _TRIM_EVERY_ROLLOUTS:
        _rollouts_since_trim = 0
        await asyncio.to_thread(trim_memory)


def format_time(seconds: float) -> str:
    """A compact human-readable duration (mirrors prime-rl): s / m s / h m / d h."""
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    if seconds < 86400:
        h, rem = divmod(int(seconds), 3600)
        return f"{h}h {rem // 60}m"
    d, rem = divmod(int(seconds), 86400)
    return f"{d}d {rem // 3600}h"


def format_count(n: int) -> str:
    """A compact count (mirrors prime-rl): 936 / 4.8K / 1.5M."""
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1e3:.1f}K"
    return f"{n / 1e6:.1f}M"


def format_reward(traces: list[Trace], digits: int = 2) -> str:
    """Headline reward over completed `traces`: the mean over the non-errored ones
    (error-corrected). When some errored, also append the global mean — over *all*
    completed, errored counting as 0 — in parens, so both the error-corrected and the
    raw reward are visible. "—" when nothing has completed (or all errored)."""
    if not traces:
        return "—"
    clean = [t for t in traces if not t.has_error]
    if not clean:
        return "—"
    reward = f"{sum(t.reward for t in clean) / len(clean):.{digits}f}"
    if len(clean) < len(traces):  # some errored → show the raw global avg alongside
        reward += f" ({sum(t.reward for t in traces) / len(traces):.{digits}f})"
    return reward
