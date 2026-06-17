"""Worker memory management: hand glibc's freed arenas back to the OS."""

import asyncio
import ctypes

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
