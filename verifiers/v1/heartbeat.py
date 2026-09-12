"""The event loop's liveness. `Heartbeat` notes each stall of the loop (a wakeup late by
`LAG_S`), so a client-side timeout whose request straddled one can be read as the
stall's, not the endpoint's."""

import asyncio
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

HEARTBEAT_S = 1.0
"""Seconds between two wakeups of the loop's heartbeat."""
LAG_S = 4.0
"""A wakeup this many seconds late is a stall of the event loop: a client timeout whose
request straddled one is the stall's, not the endpoint's (stalls chained into minutes of
holds while the upstream kept answering)."""


class Heartbeat:
    """The event loop's liveness: a task that wakes every `HEARTBEAT_S` and notes each
    wakeup over `LAG_S` late as a stall `(from, to)` in `time.monotonic()`, one warning
    per stall (so an inspection of the log sees what lived in memory alone)."""

    def __init__(self) -> None:
        self.stalls: deque[tuple[float, float]] = deque(maxlen=64)
        self._last = time.monotonic()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._last = time.monotonic()
            self._task = asyncio.create_task(self._beat())

    async def _beat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            now = time.monotonic()
            if now - self._last > HEARTBEAT_S + LAG_S:
                self.stalls.append((self._last, now))
                logger.warning("[loop] stalled %.1f s", now - self._last)
            self._last = now

    def stalled_since(self, began: float) -> bool:
        """Whether the loop stalled after `began` (`time.monotonic()`): a noted stall
        ended after it, or the beat is late right now (the stall the caller woke from,
        not yet noted); never while the beat is not running."""
        if self._task is None:
            return False
        now = time.monotonic()
        return now - self._last > HEARTBEAT_S + LAG_S or any(
            to > began for _, to in self.stalls
        )

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None


__all__ = ["HEARTBEAT_S", "LAG_S", "Heartbeat"]
