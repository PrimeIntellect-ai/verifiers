"""The event loop's liveness and the outage rules for a model endpoint's failures.
`Heartbeat` notes each stall of the loop (a wakeup late by `LAG_S`), so a client-side
timeout whose request straddled one is read as the stall's, not the endpoint's; `outage`
says whether a `ProviderError` failed before any response (an upstream 502/503/504, a
connection not established), `timed_out` whether it is the client's own timeout, and
`Timeouts` is the rule that makes an outage of them: the timeouts of
`TIMEOUTS_FOR_OUTAGE` distinct sessions within `TIMEOUT_WINDOW_S` (one alone is a
saturated server's slow answer, the caller's to retry)."""

import asyncio
import time
from collections import deque
from collections.abc import Callable

import httpx

from verifiers.v1.errors import ProviderError

HEARTBEAT_S = 1.0
"""Seconds between two wakeups of the loop's heartbeat."""
LAG_S = 4.0
"""A wakeup this many seconds late is a stall of the event loop: a client timeout whose
request straddled one is the stall's, not the endpoint's (stalls chained into minutes of
holds while the upstream kept answering)."""
OUTAGE_STATUSES = frozenset({502, 503, 504})
"""The statuses of a request that failed before any response: an upstream 502/503/504,
or the client's own status for a connection it could not establish
(`EvalClient._request`: refused or reset 503, a connect timeout 504)."""
TIMEOUTS_FOR_OUTAGE = 3
"""Client-side timeouts from as many distinct sessions within `TIMEOUT_WINDOW_S` that
read as an outage. One is a saturated server's slow answer, not an outage: lone connect
timeouts under 45-56 concurrent streams once held every request 11 times (431 s, up to
71 requests released at once)."""
TIMEOUT_WINDOW_S = 60.0
"""Seconds within which the timeouts of `TIMEOUTS_FOR_OUTAGE` sessions must fall."""


class Heartbeat:
    """The event loop's liveness: a task that wakes every `HEARTBEAT_S` and notes each
    wakeup over `LAG_S` late as a stall `(from, to)` in `time.monotonic()`, one `[loop]
    stalled N s` log line per stall (so an inspection of the log sees what lived in
    memory alone)."""

    def __init__(self, log: Callable[[str], None] = print):
        self.log = log
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
                self.log(f"[loop] stalled {now - self._last:.1f} s")
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


def outage(error: BaseException) -> bool:
    """Whether a model call failed with the provider unreachable: a typed
    `ProviderError` at one of `OUTAGE_STATUSES`."""
    return isinstance(error, ProviderError) and error.status_code in OUTAGE_STATUSES


def timed_out(error: BaseException) -> bool:
    """Whether the failure is the client's own timeout (`EvalClient._request` raises an
    httpx timeout as a 504) rather than an upstream 5xx or a connection refused or
    reset."""
    return isinstance(error.__cause__, httpx.TimeoutException)


class Timeouts:
    """The outage rule for one endpoint's client-side timeouts: `note(session, began)`
    records one and says whether the timeouts of `count` distinct sessions now fall
    within `window` seconds; one whose request straddled a stall of the loop
    (`Heartbeat.stalled_since`) is neither noted nor counted: the loop expired it, not
    the endpoint."""

    def __init__(
        self,
        heartbeat: Heartbeat,
        *,
        count: int = TIMEOUTS_FOR_OUTAGE,
        window: float = TIMEOUT_WINDOW_S,
    ):
        self.heartbeat, self.count, self.window = heartbeat, count, window
        self.seen: deque[tuple[float, str | None]] = deque()
        """(`time.monotonic()`, session) of the timeouts within the window."""

    def note(self, session: str | None, began: float) -> bool:
        now = time.monotonic()
        if self.heartbeat.stalled_since(began):
            return False
        self.seen.append((now, session))
        while now - self.seen[0][0] > self.window:
            self.seen.popleft()
        return len({session for _, session in self.seen}) >= self.count


__all__ = [
    "HEARTBEAT_S",
    "LAG_S",
    "OUTAGE_STATUSES",
    "TIMEOUTS_FOR_OUTAGE",
    "TIMEOUT_WINDOW_S",
    "Heartbeat",
    "Timeouts",
    "outage",
    "timed_out",
]
