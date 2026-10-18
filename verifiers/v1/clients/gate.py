"""The request gate: a per-model cap on concurrent upstream requests, shared host-wide.

Providers limit concurrent requests per API key and model (Prime Inference answers
`429 Too many concurrent requests for this model (limit: 32)`). Every model request a
rollout makes, nested agents and judges included, leaves through an `httpx.AsyncClient`
from `clients.base.build_async_httpx`. With `BaseClientConfig.request_gate` set, that
client is a `GatedAsyncClient`, and each request first takes a permit of the gate for its
(endpoint, API key, model).

Permits are `flock`s on slot files in a shared directory, so every process on the host
(env-server workers, concurrent runs) draws from one budget, and a process that dies
releases its permits with its file descriptors. A state file beside the slots, `flock`ed
around each admission, holds the cap, the adaptive limit and running counters as JSON
(readable while a run is live). Every process using a gate holds a shared `flock` on its
users file; the first one in, finding no other user, starts the state afresh, so what an
earlier run learned or was set to does not carry over to the next.

The cap (`max_concurrent` in the state) is shared: the first admission after a fresh start
writes its config's `max_concurrent`, and from then on every process admits under the
shared cap, whatever its own config says. `resize` moves the cap of live gates (an
operator, e.g. prime-rl's `eval control resize-requests`): a raise takes effect at the
next admission, a lower sheds nothing and admits no one until fewer than the new cap are
in flight. Admission scans the slots up to the cap and any held above it (`slots`), so
requests admitted under a higher cap count until they finish.

The cap is never exceeded. Under it, with `adaptive`, a 429/529 lowers the limit by
`decrease_factor` (at most once per `decrease_cooldown`), and the limit climbs back to the
cap in a straight line over `recover_seconds` from that decrease, whatever the traffic: a
burst of rejections slows the gate for a minute, never for hours. A raise of the cap
restarts the limit at the new cap; a lower keeps a lowered limit already below it.

Admission is first come, first served across every process of the host: a request that
cannot take a slot at once takes a ticket in the shared state, and a free slot goes to
the oldest live ticket, whichever process holds it. A process's waiters take their
tickets in arrival order and only the oldest polls; a release in the process wakes it at
once, a release elsewhere at its next poll. The tickets of a process that died are
dropped at the next admission. The state keeps the queue's length and an average of the
time an admitted request waited (`wait_seconds`).

A rejected request returns its permit, waits for
`Retry-After` (else exponential backoff) and queues again, up to `max_rejections` times;
after that the provider's response reaches the caller unchanged, as final: the
interception server does not retry it.

A streamed response holds its permit until it is closed.
"""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import fcntl
import hashlib
import json
import logging
import os
import random
import re
import tempfile
import threading
import time
import weakref
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic_core import from_json

from verifiers.v1.configs.client import RequestGateConfig

logger = logging.getLogger(__name__)

REJECTION_STATUSES = frozenset({429, 529})
"""Provider statuses that mean "too much load from you": 429 (rate or concurrency limit)
and 529 (Anthropic's overloaded)."""
POLL_SECONDS = (0.05, 0.2)
"""Jittered interval at which a blocked admission rechecks the shared slots: only the
first waiter of the process polls (a release in the process wakes it at once); the rest
wait their turn in arrival order."""
STATE_BYTES = 1 << 22
"""Most bytes of a gate's state file read back (its ticket queue included)."""
WAIT_SMOOTHING = 0.05
"""Weight of each admitted request's wait in the state's moving average."""
MODEL_EXTENSION = "verifiers.model"
"""`httpx.Request.extensions` key a caller sets to name the request's model, so the gate
need not parse the body (the eval client sets it; SDK requests fall back to parsing)."""

Outcome = Literal["ok", "rejected", "other"]


def default_gate_dir() -> Path:
    return Path(tempfile.gettempdir()) / f"verifiers-request-gates-{os.getuid()}"


def gate_dir(config: RequestGateConfig) -> Path:
    """The directory of the shared gate state `config` uses."""
    return Path(config.dir) if config.dir else default_gate_dir()


def gate_key(
    url: httpx.URL, credential: str, model: str, scope: str | None = None
) -> str:
    """The gate a request belongs to: one per endpoint origin, credential and model, since
    providers limit per key and model. The credential is only hashed."""
    origin = f"{url.scheme}://{url.host}:{url.port or ''}"
    identity = f"{origin}\n{credential}\n{model}"
    if scope is not None:
        identity += f"\n{scope}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


class RequestPermit:
    """An atomic pool admission transferred to the first upstream request."""

    def __init__(self, gate, config, slot):
        self.gate, self.config, self.slot = gate, config, slot
        self.used = False

    def take(self, gate):
        if not self.used and gate is self.gate:
            self.used = True
            return self.slot
        return None

    def close(self):
        if not self.used:
            self.used = True
            self.gate.release(self.slot, self.config, "other")


reserved_permit: ContextVar[RequestPermit | None] = ContextVar(
    "model_pool_permit", default=None
)


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)[:64]


def _count(value: Any) -> int:
    """A positive count from the shared state, else 0."""
    return value if type(value) is int and value > 0 else 0


@contextlib.contextmanager
def _locked_state(fd: int) -> Iterator[dict[str, Any]]:
    """The state in file `fd` under its exclusive `flock`, written back on exit if
    changed."""
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        raw = os.pread(fd, STATE_BYTES, 0)
        try:
            state = json.loads(raw) if raw else {}
        except ValueError:
            state = {}
        if not isinstance(state, dict):
            state = {}
        before = dict(state)
        yield state
        if state != before:
            data = json.dumps(state, sort_keys=True).encode()
            os.ftruncate(fd, 0)
            os.pwrite(fd, data, 0)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


class _Waiter:
    """One blocked admission: woken (from any thread) when it reaches the head of the
    process's queue, and while there by each release in the process."""

    def __init__(
        self, loop: asyncio.AbstractEventLoop, ticket: int | None = None
    ) -> None:
        self.loop = loop
        self.event = asyncio.Event()
        self.ticket = ticket
        """Its place in the host-wide queue (`ModelGate.take_ticket`)."""

    def wake(self) -> bool:
        """False when its loop is gone (it will never run again)."""
        try:
            self.loop.call_soon_threadsafe(self.event.set)
        except RuntimeError:
            return False
        return True


class ModelGate:
    """One (endpoint, key, model) budget: process-local view of the shared slot files.
    Thread-safe; admission is FIFO among the process's waiters, of every event loop:
    only the first polls the shared slots."""

    def __init__(self, directory: Path, key: str, model: str) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.model = model
        self.prefix = directory / f"{_slug(model)}-{key}"
        self.state_path = self.prefix.with_name(self.prefix.name + ".state.json")
        self._mutex = threading.Lock()
        self._releases: deque[tuple[int, RequestGateConfig, Outcome]] = deque()
        """Releases not yet applied (see `release`)."""
        self._fds: dict[int, int] = {}
        self._held: set[int] = set()
        self._state_fd: int | None = None
        self._users_fd: int | None = os.open(
            self.prefix.with_name(self.prefix.name + ".users"),
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        try:
            fcntl.flock(self._users_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass  # another live process uses the gate: keep what it learned
        else:
            with self._state() as state:
                state.clear()
        fcntl.flock(self._users_fd, fcntl.LOCK_SH)
        self._queue: deque[_Waiter] = deque()
        """The process's blocked admissions, oldest first; the head polls."""
        self._stale: set[int] = set()
        """Tickets of waiters whose loop closed, dropped at the next admission."""
        self._queue_lock = threading.RLock()
        """Guards `_queue`; reentrant, as a finalizer's release may run inside it."""

    def _slot_fd(self, slot: int) -> int:
        fd = self._fds.get(slot)
        if fd is None:
            path = self.prefix.with_name(f"{self.prefix.name}.slot{slot}")
            fd = self._fds[slot] = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        return fd

    @contextlib.contextmanager
    def _state(self) -> Iterator[dict[str, Any]]:
        """The shared state under its exclusive `flock`, written back on exit. Callers
        hold `_mutex` (an `flock` does not exclude threads sharing the descriptor)."""
        if self._state_fd is None:
            self._state_fd = os.open(self.state_path, os.O_RDWR | os.O_CREAT, 0o600)
        with _locked_state(self._state_fd) as state:
            yield state

    @staticmethod
    def _cap(state: dict[str, Any], config: RequestGateConfig) -> int:
        """The shared cap; the first admission after a fresh start sets it from `config`."""
        cap = _count(state.get("max_concurrent"))
        if not cap:
            cap = state["max_concurrent"] = config.max_concurrent
        return cap

    @staticmethod
    def _limit(state: dict[str, Any], config: RequestGateConfig, cap: int) -> float:
        """The limit now: the cap, or after a rejection the lowered limit climbing back
        to the cap over `recover_seconds` from the decrease."""
        lowered = state.get("lowered")
        if not config.adaptive or not isinstance(lowered, int | float):
            return float(cap)
        since = time.time() - float(state.get("decreased_at", 0.0))
        if since >= config.recover_seconds:
            state.pop("lowered", None)
            return float(cap)
        return _clamp(
            lowered + (cap - lowered) * since / config.recover_seconds, config, cap
        )

    @staticmethod
    def _live_tickets(state: dict[str, Any]) -> list[list[Any]]:
        """The queue's tickets `[number, pid, since]`, oldest first, without those of
        processes that are gone."""
        tickets = state.get("tickets")
        if not isinstance(tickets, list):
            return []
        alive: dict[int, bool] = {}
        live = []
        for ticket in tickets:
            if not (isinstance(ticket, list) and len(ticket) == 3):
                continue
            pid = ticket[1]
            if pid not in alive:
                try:
                    os.kill(pid, 0)
                    alive[pid] = True
                except ProcessLookupError:
                    alive[pid] = False
                except (PermissionError, TypeError, OverflowError):
                    alive[pid] = isinstance(pid, int)
            if alive[pid]:
                live.append(ticket)
        live.sort(key=lambda ticket: ticket[0])
        return live

    def take_ticket(self) -> int:
        """Join the host-wide queue: a ticket behind every waiting request."""
        with self._mutex, self._state() as state:
            number = _count(state.get("next_ticket")) or 1
            state["next_ticket"] = number + 1
            state["tickets"] = [
                *self._live_tickets(state),
                [number, os.getpid(), time.time()],
            ]
            state["waiting"] = len(state["tickets"])
            return number

    def drop_ticket(self, number: int) -> None:
        """Leave the queue without a slot (the request was cancelled)."""
        with self._mutex, self._state() as state:
            state["tickets"] = [t for t in self._live_tickets(state) if t[0] != number]
            state["waiting"] = len(state["tickets"])

    def try_admit(
        self, config: RequestGateConfig, ticket: int | None = None
    ) -> int | None:
        """Take a free slot if fewer than the limit are held host-wide and no older
        ticket waits for it, else None. Without a `ticket` the request is behind every
        waiting one; an admitted ticket leaves the queue."""
        try:
            with self._mutex, self._state() as state:
                cap = self._cap(state, config)
                limit = self._limit(state, config, cap)
                tickets = self._live_tickets(state)
                if self._stale:
                    stale, self._stale = self._stale, set()
                    tickets = [t for t in tickets if t[0] not in stale]
                ahead = (
                    len(tickets)
                    if ticket is None
                    else sum(1 for t in tickets if t[0] < ticket)
                )
                held, free, top = 0, None, 0
                for slot in range(max(cap, _count(state.get("slots")))):
                    if slot in self._held:
                        held, top = held + 1, slot + 1
                        continue
                    fd = self._slot_fd(slot)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        held, top = held + 1, slot + 1
                        continue
                    if free is None:
                        free = slot
                    else:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                # Room for this request: fewer than the limit held, counting the older
                # tickets as ahead of it.
                if free is not None and held + ahead >= int(limit):
                    fcntl.flock(self._fds[free], fcntl.LOCK_UN)
                    free = None
                if free is not None:
                    self._held.add(free)
                    state["admitted"] = state.get("admitted", 0) + 1
                    top = max(top, free + 1)
                    if ticket is not None:
                        mine = next((t for t in tickets if t[0] == ticket), None)
                        tickets = [t for t in tickets if t[0] != ticket]
                        if mine is not None:
                            waited = max(0.0, time.time() - float(mine[2]))
                            average = float(state.get("wait_seconds", 0.0))
                            state["wait_seconds"] = round(
                                average + WAIT_SMOOTHING * (waited - average), 3
                            )
                    elif "wait_seconds" in state:
                        average = float(state["wait_seconds"])
                        state["wait_seconds"] = round(average * (1 - WAIT_SMOOTHING), 3)
                state["tickets"] = tickets
                state["waiting"] = len(tickets)
                # slots above the cap stay scanned while a request admitted under a
                # higher cap holds one; admission only takes the lowest free slot
                state.update(model=self.model, slots=max(cap, top), limit=limit)
                state["in_flight"] = held + (free is not None)
                return free
        finally:
            self._apply_releases()

    def release(self, slot: int, config: RequestGateConfig, outcome: Outcome) -> None:
        """Return `slot` and feed the call's outcome to the adaptive limit. A release
        may come from a garbage-collector finalizer (an unclosed stream), which can run
        inside this thread's own `_mutex` section; `_mutex` is not reentrant, so the
        release is queued, and applied here when `_mutex` is free, else by its holder
        when it leaves."""
        self._releases.append((slot, config, outcome))
        self._apply_releases()
        self._wake_head()

    def _wake_head(self) -> None:
        """Wake the first waiter: a slot of this process just freed."""
        with self._queue_lock:
            while self._queue and not self._queue[0].wake():
                self._forget(self._queue.popleft())  # its loop closed

    def _forget(self, waiter: _Waiter) -> None:
        """A waiter whose loop closed never runs its cleanup: its ticket is dropped at the
        process's next admission, or it would hold a place in the host-wide queue for as
        long as the process lives. Not here: a release may run inside `_mutex`."""
        if waiter.ticket is not None:
            self._stale.add(waiter.ticket)

    def _apply_releases(self) -> None:
        # Recheck after each pass: a release queued while the mutex was held, by
        # another thread or a finalizer, finds it busy and leaves the work to us.
        while self._releases and self._mutex.acquire(blocking=False):
            try:
                while self._releases:
                    self._release(*self._releases.popleft())
            finally:
                self._mutex.release()

    def _release(self, slot: int, config: RequestGateConfig, outcome: Outcome) -> None:
        """`release` under `_mutex`."""
        if slot not in self._held:
            return
        self._held.discard(slot)
        fcntl.flock(self._fds[slot], fcntl.LOCK_UN)
        if outcome == "other":
            return
        with self._state() as state:
            cap = self._cap(state, config)
            limit = self._limit(state, config, cap)
            now = time.time()
            if outcome == "ok":
                return
            state["rejections"] = state.get("rejections", 0) + 1
            if (
                config.adaptive
                and now - state.get("decreased_at", 0.0) >= config.decrease_cooldown
            ):
                lowered = _clamp(limit * config.decrease_factor, config, cap)
                state.update(lowered=lowered, limit=lowered, decreased_at=now)
                logger.warning(
                    "request gate %s: provider rejected a request, limit %d -> %d",
                    self.model,
                    int(limit),
                    int(lowered),
                )

    async def acquire(self, config: RequestGateConfig) -> int:
        """Wait for a slot: at once when no request of the host waits and one is free,
        else first come, first served across the host (module doc). The process's
        waiters queue in arrival order and only the first polls the shared slots, every
        `POLL_SECONDS`; a release in this process wakes it at once."""
        with self._queue_lock:
            idle = not self._queue
        if idle:
            slot = self.try_admit(config)
            if slot is not None:
                return slot
        ticket = self.take_ticket()
        waiter = _Waiter(asyncio.get_running_loop(), ticket)
        with self._queue_lock:
            self._queue.append(waiter)
            first = self._queue[0] is waiter
        admitted = False
        try:
            if not first:
                await waiter.event.wait()  # set once it is first
            while True:
                waiter.event.clear()
                slot = self.try_admit(config, ticket)
                if slot is not None:
                    admitted = True
                    return slot
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(random.uniform(*POLL_SECONDS)):
                        await waiter.event.wait()
        finally:
            if not admitted:
                with contextlib.suppress(OSError):
                    self.drop_ticket(ticket)
            with self._queue_lock:
                was_first = bool(self._queue) and self._queue[0] is waiter
                with contextlib.suppress(ValueError):
                    self._queue.remove(waiter)
                if was_first:
                    while self._queue and not self._queue[0].wake():
                        self._forget(self._queue.popleft())

    def close_inherited(self) -> None:
        """Drop the descriptors a forked child inherited: their locks are the parent's."""
        for fd in [*self._fds.values(), self._state_fd, self._users_fd]:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
        self._fds.clear()
        self._held.clear()
        self._releases.clear()
        self._queue.clear()
        self._stale.clear()
        self._queue_lock = threading.RLock()
        self._state_fd = self._users_fd = None
        self._mutex = threading.Lock()


def _clamp(limit: float, config: RequestGateConfig, cap: int) -> float:
    """`limit` within `[min_concurrent, cap]`; the cap wins over a higher floor."""
    return min(max(float(limit), config.min_concurrent), cap)


def _summary(state: dict[str, Any], updated: float) -> dict[str, Any]:
    cap = _count(state.get("max_concurrent")) or None
    limit = state.get("limit")
    if cap is not None and isinstance(limit, int | float):
        limit = int(min(limit, cap))
    else:
        limit = cap
    return {
        "model": state.get("model"),
        "max_concurrent": cap,
        "limit": limit,
        "in_flight": state.get("in_flight"),
        "admitted": state.get("admitted", 0),
        "rejections": state.get("rejections", 0),
        "waiting": state.get("waiting", 0),
        "wait_seconds": state.get("wait_seconds", 0.0),
        "updated": updated,
    }


def gate_states(directory: Path) -> list[dict[str, Any]]:
    """The gates under `directory`: each one's model, cap (`max_concurrent`), effective
    `limit`, `in_flight` (as of its last admission attempt), counters, and `updated`, the
    time its state last changed. Gates of finished runs stay listed until a run reuses
    them (their `updated` tells)."""
    states = []
    for path in sorted(directory.glob("*.state.json")):
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            raw = os.pread(fd, STATE_BYTES, 0)
            updated = os.fstat(fd).st_mtime
        finally:
            os.close(fd)
        try:
            state = json.loads(raw) if raw else {}
        except ValueError:
            continue
        if isinstance(state, dict):
            states.append(_summary(state, updated))
    return states


def resize(
    directory: Path, max_concurrent: int, model: str | None = None
) -> list[dict[str, Any]]:
    """Set the shared cap of the gates under `directory` (only `model`'s, if given) to
    `max_concurrent`, for every process using them; returns their states as
    `gate_states` does. A raise restarts the adaptive limit at the new cap, a lower keeps
    a learned limit already below it (module doc). It holds until the gate's users are
    gone: the next run starts from its own config."""
    if max_concurrent < 1:
        raise ValueError(f"a request cap must be at least 1, not {max_concurrent}")
    states = []
    for path in sorted(directory.glob("*.state.json")):
        try:
            fd = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            continue
        try:
            with _locked_state(fd) as state:
                if model is not None and state.get("model") != model:
                    continue
                old = _count(state.get("max_concurrent"))
                if max_concurrent > old:
                    state.pop("limit", None)
                    state.pop("lowered", None)
                state["max_concurrent"] = max_concurrent
                state["slots"] = max(max_concurrent, old, _count(state.get("slots")))
                states.append(_summary(state, time.time()))
        finally:
            os.close(fd)
    return states


_GATES: dict[tuple[Path, str], ModelGate] = {}
_GATES_LOCK = threading.Lock()


def _reset_after_fork() -> None:
    global _GATES_LOCK
    _GATES_LOCK = threading.Lock()
    for gate in _GATES.values():
        gate.close_inherited()
    _GATES.clear()


os.register_at_fork(after_in_child=_reset_after_fork)


def model_gate(directory: Path, key: str, model: str) -> ModelGate:
    """The process's `ModelGate` for `key` under `directory`."""
    with _GATES_LOCK:
        gate = _GATES.get((directory, key))
        if gate is None:
            gate = _GATES[(directory, key)] = ModelGate(directory, key, model)
        return gate


def request_model(request: httpx.Request) -> str | None:
    """The model a request asks for: the `MODEL_EXTENSION`, else the JSON body's `model`."""
    model = request.extensions.get(MODEL_EXTENSION)
    if isinstance(model, str):
        return model
    if request.method != "POST":
        return None
    try:
        body = from_json(request.content)
    except (httpx.RequestNotRead, ValueError):
        return None
    model = body.get("model") if isinstance(body, dict) else None
    return model if isinstance(model, str) else None


def retry_after(response: httpx.Response) -> float | None:
    """Seconds the provider asked to wait: `retry-after-ms`, else `Retry-After` as
    seconds or an HTTP date."""
    headers = response.headers
    with contextlib.suppress(ValueError):
        if (ms := headers.get("retry-after-ms")) is not None:
            return max(0.0, float(ms) / 1000)
    value = headers.get("retry-after")
    if value is None:
        return None
    with contextlib.suppress(ValueError):
        return max(0.0, float(value))
    with contextlib.suppress(TypeError, ValueError):
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - time.time())
    return None


def _credential(request: httpx.Request) -> str:
    headers = request.headers
    return headers.get("authorization") or headers.get("x-api-key") or ""


class _ReleasingStream(httpx.AsyncByteStream):
    """A streamed body that returns its permit when closed (or collected unclosed)."""

    def __init__(self, inner: httpx.AsyncByteStream, release: Callable[[], None]):
        self._inner = inner
        self._release = release

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._inner:
            yield chunk

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            self._release()


class GatedAsyncClient(httpx.AsyncClient):
    """An `httpx.AsyncClient` whose model requests pass the request gate (module doc).
    Requests that name no model pass straight through."""

    def __init__(self, gate: RequestGateConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gate_config = gate
        self.gate_dir = gate_dir(gate)

    async def send(
        self, request: httpx.Request, *, stream: bool = False, **kwargs: Any
    ) -> httpx.Response:
        model = request_model(request)
        if model is None:
            return await super().send(request, stream=stream, **kwargs)
        config = self.gate_config
        gate = model_gate(
            self.gate_dir,
            gate_key(request.url, _credential(request), model, config.scope),
            model,
        )
        rejections = 0
        while True:
            permit = reserved_permit.get()
            slot = permit.take(gate) if permit is not None else None
            if slot is None:
                slot = await gate.acquire(config)
            try:
                response = await super().send(request, stream=stream, **kwargs)
            except BaseException:
                gate.release(slot, config, "other")
                raise
            status = response.status_code
            if status in REJECTION_STATUSES and rejections < config.max_rejections:
                hinted = retry_after(response)
                try:
                    await response.aclose()
                finally:  # a cancel or error while closing must not keep the permit
                    gate.release(slot, config, "rejected")
                wait = (
                    min(hinted, config.max_backoff)
                    if hinted is not None
                    else min(config.backoff * 2**rejections, config.max_backoff)
                    * random.uniform(0.5, 1.0)
                )
                rejections += 1
                logger.info(
                    "request gate %s: upstream %d, retry %d/%d in %.1fs",
                    model,
                    status,
                    rejections,
                    config.max_rejections,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            outcome: Outcome = (
                "rejected"
                if status in REJECTION_STATUSES
                else "ok"
                if status < 400
                else "other"
            )
            if not stream:
                gate.release(slot, config, outcome)
                return response
            done = False

            def release_once(
                slot: int = slot, outcome: Outcome = outcome, gate: ModelGate = gate
            ) -> None:
                nonlocal done
                if not done:
                    done = True
                    gate.release(slot, config, outcome)

            assert isinstance(response.stream, httpx.AsyncByteStream)
            response.stream = _ReleasingStream(response.stream, release_once)
            weakref.finalize(response, release_once)
            return response
