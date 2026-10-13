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
around each admission, holds the adaptive limit and running counters as JSON (readable
while a run is live). The limit is AIMD: a 429/529 multiplies it by `decrease_factor` (at
most once per `decrease_cooldown`), each success adds `1/limit`, bounded by
`[min_concurrent, max_concurrent]`. A rejected request returns its permit, waits for
`Retry-After` (else exponential backoff) and queues again, up to `max_rejections` times;
after that the provider's response reaches the caller unchanged.

A streamed response holds its permit until it is closed. `observe_throttle` collects, for
the calls made in its context, what the gate did; the interception server records it on
each `ModelCall` (`RequestThrottle`).
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
from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel
from pydantic_core import from_json

from verifiers.v1.configs.client import RequestGateConfig

logger = logging.getLogger(__name__)

REJECTION_STATUSES = frozenset({429, 529})
"""Provider statuses that mean "too much load from you": 429 (rate or concurrency limit)
and 529 (Anthropic's overloaded)."""
POLL_SECONDS = (0.05, 0.2)
"""Jittered interval at which a blocked admission rechecks the shared slots."""
MODEL_EXTENSION = "verifiers.model"
"""`httpx.Request.extensions` key a caller sets to name the request's model, so the gate
need not parse the body (the eval client sets it; SDK requests fall back to parsing)."""

Outcome = Literal["ok", "rejected", "other"]


class RequestThrottle(BaseModel):
    """What a request gate (`BaseClientConfig.request_gate`) did to one model call: the
    gate fills it in for the requests of an `observe_throttle` context, and the
    interception server records it on the call (`ModelCall.throttle`)."""

    queued: float = 0.0
    """Seconds spent waiting for a permit, rejection waits included."""
    rejections: int = 0
    """Provider rejections (429/529) the gate absorbed and retried for this call."""
    retry_after: float = 0.0
    """Seconds of those waits that the provider asked for (`Retry-After`)."""
    limit: int | None = None
    """The gate's concurrency limit when the call was last admitted."""


_OBSERVATION: ContextVar[RequestThrottle | None] = ContextVar(
    "request_throttle", default=None
)


@contextlib.contextmanager
def observe_throttle() -> Iterator[RequestThrottle]:
    """Collect into the yielded `RequestThrottle` what the gate does to the upstream requests
    made in this context. `limit` stays None when no gate touched them."""
    throttle = RequestThrottle()
    token = _OBSERVATION.set(throttle)
    try:
        yield throttle
    finally:
        _OBSERVATION.reset(token)


def default_gate_dir() -> Path:
    return Path(tempfile.gettempdir()) / f"verifiers-request-gates-{os.getuid()}"


def gate_key(url: httpx.URL, credential: str, model: str) -> str:
    """The gate a request belongs to: one per endpoint origin, credential and model, since
    providers limit per key and model. The credential is only hashed."""
    origin = f"{url.scheme}://{url.host}:{url.port or ''}"
    return hashlib.sha256(f"{origin}\n{credential}\n{model}".encode()).hexdigest()[:16]


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)[:64]


class ModelGate:
    """One (endpoint, key, model) budget: process-local view of the shared slot files.
    Thread-safe; admission is FIFO among the waiters of one event loop."""

    def __init__(self, directory: Path, key: str, model: str) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.model = model
        self.prefix = directory / f"{_slug(model)}-{key}"
        self.state_path = self.prefix.with_name(self.prefix.name + ".state.json")
        self._mutex = threading.Lock()
        self._fds: dict[int, int] = {}
        self._held: set[int] = set()
        self._state_fd: int | None = None
        self._loop_locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Lock
        ] = weakref.WeakKeyDictionary()

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
        fd = self._state_fd
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            raw = os.pread(fd, 1 << 16, 0)
            try:
                state = json.loads(raw) if raw else {}
            except ValueError:
                state = {}
            if not isinstance(state, dict):
                state = {}
            yield state
            data = json.dumps(state, sort_keys=True).encode()
            os.ftruncate(fd, 0)
            os.pwrite(fd, data, 0)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)

    @staticmethod
    def _limit(state: dict[str, Any], config: RequestGateConfig) -> float:
        limit = state.get("limit")
        if not isinstance(limit, int | float) or not config.adaptive:
            limit = config.max_concurrent
        return min(max(float(limit), config.min_concurrent), config.max_concurrent)

    def try_admit(self, config: RequestGateConfig) -> tuple[int | None, int]:
        """Take a free slot if fewer than the limit are held host-wide: (slot or None,
        the integer limit it was checked against)."""
        with self._mutex, self._state() as state:
            limit = int(self._limit(state, config))
            held, free = 0, None
            for slot in range(config.max_concurrent):
                if slot in self._held:
                    held += 1
                    continue
                fd = self._slot_fd(slot)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    held += 1
                    continue
                if free is None:
                    free = slot
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            if free is not None and held >= limit:
                fcntl.flock(self._fds[free], fcntl.LOCK_UN)
                free = None
            if free is not None:
                self._held.add(free)
                state["admitted"] = state.get("admitted", 0) + 1
            state.update(model=self.model, cap=config.max_concurrent)
            state["limit"] = self._limit(state, config)
            state["in_flight"] = held + (free is not None)
            return free, limit

    def release(self, slot: int, config: RequestGateConfig, outcome: Outcome) -> None:
        """Return `slot` and feed the call's outcome to the adaptive limit."""
        with self._mutex:
            if slot not in self._held:
                return
            self._held.discard(slot)
            fcntl.flock(self._fds[slot], fcntl.LOCK_UN)
            if outcome == "other":
                return
            with self._state() as state:
                limit = self._limit(state, config)
                now = time.time()
                if outcome == "ok":
                    state["limit"] = min(config.max_concurrent, limit + 1 / limit)
                    return
                state["rejections"] = state.get("rejections", 0) + 1
                if (
                    config.adaptive
                    and now - state.get("decreased_at", 0.0) >= config.decrease_cooldown
                ):
                    lowered = max(config.min_concurrent, limit * config.decrease_factor)
                    state.update(limit=lowered, decreased_at=now)
                    logger.warning(
                        "request gate %s: provider rejected a request, limit %d -> %d",
                        self.model,
                        int(limit),
                        int(lowered),
                    )

    async def acquire(
        self, config: RequestGateConfig, throttle: RequestThrottle
    ) -> int:
        """Wait for a slot; adds the wait to `throttle.queued` and sets `throttle.limit`."""
        started = time.monotonic()
        loop = asyncio.get_running_loop()
        lock = self._loop_locks.get(loop)
        if lock is None:
            lock = self._loop_locks[loop] = asyncio.Lock()
        async with lock:
            while True:
                slot, limit = self.try_admit(config)
                if slot is not None:
                    break
                await asyncio.sleep(random.uniform(*POLL_SECONDS))
        throttle.queued += time.monotonic() - started
        throttle.limit = limit
        return slot

    def close_inherited(self) -> None:
        """Drop the descriptors a forked child inherited: their locks are the parent's."""
        for fd in [*self._fds.values(), self._state_fd]:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
        self._fds.clear()
        self._held.clear()
        self._state_fd = None
        self._mutex = threading.Lock()


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
        self.gate_dir = Path(gate.dir) if gate.dir else default_gate_dir()

    async def send(
        self, request: httpx.Request, *, stream: bool = False, **kwargs: Any
    ) -> httpx.Response:
        model = request_model(request)
        if model is None:
            return await super().send(request, stream=stream, **kwargs)
        config = self.gate_config
        gate = model_gate(
            self.gate_dir, gate_key(request.url, _credential(request), model), model
        )
        throttle = _OBSERVATION.get() or RequestThrottle()
        rejections = 0
        while True:
            slot = await gate.acquire(config, throttle)
            try:
                response = await super().send(request, stream=stream, **kwargs)
            except BaseException:
                gate.release(slot, config, "other")
                raise
            status = response.status_code
            if status in REJECTION_STATUSES and rejections < config.max_rejections:
                hinted = retry_after(response)
                await response.aclose()
                gate.release(slot, config, "rejected")
                wait = (
                    min(hinted, config.max_backoff)
                    if hinted is not None
                    else min(config.backoff * 2**rejections, config.max_backoff)
                    * random.uniform(0.5, 1.0)
                )
                rejections += 1
                throttle.rejections += 1
                throttle.retry_after += min(hinted or 0.0, config.max_backoff)
                logger.info(
                    "request gate %s: upstream %d, retry %d/%d in %.1fs",
                    model,
                    status,
                    rejections,
                    config.max_rejections,
                    wait,
                )
                await asyncio.sleep(wait)
                throttle.queued += wait
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
