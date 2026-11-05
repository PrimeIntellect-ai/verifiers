"""The request gate: a per-model cap on concurrent upstream requests, shared host-wide.

Providers limit concurrent requests per API key and model (Prime Inference answers
`429 Too many concurrent requests for this model (limit: 32)`). Every model request a
rollout makes, nested agents and judges included, leaves through an `httpx.AsyncClient`
from `clients.base.build_async_httpx`. With `BaseClientConfig.request_gate` set, that
client is a `GatedAsyncClient`, and each request first takes a permit of the gate for its
(endpoint, API key, model).

Admission is decided by one daemon per gate directory and host (`clients.gate_daemon`),
which holds every gate's slots and queue in memory. The first process to use the directory
starts it (under `daemon.lock`); every process keeps one unix-socket connection to it, so
every process on the host (env-server workers, concurrent runs) draws from one budget, and
a process that dies closes its connection, which returns its permits and drops its queued
requests. An admission is one message and one reply; a release is one message, sent
without waiting, by a writer thread, so neither ever blocks an event loop, and a release
from a garbage-collector finalizer takes no lock. A daemon that dies is started again by
the next process to reconnect, and every process re-registers the permits it holds and the
requests it has queued. The daemon exits once no process has been connected for a while.

Every process using a gate holds a shared `flock` on its users file; the first one in,
finding no other user, starts the gate's state afresh, so what an earlier run learned or
was set to does not carry over to the next. The daemon writes each gate's state to
`<gate>.state.json` about once a second for dashboards (with requests in flight and waiting
by process); `gate_states` and `resize` ask the daemon.

The cap (`max_concurrent` in the state) is shared: the largest `max_concurrent` among the
configs admitting through the gate, so a seat configured with a smaller gate for the same
model never shrinks everyone's cap by admitting first after a restart. `resize` moves the
cap of live gates (an operator, e.g. prime-rl's `eval control resize-requests`) and holds
it against every config until the gate's users are gone: a raise takes effect at once, a
lower sheds nothing and admits no one until fewer than the new cap are in flight. Requests
admitted under a higher cap count until they finish; admission takes the lowest slot no one
holds.

The cap is never exceeded. Under it, with `adaptive`, a 429/529 lowers the limit by
`decrease_factor` (at most once per `decrease_cooldown`), and the limit climbs back to the
cap in a straight line over `recover_seconds` from that decrease, whatever the traffic: a
burst of rejections slows the gate for a minute, never for hours. A raise of the cap
restarts the limit at the new cap; a lower keeps a lowered limit already below it.

Admission is first come, first served across every process of the host: a request that
cannot take a slot at once joins the daemon's queue, and a freed slot goes to the oldest
waiting request, whichever process sent it, in the same step that frees it. The state keeps
the queue's length and an average of the time an admitted request waited (`wait_seconds`).

A rejected request returns its permit, waits for
`Retry-After` (else exponential backoff) and queues again, up to `max_rejections` times;
after that the provider's response reaches the caller unchanged, as final: the
interception server does not retry it.

A streamed response holds its permit until it is closed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import email.utils
import fcntl
import hashlib
import itertools
import json
import logging
import os
import queue
import random
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from pydantic_core import from_json

from verifiers.v1.configs.client import RequestGateConfig

logger = logging.getLogger(__name__)

REJECTION_STATUSES = frozenset({429, 529})
"""Provider statuses that mean "too much load from you": 429 (rate or concurrency limit)
and 529 (Anthropic's overloaded)."""
STATE_BYTES = 1 << 22
"""Most bytes of a gate's state file read back."""
MODEL_EXTENSION = "verifiers.model"
"""`httpx.Request.extensions` key a caller sets to name the request's model, so the gate
need not parse the body (the eval client sets it; SDK requests fall back to parsing)."""
DAEMON_SCRIPT = Path(__file__).with_name("gate_daemon.py")
START_SECONDS = 30.0
"""Longest wait for a daemon this process started to accept connections."""
REPLY_SECONDS = 60.0
"""Longest wait for the daemon's answer to a synchronous call."""

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
    # Anthropic sends the bare key; OpenAI sends the same key as a Bearer
    # credential. Keep existing Bearer gate identities while sharing their cap.
    if credential and not any(char.isspace() for char in credential):
        credential = f"Bearer {credential}"
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


def _wire(config: RequestGateConfig) -> list[Any]:
    """What the daemon needs of a config (`gate_daemon.Cfg`)."""
    return [
        config.max_concurrent,
        config.min_concurrent,
        config.adaptive,
        config.decrease_factor,
        config.decrease_cooldown,
        config.recover_seconds,
    ]


def _encode(message: list[Any]) -> bytes:
    return json.dumps(message, separators=(",", ":")).encode() + b"\n"


def _address(directory: Path) -> str:
    """The daemon's socket: in the directory, or, where that path is too long for a unix
    socket, an abstract one named after it."""
    path = str(directory / "daemon.sock")
    if len(path.encode()) < 100 or not sys.platform.startswith("linux"):
        return path
    digest = hashlib.sha256(str(directory).encode()).hexdigest()[:24]
    return f"\0verifiers-request-gate-{os.getuid()}-{digest}"


def _connect(address: str) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(address)
    except OSError:
        sock.close()
        raise
    return sock


def _start_daemon(directory: Path, address: str) -> None:
    """Start the directory's daemon, detached (through a shell that exits at once, so it
    is no process's child) and logging to `daemon.log`."""
    with open(directory / "daemon.log", "ab") as log:
        subprocess.run(
            [
                "/bin/sh",
                "-c",
                '"$@" &',
                "sh",
                sys.executable,
                "-I",
                "-S",
                str(DAEMON_SCRIPT),
                str(directory),
                address,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            start_new_session=True,
            check=True,
        )


def _open(directory: Path, address: str) -> socket.socket:
    """A connection to the directory's daemon, started if none answers."""
    with contextlib.suppress(OSError):
        return _connect(address)
    lock = os.open(directory / "daemon.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with contextlib.suppress(OSError):
            return _connect(address)  # another process started it meanwhile
        _start_daemon(directory, address)
        deadline = time.monotonic() + START_SECONDS
        while True:
            try:
                return _connect(address)
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"the request gate daemon of {directory} did not start "
                        f"(see {directory / 'daemon.log'})"
                    ) from None
                time.sleep(0.01)
    finally:
        os.close(lock)


class _Admission:
    """A request for a slot awaiting the daemon's answer, delivered to its event loop
    (`loop`) or, without one, to a thread blocked on `future`."""

    def __init__(
        self,
        gate: ModelGate,
        config: RequestGateConfig,
        mode: Literal["try", "wait"],
        loop: asyncio.AbstractEventLoop | None,
    ) -> None:
        self.gate, self.config, self.mode, self.loop = gate, config, mode, loop
        self.since = time.time()
        self.future: asyncio.Future[int | None] | concurrent.futures.Future[int | None]
        if loop is not None:
            self.future = loop.create_future()
        else:
            self.future = concurrent.futures.Future()

    def message(self, rid: int, since: float | None = None) -> list[Any]:
        return ["admit", self.gate.name, rid, _wire(self.config), self.mode, since]

    def resolve(self, slot: int | None) -> None:
        """From the reader thread: the daemon's answer (None: no slot)."""
        if slot is not None:
            self.gate._held.add(slot)
        if self.loop is None:
            assert isinstance(self.future, concurrent.futures.Future)
            self.future.set_result(slot)
            return
        try:
            self.loop.call_soon_threadsafe(self._deliver, slot)
        except RuntimeError:  # its loop closed: no one will take the slot
            if slot is not None:
                self.gate.release(slot, self.config, "other")

    def _deliver(self, slot: int | None) -> None:
        if not self.future.done():
            self.future.set_result(slot)
        elif slot is not None:  # cancelled meanwhile
            self.gate.release(slot, self.config, "other")


class _Call:
    """A daemon call awaiting its reply."""

    def __init__(self) -> None:
        self.future: concurrent.futures.Future[Any] = concurrent.futures.Future()

    def resolve(self, value: Any) -> None:
        self.future.set_result(value)


class _Connection:
    """The process's connection to the gate daemon of a directory, shared by its gates,
    threads and event loops. Messages go out through `outbox` and a writer thread (a
    `SimpleQueue` put is safe from a finalizer); a reader thread delivers the daemon's
    answers and, if the daemon goes away, reconnects (starting it again) and re-registers
    the gates, the permits held and the admissions still waiting."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.address = _address(directory)
        self.pid = os.getpid()
        self.outbox: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self.pending: dict[int, _Admission | _Call] = {}
        self.gates: list[ModelGate] = []
        self.gates_lock = threading.Lock()
        self.ids = itertools.count(1)
        self.closed = False
        self.io = threading.Lock()
        """Held while writing, and while a reconnect swaps the socket and re-registers."""
        self.sock = _open(directory, self.address)
        self.sock.sendall(_encode(["hello", self.pid]))
        threading.Thread(
            target=self._reading, name="request-gate-reader", daemon=True
        ).start()
        threading.Thread(
            target=self._writing, name="request-gate-writer", daemon=True
        ).start()

    def send(self, message: list[Any]) -> None:
        self.outbox.put(_encode(message))

    def admit(self, admission: _Admission) -> int:
        rid = next(self.ids)
        self.pending[rid] = admission
        self.send(admission.message(rid))
        return rid

    def call(self, message: list[Any]) -> Any:
        """Send `[kind, rid, *message[1:]]` and wait for the reply."""
        rid = next(self.ids)
        call = self.pending[rid] = _Call()
        self.send([message[0], rid, *message[1:]])
        try:
            return call.future.result(REPLY_SECONDS)
        except concurrent.futures.TimeoutError:
            self.pending.pop(rid, None)
            raise TimeoutError(
                f"the request gate daemon of {self.directory} did not answer"
            ) from None

    def _writing(self) -> None:
        while True:
            chunk = self.outbox.get()
            if chunk is None:
                return
            chunks = [chunk]
            with contextlib.suppress(queue.Empty):
                while len(chunks) < 4096:
                    chunk = self.outbox.get_nowait()
                    if chunk is None:
                        return
                    chunks.append(chunk)
            with self.io:
                if self.closed:
                    return
                try:
                    self.sock.sendall(b"".join(chunks))
                except OSError:
                    pass  # the reader reconnects and re-registers what these carried

    def _reading(self) -> None:
        buffer = b""
        sock = self.sock
        while True:
            try:
                data = sock.recv(1 << 16)
            except OSError:
                data = b""
            if self.closed:
                return
            if not data:
                sock = self._reconnect()
                buffer = b""
                continue
            buffer += data
            if b"\n" not in data:
                continue
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                try:
                    self._dispatch(json.loads(line))
                except Exception:
                    logger.exception("request gate: bad message from the daemon")

    def _dispatch(self, message: list[Any]) -> None:
        kind, rid = message[0], message[1]
        pending = self.pending.pop(rid, None)
        if kind == "grant":
            slot = message[3]
            if isinstance(pending, _Admission):
                pending.resolve(slot)
            else:  # given up meanwhile (a timed-out call): return it
                self.send(["release", message[2], slot, "other", None])
        elif kind == "deny":
            if isinstance(pending, _Admission):
                pending.resolve(None)
        elif kind == "reply":
            if isinstance(pending, _Call):
                pending.resolve(message[2])

    def _reconnect(self) -> socket.socket:
        """A new connection, the process's gates, permits and waiting admissions
        registered on it. Answers the old daemon owed now never come: calls and `try`
        admissions fail (no slot); waiting admissions queue again by their age."""
        delay = 0.05
        while True:
            try:
                sock = _open(self.directory, self.address)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                logger.warning("request gate: cannot reach the daemon: %s", error)
                time.sleep(delay)
                delay = min(delay * 2, 2.0)
                continue
            with self.io:
                old, self.sock = self.sock, sock
                messages = [["hello", self.pid]]
                with self.gates_lock:
                    gates = list(self.gates)
                for gate in gates:
                    messages.append(["open", None, gate.name, gate.model, False])
                    messages.extend(
                        ["hold", gate.name, slot] for slot in list(gate._held)
                    )
                for rid, pending in list(self.pending.items()):
                    if isinstance(pending, _Admission) and pending.mode == "wait":
                        messages.append(pending.message(rid, pending.since))
                    elif self.pending.pop(rid, None) is pending:
                        if isinstance(pending, _Admission):
                            pending.resolve(None)
                        else:
                            pending.future.set_exception(
                                ConnectionError("the request gate daemon went away")
                            )
                with contextlib.suppress(OSError):
                    sock.sendall(b"".join(_encode(m) for m in messages))
            with contextlib.suppress(OSError):
                old.close()
            logger.warning(
                "request gate: reconnected to the daemon of %s (%d gates, %d waiting)",
                self.directory,
                len(gates),
                sum(isinstance(p, _Admission) for p in self.pending.values()),
            )
            return sock

    def abandon(self) -> None:
        """In a forked child: drop the parent's connection without touching it (the
        parent's threads are not here, and its socket stays open in the parent)."""
        self.closed = True
        with contextlib.suppress(OSError):
            self.sock.close()


_CONNECTIONS: dict[Path, _Connection] = {}
_CONNECTIONS_LOCK = threading.Lock()


def _connection(directory: Path) -> _Connection:
    key = Path(os.path.abspath(directory))
    with _CONNECTIONS_LOCK:
        connection = _CONNECTIONS.get(key)
        if connection is None:
            connection = _CONNECTIONS[key] = _Connection(key)
        return connection


class ModelGate:
    """One (endpoint, key, model) budget: the process's handle on a gate of the
    directory's daemon. Thread-safe; usable from any event loop."""

    def __init__(self, directory: Path, key: str, model: str) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.model = model
        self.name = f"{_slug(model)}-{key}"
        self.prefix = directory / self.name
        self.state_path = self.prefix.with_name(self.prefix.name + ".state.json")
        """The daemon's periodic snapshot of the gate (`state` is current)."""
        self._held: set[int] = set()
        """Slots this handle holds (re-registered if the daemon restarts)."""
        self._conn = _connection(directory)
        self._users_fd: int | None = os.open(
            self.prefix.with_name(self.prefix.name + ".users"),
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        try:
            fcntl.flock(self._users_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fresh = False  # another live process uses the gate: keep what it learned
        else:
            fresh = True
        with self._conn.gates_lock:
            self._conn.gates.append(self)
        if fresh:  # applied before any other process can join (it waits on the lock)
            self._conn.call(["open", self.name, model, True])
        else:
            self._conn.send(["open", None, self.name, model, False])
        fcntl.flock(self._users_fd, fcntl.LOCK_SH)

    def try_admit(self, config: RequestGateConfig) -> int | None:
        """A slot now, if no request of the host waits and fewer than the limit are
        held, else None. Blocks this thread for one round trip to the daemon."""
        admission = _Admission(self, config, "try", None)
        rid = self._conn.admit(admission)
        future = cast("concurrent.futures.Future[int | None]", admission.future)
        try:
            return future.result(REPLY_SECONDS)
        except concurrent.futures.TimeoutError:
            if self._conn.pending.pop(rid, None) is not None:
                return None
            return future.result()  # answered just now

    async def try_acquire(self, config: RequestGateConfig) -> int | None:
        """`try_admit` without blocking the event loop."""
        return await self._admission(config, "try")

    async def acquire(self, config: RequestGateConfig) -> int:
        """Wait for a slot: first come, first served across the host (module doc)."""
        slot = await self._admission(config, "wait")
        assert slot is not None
        return slot

    async def _admission(
        self, config: RequestGateConfig, mode: Literal["try", "wait"]
    ) -> int | None:
        admission = _Admission(self, config, mode, asyncio.get_running_loop())
        rid = self._conn.admit(admission)
        future = cast("asyncio.Future[int | None]", admission.future)
        try:
            return await future
        except BaseException:
            if self._conn.pending.pop(rid, None) is not None:
                if mode == "wait":
                    self._conn.send(["cancel", self.name, rid])
            elif (
                future.done()
                and not future.cancelled()
                and (slot := future.result()) is not None
            ):  # admitted as the waiter was cancelled: give the slot back
                self.release(slot, config, "other")
            raise

    def release(self, slot: int, config: RequestGateConfig, outcome: Outcome) -> None:
        """Return `slot` and feed the call's outcome to the adaptive limit. Never blocks
        and takes no lock: safe from a garbage-collector finalizer."""
        try:
            self._held.remove(slot)
        except KeyError:
            return
        self._conn.send(["release", self.name, slot, outcome, _wire(config)])

    def state(self) -> dict[str, Any]:
        """The gate's current state, as the daemon holds it."""
        return self._conn.call(["state", self.name])

    def close_inherited(self) -> None:
        """Let go of the gate: drop the users lock this handle holds (in a forked child,
        the parent's) and forget its permits without returning them."""
        if self._users_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._users_fd)
            self._users_fd = None
        self._held.clear()
        with contextlib.suppress(ValueError):
            self._conn.gates.remove(self)


def _ask(directory: Path, message: list[Any]) -> Any:
    """A call to the directory's daemon, through this process's connection if it has one
    (so it follows this process's releases), else a short one of its own; None if no
    daemon runs there."""
    key = Path(os.path.abspath(directory))
    connection = _CONNECTIONS.get(key)
    if connection is not None:
        return connection.call(message)
    try:
        sock = _connect(_address(key))
    except OSError:
        return None
    with sock:
        sock.settimeout(REPLY_SECONDS)
        sock.sendall(
            _encode(["hello", os.getpid()]) + _encode([message[0], 0, *message[1:]])
        )
        buffer = b""
        while b"\n" not in buffer:
            data = sock.recv(1 << 16)
            if not data:
                return None
            buffer += data
    return json.loads(buffer.split(b"\n", 1)[0])[2]


def gate_states(directory: Path) -> list[dict[str, Any]]:
    """The gates under `directory`: each one's model, cap (`max_concurrent`), effective
    `limit`, `in_flight`, counters, and `updated`. With no daemon running there (no
    process uses the directory), the gates of the last run as its daemon last wrote them,
    `updated` telling when."""
    states = _ask(directory, ["states", None])
    if states is not None:
        return states
    states = []
    for path in sorted(Path(directory).glob("*.state.json")):
        try:
            raw = path.read_bytes()[:STATE_BYTES]
            updated = path.stat().st_mtime
            state = json.loads(raw) if raw else {}
        except (OSError, ValueError):
            continue
        if isinstance(state, dict):
            states.append(_summary(state, updated))
    return states


def _summary(state: dict[str, Any], updated: float) -> dict[str, Any]:
    cap = state.get("max_concurrent")
    cap = cap if type(cap) is int and cap > 0 else None
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


def resize(
    directory: Path, max_concurrent: int, model: str | None = None
) -> list[dict[str, Any]]:
    """Set the shared cap of the gates under `directory` (only `model`'s, if given) to
    `max_concurrent`, for every process using them; returns their states as
    `gate_states` does. A raise restarts the adaptive limit at the new cap, a lower keeps
    a learned limit already below it (module doc). It holds against every config until
    the gate's users are gone: the next run starts from its own config. With no daemon
    running there, no gate is live: nothing to resize."""
    if max_concurrent < 1:
        raise ValueError(f"a request cap must be at least 1, not {max_concurrent}")
    return _ask(directory, ["resize", max_concurrent, model]) or []


_GATES: dict[tuple[Path, str], ModelGate] = {}
_GATES_LOCK = threading.Lock()


def _reset_after_fork() -> None:
    global _GATES_LOCK, _CONNECTIONS_LOCK
    _GATES_LOCK = threading.Lock()
    _CONNECTIONS_LOCK = threading.Lock()
    for connection in _CONNECTIONS.values():
        connection.abandon()
    _CONNECTIONS.clear()
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
