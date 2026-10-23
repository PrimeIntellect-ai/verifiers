"""The request gate: a per-model cap on concurrent model requests, shared host-wide.

Providers limit concurrent requests per API key and model (Prime Inference answers
`429 Too many concurrent requests for this model (limit: 32)`). A gate is one such budget:
an (endpoint, API key, model) with a cap, `RequestGateConfig.max_concurrent`.

Every model request a rollout makes, nested agents and judges included, leaves through an
`httpx.AsyncClient` from `clients.base.build_async_httpx`. With `BaseClientConfig.request_gate`
set, that client is a `GatedAsyncClient`, and each request first takes a permit of the gate
for its (endpoint, API key, model). A streamed response holds its permit until it is closed.

Admission is decided by one daemon per gate directory and host (`request_gate_daemon`),
which holds every gate's slots and queue in memory. The first process to use the directory
starts it (under `daemon.lock`); every process keeps one unix-socket connection to it, so
every process on the host (env-server workers, concurrent runs) draws from one budget, and
a process that dies closes its connection, which returns its permits and drops its queued
requests. An admission is one message and one reply; a release is one message, sent
without waiting by a writer thread, so neither ever blocks an event loop, and a release
from a garbage-collector finalizer takes no lock. A daemon that dies is started again by
the next process to reconnect, and every process re-registers the permits it holds and the
requests it has queued. The daemon exits once no process has been connected for a while.

Admission is first come, first served across every process of the host: a request that
cannot take a slot at once joins the daemon's queue, and a freed slot goes to the oldest
waiting request, whichever process sent it, in the same step that frees it.

The cap is shared: the largest `max_concurrent` among the configs admitting through the
gate, so a smaller config for the same model never shrinks everyone's cap by admitting
first. `resize` (or `vf-gate --resize`) moves the cap of live gates and holds it against
every config until the gate's users are gone: a raise takes effect at once, a lower sheds
nothing and admits no one until fewer than the new cap are in flight.

Every process using a gate holds a shared `flock` on its users file; the first one in,
finding no other user, starts the gate's state afresh, so what an earlier run was set to
does not carry over to the next. The daemon writes each gate's state to
`<gate>.state.json` about once a second for dashboards (with requests in flight and waiting
by process); `gate_states` and `resize` ask the daemon.

The gate directory must be the user's own with mode 0700 (it is refused otherwise), its
files are never opened through a symlink, and the daemon serves only its own uid.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import fcntl
import hashlib
import itertools
import json
import logging
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from pydantic_core import from_json

from verifiers.v1.configs.client import RequestGateConfig
from verifiers.v1.request_gate_daemon import check_private, open_private, peer_uid
from verifiers.v1.utils.paths import CACHE_DIR

logger = logging.getLogger(__name__)

DAEMON_SCRIPT = Path(__file__).with_name("request_gate_daemon.py")
START_SECONDS = 30.0
"""Longest wait for a daemon this process started to accept connections."""
REPLY_SECONDS = 60.0
"""Longest wait for the daemon's answer to a synchronous call."""
STATE_BYTES = 1 << 22
"""Most bytes of a gate's state snapshot read back."""


def default_gate_dir() -> Path:
    """`$XDG_RUNTIME_DIR/verifiers-request-gates` (per user, per host), else one per host
    under the user cache, so a home shared across hosts still gives each host its own."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and os.path.isdir(runtime):
        return Path(runtime) / "verifiers-request-gates"
    return CACHE_DIR / "request-gates" / socket.gethostname()


def gate_dir(config: RequestGateConfig) -> Path:
    """The directory of the host's gate daemon `config` uses."""
    return Path(config.dir) if config.dir else default_gate_dir()


def private_dir(directory: Path) -> Path:
    """`directory`, created if missing; refused unless it is the user's own, mode 0700."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    check_private(directory)
    return directory


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).lstrip(".")[:64] or "model"


def _wire(config: RequestGateConfig) -> dict[str, Any]:
    """What the daemon needs of a config (`request_gate_daemon.Cfg`)."""
    return {"max_concurrent": config.max_concurrent}


def _encode(message: list[Any]) -> bytes:
    return json.dumps(message, separators=(",", ":")).encode() + b"\n"


def _address(directory: Path) -> str:
    """The daemon's socket: in the directory or, where that path is too long for a unix
    socket, in a private directory under /tmp, named after the gate directory."""
    path = str(directory / "daemon.sock")
    if len(path.encode()) < 100:
        return path
    digest = hashlib.sha256(str(directory).encode()).hexdigest()[:24]
    sockets = private_dir(Path(f"/tmp/verifiers-request-gates-{os.getuid()}"))
    return str(sockets / f"{digest}.sock")


def _connect(address: str) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(address)
        uid = peer_uid(sock)
    except OSError:
        sock.close()
        raise
    if uid != os.getuid():
        sock.close()
        raise PermissionError(
            f"the request gate socket {address} is served by uid {uid}"
        )
    return sock


def _daemon_alive(directory: Path) -> bool:
    """Whether a daemon serves the directory: it holds `daemon.alive` locked for life."""
    fd = open_private(directory / "daemon.alive", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def _start_daemon(directory: Path, address: str) -> None:
    """Start the directory's daemon, detached (through a shell that exits at once, so it
    is no process's child) and logging to `daemon.log`."""
    log = open_private(directory / "daemon.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    try:
        subprocess.run(
            ["/bin/sh", "-c", '"$@" &', "sh", sys.executable, "-I", "-S"]
            + [str(DAEMON_SCRIPT), str(directory), address],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            start_new_session=True,
            check=True,
        )
    finally:
        os.close(log)


def _open(directory: Path, address: str) -> socket.socket:
    """A connection to the directory's daemon, started if none runs. A daemon that runs
    but refuses (a full listen backlog) is waited for, never doubled."""
    with contextlib.suppress(ConnectionError, FileNotFoundError):
        return _connect(address)
    lock = open_private(directory / "daemon.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        deadline = time.monotonic() + START_SECONDS
        started = False
        while True:
            with contextlib.suppress(ConnectionError, FileNotFoundError):
                return _connect(address)
            if not started and not _daemon_alive(directory):
                _start_daemon(directory, address)
                started = True
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"the request gate daemon of {directory} did not start "
                    f"(see {directory / 'daemon.log'})"
                )
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
                self.gate.release(slot)

    def _deliver(self, slot: int | None) -> None:
        if not self.future.done():
            self.future.set_result(slot)
        elif slot is not None:  # cancelled meanwhile
            self.gate.release(slot)


class _Call:
    """A daemon call awaiting its reply."""

    def __init__(self) -> None:
        self.future: concurrent.futures.Future[Any] = concurrent.futures.Future()


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
        self.outbox: queue.SimpleQueue[bytes] = queue.SimpleQueue()
        self.pending: dict[int, _Admission | _Call] = {}
        self.gates: list[ModelGate] = []
        self.gates_lock = threading.Lock()
        self.ids = itertools.count(1)
        self.closed = False
        self.io = threading.Lock()
        """Held while writing, and while a reconnect swaps the socket and re-registers."""
        self.sock = _open(directory, self.address)
        self.sock.sendall(_encode(["hello", self.pid]))
        for target in (self._reading, self._writing):
            threading.Thread(target=target, name="request-gate", daemon=True).start()

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
            chunks = [self.outbox.get()]
            with contextlib.suppress(queue.Empty):
                while len(chunks) < 4096:
                    chunks.append(self.outbox.get_nowait())
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
            if isinstance(pending, _Admission):
                pending.resolve(message[3])
            else:  # given up meanwhile (a timed-out call): return it
                self.send(["release", message[2], message[3]])
        elif kind == "deny" and isinstance(pending, _Admission):
            pending.resolve(None)
        elif kind == "reply" and isinstance(pending, _Call):
            pending.future.set_result(message[2])

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
                messages: list[list[Any]] = [["hello", self.pid]]
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
                "request gate: reconnected to the daemon of %s (%d gates)",
                self.directory,
                len(gates),
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
    directory's daemon. Thread-safe; usable from any event loop. Construction blocks
    (it may start the daemon): build it off the event loop, as `model_gate` callers do."""

    def __init__(self, directory: Path, key: str, model: str) -> None:
        private_dir(directory)
        self.model = model
        self.name = f"{_slug(model)}-{key}"
        self._held: set[int] = set()
        """Slots this handle holds (re-registered if the daemon restarts)."""
        self._conn = _connection(directory)
        self._users_fd: int | None = open_private(
            directory / f"{self.name}.users", os.O_RDWR | os.O_CREAT
        )
        try:
            fcntl.flock(self._users_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fresh = False  # another live process uses the gate: keep its state
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
        """A slot now, if no request of the host waits and fewer than the cap are held,
        else None. Blocks this thread for one round trip to the daemon."""
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
        """Wait for a slot: first come, first served across the host."""
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
                self.release(slot)
            raise

    def release(self, slot: int) -> None:
        """Return `slot`. Never blocks and takes no lock: safe from a finalizer."""
        try:
            self._held.remove(slot)
        except KeyError:
            return
        self._conn.send(["release", self.name, slot])

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
    if not key.is_dir():
        return None
    check_private(key)
    try:
        sock = _connect(_address(key))
    except (ConnectionError, FileNotFoundError):
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


def gate_states(directory: Path, model: str | None = None) -> list[dict[str, Any]]:
    """The gates under `directory` (only `model`'s, if given): each one's model, cap
    (`max_concurrent`), `in_flight`, `waiting`, counters, and `updated`. With no daemon
    running there (no process uses the directory), the gates of the last run as its
    daemon last wrote them, `updated` telling when."""
    states = _ask(directory, ["states", model])
    if states is not None:
        return states
    states = []
    for path in sorted(Path(directory).glob("*.state.json")):
        try:
            state = json.loads(path.read_bytes()[:STATE_BYTES] or b"{}")
            updated = path.stat().st_mtime
        except (OSError, ValueError):
            continue
        if isinstance(state, dict) and (model is None or state.get("model") == model):
            for key in ("in_flight_by_pid", "waiting_by_pid", "daemon_pid"):
                state.pop(key, None)
            states.append({**state, "updated": updated})
    return states


def resize(
    directory: Path, max_concurrent: int, model: str | None = None
) -> list[dict[str, Any]]:
    """Set the shared cap of the live gates under `directory` (only `model`'s, if given)
    to `max_concurrent`, for every process using them; returns their states as
    `gate_states` does. It holds against every config until the gate's users are gone:
    the next run starts from its own config. With no daemon running there, no gate is
    live: nothing to resize."""
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


def gate_key(url: httpx.URL, credential: str, model: str) -> str:
    """The gate a request belongs to: one per endpoint origin, credential and model, since
    providers limit per key and model. The credential is only hashed."""
    # Anthropic sends the bare key, OpenAI the same key as a Bearer credential: one gate.
    if credential and not any(char.isspace() for char in credential):
        credential = f"Bearer {credential}"
    origin = f"{url.scheme}://{url.host}:{url.port or ''}"
    return hashlib.sha256(f"{origin}\n{credential}\n{model}".encode()).hexdigest()[:16]


def request_model(request: httpx.Request) -> str | None:
    """The model a request asks for: its JSON body's `model`."""
    if request.method != "POST":
        return None
    try:
        body = from_json(request.content)
    except (httpx.RequestNotRead, ValueError):
        return None
    model = body.get("model") if isinstance(body, dict) else None
    return model if isinstance(model, str) else None


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

    async def _gate(self, request: httpx.Request, model: str) -> ModelGate:
        key = gate_key(request.url, _credential(request), model)
        gate = _GATES.get((self.gate_dir, key))
        if gate is None:  # its first use may start the daemon: off the event loop
            gate = await asyncio.to_thread(model_gate, self.gate_dir, key, model)
        return gate

    async def send(
        self, request: httpx.Request, *, stream: bool = False, **kwargs: Any
    ) -> httpx.Response:
        model = request_model(request)
        if model is None:
            return await super().send(request, stream=stream, **kwargs)
        gate = await self._gate(request, model)
        slot = await gate.acquire(self.gate_config)
        try:
            response = await super().send(request, stream=stream, **kwargs)
        except BaseException:
            gate.release(slot)
            raise
        if not stream:
            gate.release(slot)
            return response
        done = False

        def release_once() -> None:
            nonlocal done
            if not done:
                done = True
                gate.release(slot)

        assert isinstance(response.stream, httpx.AsyncByteStream)
        response.stream = _ReleasingStream(response.stream, release_once)
        weakref.finalize(response, release_once)
        return response
