"""The request gates' admission daemon: one process per gate directory and host.

`request_gate` starts it on first use (`python -I -S request_gate_daemon.py DIR SOCKET`).
It holds every gate of the directory in memory (cap, admitted slots, the FIFO queue of
waiting requests) and serves the host's processes over a unix socket, one connection per
process. A process that dies closes its connection, and its slots and queued requests go
with it. The daemon exits once no process has been connected for `IDLE_SECONDS`.

Standard library only, so it starts in tens of milliseconds and imports nothing of the
caller's environment. It serves only its own user: the directory must be the user's own
with mode 0700, and a connection from another uid is closed. The wire protocol is one
JSON array per line:

client -> daemon
    ["hello", pid]
    ["open", rid, gate, model, fresh]       register a gate; `fresh` starts its state anew
                                            (a reply when `rid` is not null)
    ["admit", gate, rid, cfg, mode, since]  mode "try": a slot now or ["deny"]; "wait":
                                            queue, ["grant"] when admitted
    ["cancel", gate, rid]                   leave the queue
    ["release", gate, slot]
    ["hold", gate, slot]                    a slot held across a daemon restart
    ["state", rid, gate]                    -> ["reply", rid, state]
    ["states", rid, model]                  -> ["reply", rid, [summary, ...]]
    ["resize", rid, n, model]               -> ["reply", rid, [summary, ...]]
daemon -> client
    ["grant", rid, gate, slot] / ["deny", rid] / ["reply", rid, value]

`cfg` is the admitting config's `{"max_concurrent": n}`. Admission semantics are
documented in `request_gate`.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import heapq
import json
import os
import re
import signal
import socket
import stat
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

IDLE_SECONDS = 30.0
"""How long the daemon outlives its last connection."""
SNAPSHOT_SECONDS = 1.0
"""Interval at which a changed gate's `<gate>.state.json` is rewritten (for dashboards)."""
REGISTER_SECONDS = 2.0
"""After a restart, how long a gate that still has users admits no one, so every process
can re-register the slots it holds before new ones are handed out."""
MAX_LINE = 1 << 20
"""Longest message accepted; a connection that sends more without a newline is closed."""
WAIT_SMOOTHING = 0.05
"""Weight of the newest admission in the gate's moving average wait."""
GATE_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")
"""What a gate name may be: it names files in the directory."""


def open_private(path: Path, flags: int) -> int:
    """`os.open` for the gate directory's files: owner-only, never through a symlink."""
    return os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)


def check_private(directory: Path) -> None:
    """Refuse a gate directory another user could write to (or that is a symlink)."""
    info = os.lstat(directory)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise PermissionError(
            f"request gate directory {directory} must be a directory of your own with "
            f"mode 0700 (it has uid {info.st_uid}, mode {oct(info.st_mode & 0o7777)})"
        )


def peer_uid(sock: Any) -> int:
    """The uid of a unix socket's peer (Linux, macOS). Elsewhere it cannot be told, and
    the gate fails closed: an OSError."""
    if hasattr(socket, "SO_PEERCRED"):  # Linux: struct ucred {pid, uid, gid}
        raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        return struct.unpack("3i", raw)[1]
    if sys.platform == "darwin":  # LOCAL_PEERCRED: struct xucred {version, uid, ...}
        raw = sock.getsockopt(0, 0x001, 76)  # SOL_LOCAL, LOCAL_PEERCRED
        return struct.unpack_from("2I", raw)[1]
    raise OSError(f"cannot tell a unix socket's peer uid on {sys.platform}")


class Cfg:
    """What the daemon uses of an admitting `RequestGateConfig`."""

    __slots__ = ("cap",)

    def __init__(self, raw: dict[str, Any]) -> None:
        self.cap = int(raw["max_concurrent"])


class Ticket:
    __slots__ = ("cfg", "conn", "dead", "rid", "since")

    def __init__(self, conn: Conn, rid: int, cfg: Cfg, since: float) -> None:
        self.conn, self.rid, self.cfg, self.since = conn, rid, cfg, since
        self.dead = False


class Gate:
    def __init__(self, name: str, grants_from: float = 0.0) -> None:
        self.name = name
        self.model: str | None = None
        self.grants_from = grants_from
        """No admission before this time (`REGISTER_SECONDS` after a restart)."""
        self.reset()

    def reset(self) -> None:
        self.cap = 0
        self.resized = False
        self.admitted = 0
        self.wait_seconds: float | None = None
        self.holders: set[tuple[Conn, int]] = set()
        """(connection, slot) pairs: two processes re-registering one slot id after a
        restart both count."""
        self.used: dict[int, int] = {}
        """Slot id -> how many hold it."""
        self.free: list[int] = []
        """Heap of released slot ids below `next`."""
        self.next = 0
        self.queue: deque[Ticket] = deque()
        self.waiting = 0
        self.changed = True

    def take_cap(self, cfg: Cfg) -> int:
        """The shared cap: the largest admitting config's, unless `resize` set one."""
        if not self.cap or (cfg.cap > self.cap and not self.resized):
            self.cap = cfg.cap
        return self.cap

    def limit(self, cfg: Cfg) -> int:
        """How many may be in flight now."""
        return self.cap

    def open_for_grants(self) -> bool:
        return time.time() >= self.grants_from

    # -- slots -----------------------------------------------------------------------

    def take_slot(self) -> int:
        """The lowest slot id no one holds."""
        while self.free:
            slot = heapq.heappop(self.free)
            if slot < self.next and slot not in self.used:
                return slot
        while self.next in self.used:
            self.next += 1
        self.next += 1
        return self.next - 1

    def hold(self, conn: Conn, slot: int) -> None:
        self.holders.add((conn, slot))
        self.used[slot] = self.used.get(slot, 0) + 1
        conn.holds.add((self.name, slot))
        self.changed = True

    def adopt(self, slot: int, conn: Conn) -> None:
        """A slot a process held before the daemon restarted."""
        if (conn, slot) in self.holders or slot < 0:
            return
        if slot >= self.next:
            for lower in range(self.next, slot):
                heapq.heappush(self.free, lower)
            self.next = slot + 1
        self.hold(conn, slot)

    def grant(self, conn: Conn, rid: int, waited: float) -> None:
        slot = self.take_slot()
        self.hold(conn, slot)
        self.admitted += 1
        average = self.wait_seconds or 0.0
        self.wait_seconds = round(average + WAIT_SMOOTHING * (waited - average), 3)
        conn.send(["grant", rid, self.name, slot])

    def free_slot(self, conn: Conn, slot: int) -> bool:
        if (conn, slot) not in self.holders:
            return False
        self.holders.discard((conn, slot))
        conn.holds.discard((self.name, slot))
        self.used[slot] -= 1
        if not self.used[slot]:
            del self.used[slot]
            heapq.heappush(self.free, slot)
        self.changed = True
        return True

    # -- admission -------------------------------------------------------------------

    def try_admit(self, conn: Conn, rid: int, cfg: Cfg) -> None:
        """A slot now if no request waits and fewer than the limit are held, else a
        denial."""
        self.take_cap(cfg)
        if (
            self.open_for_grants()
            and not self.waiting
            and len(self.holders) < self.limit(cfg)
        ):
            self.grant(conn, rid, 0.0)
        else:
            conn.send(["deny", rid])

    def enqueue(self, conn: Conn, rid: int, cfg: Cfg, since: float | None) -> None:
        if (self.name, rid) in conn.tickets:
            return  # sent again across a reconnect
        self.take_cap(cfg)
        ticket = Ticket(conn, rid, cfg, since if since is not None else time.time())
        conn.tickets[(self.name, rid)] = ticket
        # Re-queued after a daemon restart, a request keeps its age (`since`) but joins
        # the back: each process re-sends its waiters oldest first, so its own order
        # holds; across processes the restart reorders them once.
        self.queue.append(ticket)
        self.waiting += 1
        self.changed = True
        self.pump()

    def cancel(self, conn: Conn, rid: int) -> None:
        ticket = conn.tickets.pop((self.name, rid), None)
        if ticket is not None and not ticket.dead:
            ticket.dead = True
            self.waiting -= 1
            self.changed = True
        self.pump()

    def pump(self) -> None:
        """Admit the oldest waiters while fewer than the limit are held."""
        if not self.open_for_grants():
            return
        queue = self.queue
        now = time.time()
        while queue:
            head = queue[0]
            if head.dead:
                queue.popleft()
                continue
            if len(self.holders) >= self.limit(head.cfg):
                break
            queue.popleft()
            head.dead = True
            self.waiting -= 1
            head.conn.tickets.pop((self.name, head.rid), None)
            self.grant(head.conn, head.rid, max(0.0, now - head.since))

    def release(self, conn: Conn, slot: int) -> None:
        if self.free_slot(conn, slot):
            self.pump()

    def resize(self, cap: int) -> None:
        self.cap = cap
        self.resized = True
        self.changed = True
        self.pump()

    # -- reporting -------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """The gate as `request_gate.gate_states` reports it."""
        return {
            "model": self.model,
            "max_concurrent": self.cap or None,
            "in_flight": len(self.holders),
            "waiting": self.waiting,
            "admitted": self.admitted,
            "wait_seconds": self.wait_seconds or 0.0,
            "resized": self.resized,
            "updated": time.time(),
        }

    def snapshot(self) -> dict[str, Any]:
        """`summary()` plus who holds and who waits, by process, for dashboards."""
        state = self.summary()
        holding: dict[str, int] = {}
        for conn, _ in self.holders:
            holding[str(conn.pid)] = holding.get(str(conn.pid), 0) + 1
        waiting: dict[str, int] = {}
        for ticket in self.queue:
            if not ticket.dead:
                pid = str(ticket.conn.pid)
                waiting[pid] = waiting.get(pid, 0) + 1
        state["in_flight_by_pid"] = holding
        state["waiting_by_pid"] = waiting
        state["daemon_pid"] = os.getpid()
        return state


def log(message: str) -> None:
    print(
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}", file=sys.stderr, flush=True
    )


class Conn(asyncio.Protocol):
    """One process's connection."""

    def __init__(self, daemon: Daemon) -> None:
        self.daemon = daemon
        self.pid = 0
        self.buffer = b""
        self.tickets: dict[tuple[str, int], Ticket] = {}
        self.holds: set[tuple[str, int]] = set()
        self.transport: asyncio.Transport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        assert isinstance(transport, asyncio.Transport)
        try:
            uid: int | None = peer_uid(transport.get_extra_info("socket"))
        except OSError:
            uid = None
        if uid != os.getuid():
            log(f"refused a connection from uid {uid}")
            transport.close()
            return
        self.transport = transport
        self.daemon.conns.add(self)

    def send(self, message: list[Any]) -> None:
        if self.transport is not None and not self.transport.is_closing():
            self.transport.write(
                json.dumps(message, separators=(",", ":")).encode() + b"\n"
            )

    def data_received(self, data: bytes) -> None:
        if self.transport is None:
            return
        self.buffer += data
        if b"\n" not in data:
            if len(self.buffer) > MAX_LINE:
                log(f"pid {self.pid} sent over {MAX_LINE} bytes in one line: closing")
                self.transport.close()
            return
        *lines, self.buffer = self.buffer.split(b"\n")
        for line in lines:
            if not line:
                continue
            try:
                self.daemon.handle(self, json.loads(line))
            except Exception as error:  # noqa: BLE001 - a bad message must not stop it
                log(f"bad message from pid {self.pid}: {line[:200]!r}: {error!r}")

    def connection_lost(self, exc: Exception | None) -> None:
        if self.transport is not None:
            self.transport = None
            self.daemon.lost(self)


def gates_in_use(directory: Path) -> set[str]:
    """The gates a live process still uses: it holds their users `flock`. Read before the
    daemon listens, so only processes of a previous daemon can hold one."""
    names = set()
    for path in directory.glob("*.users"):
        try:
            fd = open_private(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            names.add(path.name.removesuffix(".users"))
        finally:
            os.close(fd)
    return names


class Daemon:
    def __init__(
        self, directory: Path, address: str, alive: int, in_use: set[str]
    ) -> None:
        self.directory, self.address, self.alive = directory, address, alive
        self.gates: dict[str, Gate] = {}
        self.conns: set[Conn] = set()
        self.idle_since = time.monotonic()
        until = time.time() + REGISTER_SECONDS
        self.recovering = dict.fromkeys(in_use, until)
        """Gates whose users outlived the previous daemon: they re-register first."""
        if in_use:
            asyncio.get_running_loop().call_later(REGISTER_SECONDS, self.recovered)

    def recovered(self) -> None:
        """The re-registration window is over: open the gates (not by the clock, which a
        timer may beat by its resolution) and admit their waiters."""
        for name in self.recovering:
            if (gate := self.gates.get(name)) is not None:
                gate.grants_from = 0.0
                gate.pump()
        self.recovering = {}

    def gate(self, name: str) -> Gate:
        gate = self.gates.get(name)
        if gate is None:
            if not GATE_NAME.fullmatch(name):
                raise ValueError(f"not a gate name: {name!r}")
            gate = self.gates[name] = Gate(name, self.recovering.get(name, 0.0))
        return gate

    def handle(self, conn: Conn, message: list[Any]) -> None:
        kind = message[0]
        if kind == "admit":
            _, name, rid, cfg, mode, since = message
            if mode == "try":
                self.gate(name).try_admit(conn, rid, Cfg(cfg))
            else:
                self.gate(name).enqueue(conn, rid, Cfg(cfg), since)
        elif kind == "release":
            if (gate := self.gates.get(message[1])) is not None:
                gate.release(conn, message[2])
        elif kind == "cancel":
            if (gate := self.gates.get(message[1])) is not None:
                gate.cancel(conn, message[2])
        elif kind == "open":
            _, rid, name, model, fresh = message
            gate = self.gate(name)
            if fresh:
                for ticket in gate.queue:
                    ticket.dead = True
                    ticket.conn.tickets.pop((name, ticket.rid), None)
                gate.reset()
            gate.model = model
            gate.changed = True
            if rid is not None:
                conn.send(["reply", rid, None])
        elif kind == "hold":
            self.gate(message[1]).adopt(message[2], conn)
        elif kind == "hello":
            conn.pid = int(message[1])
        elif kind == "state":
            _, rid, name = message
            gate = self.gates.get(name)
            conn.send(["reply", rid, gate.summary() if gate is not None else {}])
        elif kind == "states":
            _, rid, model = message
            conn.send(["reply", rid, self.summaries(model)])
        elif kind == "resize":
            _, rid, cap, model = message
            for gate in self.gates.values():
                if model is None or gate.model == model:
                    gate.resize(int(cap))
            conn.send(["reply", rid, self.summaries(model)])
        else:
            log(f"unknown message from pid {conn.pid}: {message!r}")

    def summaries(self, model: str | None) -> list[dict[str, Any]]:
        return [
            gate.summary()
            for _, gate in sorted(self.gates.items())
            if gate.cap and (model is None or gate.model == model)
        ]

    def lost(self, conn: Conn) -> None:
        """A process went away: its queued requests leave and its slots come back."""
        self.conns.discard(conn)
        if not self.conns:
            self.idle_since = time.monotonic()
        touched: set[str] = set()
        for (name, _), ticket in conn.tickets.items():
            if not ticket.dead:
                ticket.dead = True
                self.gates[name].waiting -= 1
                touched.add(name)
        conn.tickets.clear()
        for name, slot in list(conn.holds):
            gate = self.gates.get(name)
            if gate is not None and gate.free_slot(conn, slot):
                touched.add(name)
        conn.holds.clear()
        for name in touched:
            self.gates[name].changed = True
            self.gates[name].pump()

    async def housekeeping(self, server: asyncio.AbstractServer) -> None:
        while True:
            await asyncio.sleep(SNAPSHOT_SECONDS)
            for gate in list(self.gates.values()):
                if gate.changed and gate.cap:
                    gate.changed = False
                    with contextlib.suppress(OSError):
                        self.write_snapshot(gate)
            idle = time.monotonic() - self.idle_since > IDLE_SECONDS
            if not self.conns and idle and self.retire(server):
                return

    def write_snapshot(self, gate: Gate) -> None:
        path = self.directory / f"{gate.name}.state.json"
        temporary = path.with_name(f".{path.name}.{os.getpid()}")
        fd = open_private(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        with os.fdopen(fd, "w") as file:
            file.write(json.dumps(gate.snapshot(), sort_keys=True))
        os.replace(temporary, path)

    def retire(self, server: asyncio.AbstractServer) -> bool:
        """Stop serving, under the start lock so no process starts a new daemon while
        this one still answers. False if a process connected meanwhile. A directory
        that is gone (a test's, cleaned up) leaves nothing to guard: stop at once."""
        try:
            lock = open_private(self.directory / "daemon.lock", os.O_RDWR | os.O_CREAT)
        except FileNotFoundError:
            server.close()
            return True
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False  # a process is starting a daemon or connecting: stay
            if self.conns:
                return False
            server.close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.address)
            pid_file = self.directory / "daemon.pid"
            with contextlib.suppress(FileNotFoundError):
                if pid_file.read_text().strip() == str(os.getpid()):
                    pid_file.unlink()
            os.close(self.alive)  # a process may start a new daemon from here on
            return True
        finally:
            os.close(lock)


async def serve(directory: Path, address: str) -> None:
    check_private(directory)
    alive = open_private(directory / "daemon.alive", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(alive, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:  # a daemon already serves the directory: leave its socket
        log("another daemon serves this directory: exiting")
        return
    in_use = gates_in_use(directory)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(address)  # a dead daemon's: the alive lock says none serves
    sock.bind(address)
    os.chmod(address, 0o600)
    sock.listen(4096)
    daemon = Daemon(directory, address, alive, in_use)
    server = await asyncio.get_running_loop().create_unix_server(
        lambda: Conn(daemon), sock=sock
    )
    pid = open_private(directory / "daemon.pid", os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with os.fdopen(pid, "w") as file:
        file.write(f"{os.getpid()}\n")
    log(f"serving {directory} at {address!r}")
    await daemon.housekeeping(server)
    log("idle: exiting")


def main() -> None:
    directory, address = Path(sys.argv[1]), sys.argv[2]
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # a Ctrl-C to the starting process
    asyncio.run(serve(directory, address))


if __name__ == "__main__":
    main()
