"""The request gates' admission daemon: one process per gate directory and host.

`clients.gate` starts it on first use (`python -I -S gate_daemon.py DIR SOCKET`); it holds
every gate of the directory in memory (cap, adaptive limit, admitted slots, the FIFO queue
of waiting requests) and serves the host's processes over a unix socket, one connection
per process. A process that dies closes its connection, and its slots and queued requests
go with it. It exits once no process has been connected for `IDLE_SECONDS`.

Standard library only, so it starts in tens of milliseconds and imports nothing of the
caller's environment. The wire protocol is one JSON array per line:

client -> daemon
    ["hello", pid]
    ["open", rid, gate, model, fresh]     register a gate; `fresh` starts its state anew
                                          (a reply when `rid` is not null)
    ["admit", gate, rid, cfg, mode, since] mode "try": a slot now or ["deny"]; "wait":
                                           queue, ["grant"] when admitted
    ["cancel", gate, rid]                 leave the queue
    ["release", gate, slot, outcome, cfg] (`cfg` may be null unless "rejected")
    ["hold", gate, slot]                  a slot held across a daemon restart
    ["state", rid, gate]                  -> ["reply", rid, state]
    ["states", rid, model]                -> ["reply", rid, [summary, ...]]
    ["resize", rid, n, model]             -> ["reply", rid, [summary, ...]]
daemon -> client
    ["grant", rid, gate, slot] / ["deny", rid] / ["reply", rid, value]

`cfg` is `[max_concurrent, min_concurrent, adaptive, decrease_factor, decrease_cooldown,
recover_seconds]`. Admission semantics are documented in `clients.gate`.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import heapq
import json
import os
import signal
import socket
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

IDLE_SECONDS = 30.0
"""How long the daemon outlives its last connection."""
SNAPSHOT_SECONDS = 1.0
"""Interval at which a changed gate's `<gate>.state.json` is rewritten (for dashboards)."""
LOWERED_POLL = 0.05
"""Recheck interval while a lowered limit, climbing back, holds waiters."""
WAIT_SMOOTHING = 0.05


class Cfg:
    __slots__ = ("adaptive", "cap", "cooldown", "factor", "floor", "recover")

    def __init__(self, raw: list[Any]) -> None:
        cap, floor, adaptive, factor, cooldown, recover = raw
        self.cap, self.floor, self.adaptive = int(cap), int(floor), bool(adaptive)
        self.factor, self.cooldown = float(factor), float(cooldown)
        self.recover = float(recover)


class Ticket:
    __slots__ = ("cfg", "conn", "dead", "rid", "since")

    def __init__(self, conn: Conn, rid: int, cfg: Cfg, since: float) -> None:
        self.conn, self.rid, self.cfg, self.since = conn, rid, cfg, since
        self.dead = False


class Gate:
    def __init__(self, daemon: Daemon, name: str) -> None:
        self.daemon, self.name = daemon, name
        self.model: str | None = None
        self.timer: asyncio.TimerHandle | None = None
        self.reset()

    def reset(self) -> None:
        self.cap = 0
        self.resized = False
        self.lowered: float | None = None
        self.decreased_at = 0.0
        self.reported: float | None = None
        """The limit as of the last admission attempt or decrease (the state's `limit`)."""
        self.rejections = self.admitted = 0
        self.wait_seconds: float | None = None
        self.holders: dict[int, Conn] = {}
        self.free: list[int] = []
        """Heap of released slots below `next`."""
        self.next = 0
        self.queue: deque[Ticket] = deque()
        self.waiting = 0
        self.changed = True

    # -- cap and limit ---------------------------------------------------------------

    def take_cap(self, cfg: Cfg) -> int:
        """The shared cap: the largest config's, or what `resize` set."""
        if not self.cap or (cfg.cap > self.cap and not self.resized):
            self.cap = cfg.cap
        return self.cap

    def limit(self, cfg: Cfg, now: float) -> float:
        cap = self.cap
        if not cfg.adaptive or self.lowered is None:
            return float(cap)
        since = now - self.decreased_at
        if since >= cfg.recover:
            self.lowered = None
            return float(cap)
        return clamp(
            self.lowered + (cap - self.lowered) * since / cfg.recover, cfg, cap
        )

    # -- slots -----------------------------------------------------------------------

    def take_slot(self) -> int:
        while self.free:
            slot = heapq.heappop(self.free)
            if slot < self.next and slot not in self.holders:
                return slot
        while self.next in self.holders:
            self.next += 1
        self.next += 1
        return self.next - 1

    def put_slot(self, slot: int) -> None:
        del self.holders[slot]
        heapq.heappush(self.free, slot)

    def adopt(self, slot: int, conn: Conn) -> None:
        if slot in self.holders or slot < 0:
            return
        self.holders[slot] = conn
        conn.holds.add((self.name, slot))
        if slot >= self.next:
            for lower in range(self.next, slot):
                heapq.heappush(self.free, lower)
            self.next = slot + 1
        self.changed = True

    def grant(self, conn: Conn, rid: int, waited: float) -> None:
        slot = self.take_slot()
        self.holders[slot] = conn
        conn.holds.add((self.name, slot))
        self.admitted += 1
        average = self.wait_seconds or 0.0
        self.wait_seconds = round(average + WAIT_SMOOTHING * (waited - average), 3)
        self.changed = True
        conn.send(["grant", rid, self.name, slot])

    # -- admission -------------------------------------------------------------------

    def try_admit(self, conn: Conn, rid: int, cfg: Cfg) -> None:
        """A slot now if no request waits and one is under the limit, else a denial."""
        self.take_cap(cfg)
        limit = self.reported = self.limit(cfg, time.time())
        if not self.waiting and len(self.holders) < int(limit):
            self.grant(conn, rid, 0.0)
        else:
            conn.send(["deny", rid])
        self.changed = True

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
        self.pump()

    def cancel(self, conn: Conn, rid: int) -> None:
        ticket = conn.tickets.pop((self.name, rid), None)
        if ticket is not None and not ticket.dead:
            ticket.dead = True
            self.waiting -= 1
            self.changed = True

    def pump(self) -> None:
        """Admit the oldest waiters while fewer than the limit are held."""
        queue = self.queue
        if not queue:
            return
        now = time.time()
        held = len(self.holders)
        while queue:
            head = queue[0]
            if head.dead:
                queue.popleft()
                continue
            limit = self.reported = self.limit(head.cfg, now)
            if held >= int(limit):
                if self.lowered is not None and held < self.cap and self.timer is None:
                    self.timer = self.daemon.loop.call_later(LOWERED_POLL, self.recheck)
                break
            queue.popleft()
            head.dead = True
            self.waiting -= 1
            head.conn.tickets.pop((self.name, head.rid), None)
            self.grant(head.conn, head.rid, max(0.0, now - head.since))
            held += 1
        self.changed = True

    def recheck(self) -> None:
        self.timer = None
        self.pump()

    def release(self, conn: Conn, slot: int, outcome: str, cfg: Cfg | None) -> None:
        if self.holders.get(slot) is not conn:
            return
        self.put_slot(slot)
        conn.holds.discard((self.name, slot))
        self.changed = True
        if outcome == "rejected" and cfg is not None:
            self.reject(cfg)
        self.pump()

    def reject(self, cfg: Cfg) -> None:
        cap = self.take_cap(cfg)
        limit = self.limit(cfg, time.time())
        now = time.time()
        self.rejections += 1
        if cfg.adaptive and now - self.decreased_at >= cfg.cooldown:
            lowered = clamp(limit * cfg.factor, cfg, cap)
            self.lowered = self.reported = lowered
            self.decreased_at = now
            log(
                f"{self.model}: provider rejected a request, limit {int(limit)} -> {int(lowered)}"
            )

    def resize(self, cap: int) -> None:
        if cap > self.cap:
            self.lowered = self.reported = None
        self.cap = cap
        self.resized = True
        self.changed = True
        self.pump()

    # -- reporting -------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The gate as the old shared state file held it, minus slot and ticket lists."""
        top = max(self.holders, default=-1) + 1
        state: dict[str, Any] = {
            "model": self.model,
            "max_concurrent": self.cap or None,
            "slots": max(self.cap, top),
            "in_flight": len(self.holders),
            "waiting": self.waiting,
            "admitted": self.admitted,
            "rejections": self.rejections,
        }
        if self.reported is not None:
            state["limit"] = self.reported
        if self.lowered is not None:
            state["lowered"] = self.lowered
            state["decreased_at"] = self.decreased_at
        if self.wait_seconds is not None:
            state["wait_seconds"] = self.wait_seconds
        if self.resized:
            state["resized"] = True
        return state

    def summary(self) -> dict[str, Any]:
        cap = self.cap or None
        limit = self.reported
        return {
            "model": self.model,
            "max_concurrent": cap,
            "limit": int(min(limit, cap))
            if cap is not None and limit is not None
            else cap,
            "in_flight": len(self.holders),
            "admitted": self.admitted,
            "rejections": self.rejections,
            "waiting": self.waiting,
            "wait_seconds": self.wait_seconds or 0.0,
            "updated": time.time(),
        }

    def snapshot(self) -> dict[str, Any]:
        """`state()` plus who holds and who waits, by process, for dashboards."""
        state = self.state()
        holding: dict[int, int] = {}
        for conn in self.holders.values():
            holding[conn.pid] = holding.get(conn.pid, 0) + 1
        waiting: dict[int, int] = {}
        for ticket in self.queue:
            if not ticket.dead:
                waiting[ticket.conn.pid] = waiting.get(ticket.conn.pid, 0) + 1
        state["in_flight_by_pid"] = {str(pid): n for pid, n in holding.items()}
        state["waiting_by_pid"] = {str(pid): n for pid, n in waiting.items()}
        state["daemon_pid"] = os.getpid()
        return state


def clamp(limit: float, cfg: Cfg, cap: int) -> float:
    return min(max(float(limit), cfg.floor), cap)


def log(message: str) -> None:
    print(
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}", file=sys.stderr, flush=True
    )


class Conn(asyncio.Protocol):
    def __init__(self, daemon: Daemon) -> None:
        self.daemon = daemon
        self.pid = 0
        self.buffer = b""
        self.tickets: dict[tuple[str, int], Ticket] = {}
        self.holds: set[tuple[str, int]] = set()
        self.transport: asyncio.Transport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        assert isinstance(transport, asyncio.Transport)
        self.transport = transport
        self.daemon.conns.add(self)

    def send(self, message: list[Any]) -> None:
        if self.transport is not None and not self.transport.is_closing():
            self.transport.write(
                json.dumps(message, separators=(",", ":")).encode() + b"\n"
            )

    def data_received(self, data: bytes) -> None:
        self.buffer += data
        if b"\n" not in data:
            return
        *lines, self.buffer = self.buffer.split(b"\n")
        for line in lines:
            if not line:
                continue
            try:
                self.daemon.handle(self, json.loads(line))
            except Exception as error:  # noqa: BLE001 - a bad message must not take the daemon down
                log(f"bad message from pid {self.pid}: {line[:200]!r}: {error!r}")

    def connection_lost(self, exc: Exception | None) -> None:
        self.transport = None
        self.daemon.lost(self)


class Daemon:
    def __init__(self, directory: Path, address: str) -> None:
        self.directory, self.address = directory, address
        self.gates: dict[str, Gate] = {}
        self.conns: set[Conn] = set()
        self.idle_since = time.monotonic()
        self.loop = asyncio.get_running_loop()

    def gate(self, name: str) -> Gate:
        gate = self.gates.get(name)
        if gate is None:
            gate = self.gates[name] = Gate(self, name)
        return gate

    def handle(self, conn: Conn, message: list[Any]) -> None:
        kind = message[0]
        if kind == "admit":
            _, name, rid, cfg, mode, since = message
            gate = self.gate(name)
            if mode == "try":
                gate.try_admit(conn, rid, Cfg(cfg))
            else:
                gate.enqueue(conn, rid, Cfg(cfg), since)
        elif kind == "release":
            _, name, slot, outcome, cfg = message
            gate = self.gates.get(name)
            if gate is not None:
                gate.release(conn, slot, outcome, Cfg(cfg) if cfg else None)
        elif kind == "cancel":
            gate = self.gates.get(message[1])
            if gate is not None:
                gate.cancel(conn, message[2])
                gate.pump()
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
            conn.send(["reply", rid, gate.state() if gate is not None else {}])
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
        for name, slot in conn.holds:
            gate = self.gates.get(name)
            if gate is not None and gate.holders.get(slot) is conn:
                gate.put_slot(slot)
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
        temporary.write_text(json.dumps(gate.snapshot(), sort_keys=True))
        os.replace(temporary, path)

    def retire(self, server: asyncio.AbstractServer) -> bool:
        """Stop serving, under the start lock so no process starts a new daemon while
        this one still answers. False if a process connected meanwhile. A directory
        that is gone (a test's, cleaned up) leaves nothing to guard: stop at once."""
        try:
            lock = os.open(
                self.directory / "daemon.lock", os.O_RDWR | os.O_CREAT, 0o600
            )
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
            if not self.address.startswith("\0"):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(self.address)
            with contextlib.suppress(FileNotFoundError):
                if (self.directory / "daemon.pid").read_text().strip() == str(
                    os.getpid()
                ):
                    (self.directory / "daemon.pid").unlink()
            return True
        finally:
            os.close(lock)


async def serve(directory: Path, address: str) -> None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if not address.startswith("\0"):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(address)
    sock.bind(address)
    if not address.startswith("\0"):
        os.chmod(address, 0o600)
    sock.listen(4096)
    daemon = Daemon(directory, address)
    server = await asyncio.get_running_loop().create_unix_server(
        lambda: Conn(daemon), sock=sock
    )
    (directory / "daemon.pid").write_text(f"{os.getpid()}\n")
    log(f"serving {directory} at {address!r}")
    await daemon.housekeeping(server)
    log("idle: exiting")


def main() -> None:
    directory, address = Path(sys.argv[1]), sys.argv[2]
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # a Ctrl-C to the starting process
    asyncio.run(serve(directory, address))


if __name__ == "__main__":
    main()
