"""The request gate's host daemon: one budget per gate across every process of the host,
first come first served, live resize, recovery from dead processes and a dead daemon, and
a gate directory only its user can touch.

Tests that start child processes bound the whole test, not each step: a child's start-up
can stall on a loaded host. A failing test prints its children's stderr."""

import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from verifiers.v1 import request_gate as gate_module
from verifiers.v1 import request_gate_daemon as daemon_module
from verifiers.v1.configs.client import RequestGateConfig
from verifiers.v1.request_gate import ModelGate
from verifiers.v1.request_gate_daemon import peer_uid

TEST_SECONDS = 120.0
"""Bound of a test that starts child processes."""


@pytest.fixture
def gates(tmp_path: Path) -> Path:
    """A fresh gate directory (mode 0700, as the gate requires)."""
    directory = tmp_path / "gates"
    directory.mkdir(mode=0o700)
    return directory


def admit_all(gate: ModelGate, config: RequestGateConfig) -> list[int]:
    slots = []
    while (slot := gate.try_admit(config)) is not None:
        slots.append(slot)
    return slots


async def until(predicate) -> None:
    """Wait for `predicate()` (run off the loop: it may ask the daemon), bounded by the
    test's own timeout."""
    while not await asyncio.to_thread(predicate):
        await asyncio.sleep(0.02)


@contextlib.asynccontextmanager
async def child(source: str, *args: str) -> AsyncIterator[asyncio.subprocess.Process]:
    """A child process running `source`; killed on exit, its stderr printed if the test
    fails."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        source,
        *args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        yield process
    except BaseException:
        if process.returncode is None:
            process.kill()
        assert process.stderr is not None
        stderr = await process.stderr.read()
        print(f"child {process.pid} stderr:\n{stderr.decode(errors='replace')}")
        raise
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()


@contextlib.contextmanager
def stopped(pid: int) -> Iterator[None]:
    """`pid` stopped (SIGSTOP) until the context exits."""
    os.kill(pid, signal.SIGSTOP)
    try:
        yield
    finally:
        os.kill(pid, signal.SIGCONT)


def daemon_pid(directory: Path) -> int:
    return int((directory / "daemon.pid").read_text())


HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from verifiers.v1.request_gate import ModelGate
    from verifiers.v1.configs.client import RequestGateConfig

    gate = ModelGate(Path(sys.argv[1]), "k", "m")
    config = RequestGateConfig(max_concurrent=int(sys.argv[2]))
    for _ in range(int(sys.argv[3])):
        assert gate.try_admit(config) is not None
    print("held", flush=True)
    time.sleep(600)
    """
)

QUEUER = textwrap.dedent(
    """
    import asyncio, sys
    from pathlib import Path
    from verifiers.v1.request_gate import ModelGate
    from verifiers.v1.configs.client import RequestGateConfig

    async def main(directory, count):
        gate = ModelGate(Path(directory), "k", "m")
        config = RequestGateConfig(max_concurrent=1)
        go = asyncio.Event()

        async def one():
            slot = await gate.acquire(config)
            print("admitted", flush=True)
            await go.wait()  # a line on stdin releases every slot
            gate.release(slot)

        async def reading():
            await asyncio.to_thread(sys.stdin.readline)
            go.set()

        await asyncio.gather(reading(), *(one() for _ in range(count)))
        gate.state()  # every release reached the daemon

    asyncio.run(main(sys.argv[1], int(sys.argv[2])))
    """
)


async def line(process: asyncio.subprocess.Process) -> bytes:
    assert process.stdout is not None
    return await process.stdout.readline()


# -- crashes ----------------------------------------------------------------------------


async def test_a_killed_process_returns_its_permits_to_a_waiter(gates):
    async with asyncio.timeout(TEST_SECONDS):
        config = RequestGateConfig(max_concurrent=1)
        async with child(HOLDER, str(gates), "1", "1") as holder:
            assert await line(holder) == b"held\n"
            gate = await asyncio.to_thread(ModelGate, gates, "k", "m")
            waiting = asyncio.create_task(gate.acquire(config))
            await until(lambda: gate.state()["waiting"] == 1)
            holder.kill()
        assert await waiting == 0, "the dead holder's slot went to the waiter"


async def test_the_oldest_waiting_request_of_the_host_is_admitted_first(gates):
    async with asyncio.timeout(TEST_SECONDS):
        gate = ModelGate(gates, "k", "m")
        config = RequestGateConfig(max_concurrent=1)
        held = gate.try_admit(config)
        assert held == 0
        async with child(QUEUER, str(gates), "1") as queued:  # queues first
            await until(lambda: gate.state()["waiting"] == 1)
            mine = asyncio.create_task(gate.acquire(config))
            await until(lambda: gate.state()["waiting"] == 2)
            gate.release(held)
            assert await line(queued) == b"admitted\n", "the older, other process's"
            await asyncio.sleep(0.2)
            assert not mine.done()
            assert queued.stdin is not None
            queued.stdin.write(b"go\n")
            assert await mine == 0
            assert await queued.wait() == 0


async def test_a_killed_process_leaves_the_queue(gates):
    async with asyncio.timeout(TEST_SECONDS):
        gate = ModelGate(gates, "k", "m")
        config = RequestGateConfig(max_concurrent=1)
        held = gate.try_admit(config)
        assert held == 0
        async with child(QUEUER, str(gates), "3") as queued:
            await until(lambda: gate.state()["waiting"] == 3)
            queued.kill()
            await queued.wait()
            await until(lambda: gate.state()["waiting"] == 0)
        gate.release(held)
        assert await gate.acquire(config) == 0, "no one ahead"


async def test_a_daemon_that_dies_comes_back_with_what_was_held_and_queued(gates):
    async with asyncio.timeout(TEST_SECONDS):
        gate = ModelGate(gates, "k", "m")
        config = RequestGateConfig(max_concurrent=2)
        first, second = gate.try_admit(config), gate.try_admit(config)
        assert (first, second) == (0, 1) and second is not None
        waiting = asyncio.create_task(gate.acquire(config))
        await until(lambda: gate.state()["waiting"] == 1)
        old = daemon_pid(gates)
        os.kill(old, signal.SIGKILL)

        def restarted() -> bool:
            try:
                state = gate.state()
                return daemon_pid(gates) != old and state.get("in_flight") == 2
            except (OSError, ValueError, ConnectionError, TimeoutError):
                return False

        await until(restarted)
        assert gate.state()["waiting"] == 1
        assert not waiting.done(), "the re-registered permits still fill the cap"
        gate.release(second)
        assert await waiting == 1
        assert gate.try_admit(config) is None
        assert gate.state()["in_flight"] == 2


@contextlib.contextmanager
def plain_child(source: str, *args: str) -> Iterator[subprocess.Popen[bytes]]:
    """A child process outside asyncio's child watcher, so it can be stopped (SIGSTOP)
    safely: on Python 3.14 the watcher takes a stopped asyncio child for an exit and
    blocks the event loop in `waitpid`. Killed on exit, its stderr printed if the test
    fails."""
    process = subprocess.Popen(
        [sys.executable, "-c", source, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        yield process
    except BaseException:
        process.kill()
        assert process.stderr is not None
        print(f"child {process.pid} stderr:\n{process.stderr.read().decode()}")
        raise
    finally:
        process.kill()
        process.wait()


async def test_a_restarted_daemon_admits_no_one_until_every_holder_is_back(gates):
    """The daemon dies while one holder is stopped. A new daemon must not hand the cap's
    free slots to the first process back before the stopped one re-registers its own."""
    async with asyncio.timeout(TEST_SECONDS):
        config = RequestGateConfig(max_concurrent=3)
        with plain_child(HOLDER, str(gates), "3", "1") as holder:
            assert holder.stdout is not None
            assert await asyncio.to_thread(holder.stdout.readline) == b"held\n"
            gate = await asyncio.to_thread(ModelGate, gates, "k", "m")
            assert gate.try_admit(config) is not None  # 2 of 3 in flight
            with stopped(holder.pid):
                os.kill(daemon_pid(gates), signal.SIGKILL)
                waiters = [asyncio.create_task(gate.acquire(config)) for _ in range(2)]
                await asyncio.sleep(0.5)  # this process is back; the holder is not

            def full() -> bool:
                try:
                    return gate.state().get("in_flight") == 3
                except (ConnectionError, TimeoutError):  # asked as the daemon died
                    return False

            await until(full)
            await asyncio.sleep(0.3)
            assert gate.state()["in_flight"] == 3, "never over the cap"
            assert sum(waiter.done() for waiter in waiters) == 1
            for waiter in waiters:
                waiter.cancel()


async def test_thousands_of_waiters_drain_under_the_cap(gates):
    gate = ModelGate(gates, "k", "m")
    config = RequestGateConfig(max_concurrent=20)
    held = peak = 0

    async def one() -> None:
        nonlocal held, peak
        slot = await gate.acquire(config)
        held += 1
        peak = max(peak, held)
        await asyncio.sleep(0.001)
        held -= 1
        gate.release(slot)

    started = time.monotonic()
    await asyncio.wait_for(asyncio.gather(*(one() for _ in range(4000))), 60)
    assert peak == 20
    assert gate.state()["admitted"] == 4000
    assert time.monotonic() - started < 30


WORKER = textwrap.dedent(
    """
    import asyncio, sys, time
    from pathlib import Path
    from verifiers.v1.request_gate import ModelGate
    from verifiers.v1.configs.client import RequestGateConfig

    async def main(directory, waiters, rounds):
        gate = ModelGate(Path(directory), "k", "m")
        config = RequestGateConfig(max_concurrent=5)
        spans = []

        async def one():
            for _ in range(rounds):
                slot = await gate.acquire(config)
                start = time.monotonic()
                await asyncio.sleep(0.002)
                spans.append((start, time.monotonic()))
                gate.release(slot)

        await asyncio.gather(*(one() for _ in range(waiters)))
        print("\\n".join(f"{a} {b}" for a, b in spans), flush=True)

    asyncio.run(main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
    """
)


async def test_many_processes_under_load_never_exceed_the_cap(gates):
    """8 processes x 20 waiters x 5 rounds against a cap of 5: every request is admitted,
    and the spans they held (on the host's monotonic clock) never overlap more than 5."""
    async with asyncio.timeout(TEST_SECONDS), contextlib.AsyncExitStack() as stack:
        workers = [
            await stack.enter_async_context(child(WORKER, str(gates), "20", "5"))
            for _ in range(8)
        ]
        outputs = await asyncio.gather(*(worker.communicate() for worker in workers))
        assert [worker.returncode for worker in workers] == [0] * 8
    events = []
    for out, _ in outputs:
        for span in out.decode().splitlines():
            start, end = map(float, span.split())
            events += [(start, 1), (end, -1)]
    assert len(events) == 2 * 8 * 20 * 5
    held = peak = 0
    for _, step in sorted(events):
        held += step
        peak = max(peak, held)
    assert peak == 5


# -- the cap and resize -----------------------------------------------------------------


def test_an_operator_raises_the_cap_of_a_live_gate_for_every_user(gates):
    config = RequestGateConfig(max_concurrent=2)
    first, second = ModelGate(gates, "k", "m"), ModelGate(gates, "k", "m")
    assert len(admit_all(first, config) + admit_all(second, config)) == 2
    (raised,) = gate_module.resize(gates, 5)
    assert raised["max_concurrent"] == 5
    taken = admit_all(second, config) + admit_all(first, config)
    assert sorted(taken) == [2, 3, 4], "the raise reached both, under one shared cap"
    (status,) = gate_module.gate_states(gates)
    assert (status["model"], status["max_concurrent"]) == ("m", 5)
    assert (status["in_flight"], status["resized"]) == (5, True)


def test_a_lowered_cap_sheds_nothing_and_admits_once_below_it(gates):
    config = RequestGateConfig(max_concurrent=4)
    first, second = ModelGate(gates, "k", "m"), ModelGate(gates, "k", "m")
    low = [first.try_admit(config), first.try_admit(config)]
    high = admit_all(second, config)
    assert (low, high) == ([0, 1], [2, 3])
    gate_module.resize(gates, 2)
    assert first.try_admit(config) is None and second.try_admit(config) is None
    for slot in low:
        assert slot is not None
        first.release(slot)
    assert first.try_admit(config) is None, "the two left above the cap still count"
    second.release(high[0])
    assert first.try_admit(config) == 0, "one in flight: below the cap of 2"
    assert second.try_admit(config) is None
    second.release(high[1])
    assert second.try_admit(config) == 1


async def test_a_resize_admits_waiters_of_other_processes(gates):
    async with asyncio.timeout(TEST_SECONDS):
        gate = ModelGate(gates, "k", "m")
        config = RequestGateConfig(max_concurrent=1)
        assert gate.try_admit(config) == 0
        async with child(QUEUER, str(gates), "2") as queued:
            await until(lambda: gate.state()["waiting"] == 2)
            (resized,) = await asyncio.to_thread(gate_module.resize, gates, 3)
            assert resized["max_concurrent"] == 3
            assert [await line(queued) for _ in range(2)] == [b"admitted\n"] * 2
            assert gate.state()["in_flight"] == 3
            assert queued.stdin is not None
            queued.stdin.write(b"go\n")
            assert await queued.wait() == 0


def test_the_shared_cap_is_the_largest_config_whichever_admits_first(gates):
    small, large = ModelGate(gates, "k", "m"), ModelGate(gates, "k", "m")
    assert len(admit_all(small, RequestGateConfig(max_concurrent=2))) == 2
    assert len(admit_all(large, RequestGateConfig(max_concurrent=6))) == 4
    assert small.try_admit(RequestGateConfig(max_concurrent=2)) is None, "at the cap"
    (status,) = gate_module.gate_states(gates)
    assert status["max_concurrent"] == 6
    other, later = ModelGate(gates, "k2", "m2"), ModelGate(gates, "k2", "m2")
    assert len(admit_all(other, RequestGateConfig(max_concurrent=6))) == 6
    other.release(0)
    assert later.try_admit(RequestGateConfig(max_concurrent=2)) == 0
    caps = {s["model"]: s["max_concurrent"] for s in gate_module.gate_states(gates)}
    assert caps["m2"] == 6, "a small config admitting after a large one keeps its cap"


def test_a_resize_lasts_only_while_a_process_uses_the_gate(gates):
    first = ModelGate(gates, "k", "m")
    assert len(admit_all(first, RequestGateConfig(max_concurrent=2))) == 2
    gate_module.resize(gates, 3)
    joining = ModelGate(gates, "k", "m")
    assert admit_all(joining, RequestGateConfig(max_concurrent=8)) == [2]
    first.close_inherited()  # both runs end: their descriptors close
    joining.close_inherited()
    fresh = ModelGate(gates, "k", "m")  # no user left: the operator's cap is gone
    assert len(admit_all(fresh, RequestGateConfig(max_concurrent=6))) == 6


def test_resize_picks_gates_by_model_and_refuses_a_zero_cap(gates, tmp_path):
    config = RequestGateConfig(max_concurrent=4)
    for model in ("a", "b"):
        ModelGate(gates, model, model).try_admit(config)
    (resized,) = gate_module.resize(gates, 9, model="b")
    assert resized["model"] == "b"
    caps = {s["model"]: s["max_concurrent"] for s in gate_module.gate_states(gates)}
    assert caps == {"a": 4, "b": 9}
    assert gate_module.resize(tmp_path / "none", 9) == [], "no daemon: no live gate"
    with pytest.raises(ValueError):
        gate_module.resize(gates, 0)


def test_with_no_daemon_the_last_snapshot_is_reported(gates):
    (gates / "m-k.state.json").write_text(
        '{"model": "m", "max_concurrent": 4, "in_flight": 0, "daemon_pid": 1}'
    )
    (status,) = gate_module.gate_states(gates)
    assert (status["model"], status["max_concurrent"]) == ("m", 4)
    assert "daemon_pid" not in status and status["updated"] > 0


# -- the daemon itself ------------------------------------------------------------------


def test_a_daemon_that_cannot_start_fails_clearly(gates, monkeypatch):
    monkeypatch.setattr(gate_module, "DAEMON_SCRIPT", gates / "missing.py")
    monkeypatch.setattr(gate_module, "START_SECONDS", 0.5)
    with pytest.raises(RuntimeError, match="did not start.*daemon.log"):
        ModelGate(gates, "k", "m")


def test_a_second_daemon_leaves_the_running_one_alone(gates):
    gate = ModelGate(gates, "k", "m")
    config = RequestGateConfig(max_concurrent=1)
    assert gate.try_admit(config) == 0
    address = gate_module._address(gates)
    second = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(gate_module.DAEMON_SCRIPT),
            str(gates),
            address,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert second.returncode == 0 and "another daemon" in second.stderr
    assert gate.try_admit(config) is None, "the first daemon still holds the slot"
    assert gate.state()["in_flight"] == 1


def raw_connection(directory: Path) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(gate_module._address(directory))
    sock.settimeout(10)
    return sock


def test_the_daemon_refuses_gate_names_that_are_not_file_names(gates, tmp_path):
    ModelGate(gates, "k", "m").try_admit(RequestGateConfig(max_concurrent=1))
    with raw_connection(gates) as sock:
        cfg = {"max_concurrent": 1}
        bad = ["admit", "../escaped", 1, cfg, "try", None]
        sock.sendall(json.dumps(bad).encode() + b"\n")
        sock.sendall(json.dumps(["states", 2, None]).encode() + b"\n")
        reply = json.loads(sock.makefile().readline())
    assert reply[1] == 2 and [s["model"] for s in reply[2]] == ["m"]
    time.sleep(1.2)  # a snapshot pass
    assert not list(tmp_path.glob("escaped*"))


def test_the_daemon_closes_a_connection_that_floods_one_line(gates):
    ModelGate(gates, "k", "m")
    with raw_connection(gates) as sock:
        with contextlib.suppress(OSError):  # the daemon may close mid-send
            sock.sendall(b"x" * (2 << 20))
        assert sock.recv(1) == b"", "closed"


def test_the_gate_directory_must_be_private(tmp_path):
    loose = tmp_path / "loose"
    loose.mkdir(mode=0o755)
    loose.chmod(0o755)
    with pytest.raises(PermissionError, match="mode 0700"):
        ModelGate(loose, "k", "m")
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(PermissionError):
        ModelGate(link, "k", "m")


def test_peer_uid_is_the_connecting_user():
    left, right = socket.socketpair(socket.AF_UNIX)
    with left, right:
        assert peer_uid(left) == os.getuid()


def test_a_forked_child_drops_the_parents_permits(gates):
    config = RequestGateConfig(max_concurrent=2)
    parent = gate_module.model_gate(gates, "fork", "m")
    held = parent.try_admit(config)
    assert held is not None
    pid = os.fork()
    if pid == 0:  # the child: report through the exit code, never return into pytest
        ok = False
        try:
            child_gate = gate_module.model_gate(gates, "fork", "m")
            ok = (
                child_gate is not parent
                and not child_gate._held
                and child_gate.try_admit(config)
                == 1  # the parent's permit still counts
                and child_gate.try_admit(config) is None
            )
        finally:
            os._exit(0 if ok else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    parent.release(held)
    assert len(admit_all(parent, config)) == 2, "the dead child took nothing with it"


async def test_a_request_waiting_on_a_busy_daemon_never_stalls_its_event_loop(gates):
    async with asyncio.timeout(TEST_SECONDS):
        gate = ModelGate(gates, "k", "m")
        config = RequestGateConfig(max_concurrent=1)
        ticks = 0

        async def ticking():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(ticking())
        with stopped(daemon_pid(gates)):
            admission = asyncio.create_task(gate.acquire(config))
            await asyncio.sleep(0.5)
            assert not admission.done() and ticks >= 20  # the loop ran meanwhile
        slot = await admission
        ticker.cancel()
        gate.release(slot)  # never waits on the daemon
        await until(lambda: gate.try_admit(config) is not None)


async def test_a_cancelled_admission_returns_the_slot_the_daemon_gave_it(gates):
    async with asyncio.timeout(TEST_SECONDS):
        gate = ModelGate(gates, "k", "m")
        config = RequestGateConfig(max_concurrent=1)
        with stopped(daemon_pid(gates)):
            admission = asyncio.create_task(gate.acquire(config))
            await asyncio.sleep(0.2)
            admission.cancel()
            with pytest.raises(asyncio.CancelledError):
                await admission
        # the daemon admits the request before it reads the cancel: the slot comes back
        await until(lambda: gate.state()["in_flight"] == 0 and not gate._held)
        assert gate.try_admit(config) == 0


# -- hardening ----------------------------------------------------------------------------


def test_a_gate_directory_owned_by_another_user_is_refused(gates, monkeypatch):
    real = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real + 1)
    with pytest.raises(PermissionError, match="your own"):
        gate_module.private_dir(gates)


def test_a_planted_symlink_is_never_followed(gates, tmp_path):
    target = tmp_path / "target"
    target.write_text("untouched")
    (gates / "daemon.log").symlink_to(target)
    with pytest.raises(OSError):
        ModelGate(gates, "k", "m")  # starting the daemon opens daemon.log
    assert target.read_text() == "untouched"


def test_a_client_refuses_a_daemon_of_another_user(gates, monkeypatch):
    real = os.getuid()
    monkeypatch.setattr(gate_module, "peer_uid", lambda sock: real + 1)
    with pytest.raises(PermissionError, match="served by uid"):
        ModelGate(gates, "k", "m")


class FakeTransport(asyncio.Transport):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def get_extra_info(self, name, default=None):
        return None

    def close(self) -> None:
        self.closed = True

    def is_closing(self) -> bool:
        return self.closed


@pytest.mark.parametrize("peer", ["other", "unknown"])
def test_the_daemon_refuses_a_connection_of_another_or_unknown_user(peer, monkeypatch):
    def peer_uid_of(sock):
        if peer == "unknown":
            raise OSError("no peer credentials here")
        return os.getuid() + 1

    monkeypatch.setattr(daemon_module, "peer_uid", peer_uid_of)
    daemon = SimpleNamespace(conns=set())
    conn = daemon_module.Conn(daemon)  # type: ignore[arg-type]
    transport = FakeTransport()
    conn.connection_made(transport)
    assert transport.closed and not daemon.conns and conn.transport is None
