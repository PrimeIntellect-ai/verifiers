"""Process lifecycle utilities."""

import contextlib
import logging
import os
import signal
import subprocess
import threading
import time
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

import setproctitle

logger = logging.getLogger(__name__)

VERIFIERS_PROC_PREFIX = "Verifiers"


def set_proc_title(name: str) -> None:
    """Set the OS-visible process title."""
    setproctitle.setproctitle(f"{VERIFIERS_PROC_PREFIX}::{name}")


def monitor_death_pipe(death_pipe: Connection) -> None:
    """Monitor a death pipe and send SIGTERM to this process when it closes.

    The parent creates a pipe and keeps the writer end open.  When the parent
    dies (even via SIGKILL), the OS closes the writer and ``reader.recv()``
    raises ``EOFError``.  This function sends SIGTERM to the current process
    so existing signal handlers can perform a clean shutdown.

    Starts a daemon thread so the caller is not blocked.
    """

    def monitor_death_pipe_thread() -> None:
        try:
            death_pipe.recv()  # blocks until writer closes
        except (EOFError, OSError):
            pass
        logger.info("Death pipe closed — parent is gone, sending SIGTERM to self")
        os.kill(os.getpid(), signal.SIGTERM)

    t = threading.Thread(
        target=monitor_death_pipe_thread, name="death-pipe-monitor", daemon=True
    )
    t.start()


def terminate_process(
    process: BaseProcess | None,
    timeout: float = 10.0,
    kill_timeout: float = 10.0,
) -> None:
    """Gracefully terminate a process, escalating to kill if needed.

    Idempotent — safe to call on None, already-exited, or already-joined
    processes.  Works with both ``mp.Process`` and ``mp.get_context("spawn").Process``.
    """
    if process is None or not process.is_alive():
        return
    process.terminate()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join(timeout=kill_timeout)


def terminate_processes(
    processes: list[BaseProcess],
    timeout: float = 10.0,
    kill_timeout: float = 10.0,
) -> None:
    """Terminate multiple processes in parallel.

    Runs :func:`terminate_process` for each process in its own thread so the
    total wait is bounded by a single timeout window, not N × timeout.
    """
    threads = [
        threading.Thread(target=terminate_process, args=(p, timeout, kill_timeout))
        for p in processes
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _proc_session_processes(session: int) -> dict[int, bool] | None:
    """Read session membership from procfs, or return None when unavailable."""
    try:
        entries = os.scandir("/proc")
    except OSError:
        return None

    members: dict[int, bool] = {}
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/stat") as stat:
                    fields = stat.read().rsplit(")", 1)[1].split()
                if int(fields[3]) == session:
                    members[int(entry.name)] = fields[0] != "Z"
            except (IndexError, OSError, ValueError):
                # Processes can exit between scandir and opening their stat file. A live
                # member is seen on the next fixed-point scan below.
                continue
    return members


def _session_processes(session: int) -> dict[int, bool] | None:
    """Map current POSIX session PIDs to whether each process is non-zombie."""
    if (members := _proc_session_processes(session)) is not None:
        return members

    try:
        listing = subprocess.run(
            ["ps", "-A", "-o", "pid=,sid=,stat="],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        logger.warning("Could not read POSIX session %d", session)
        return None

    members = {}
    for line in listing.splitlines():
        try:
            raw_pid, raw_sid, state = line.split()
            pid, sid = int(raw_pid), int(raw_sid)
        except ValueError:
            continue
        if sid == session:
            members[pid] = not state.startswith("Z")
    return members


def kill_process_session(process: BaseProcess, timeout: float = 10.0) -> None:
    """Kill a worker-owned POSIX session, including its separate process groups.

    Pool workers are session leaders, and framework subprocess runtimes create their own
    process groups inside that session.  Session membership therefore survives worker
    exit and does not rely on a stale parent-PID snapshot.  Members are stopped until the
    session is stable, preventing new forks between discovery and ``SIGKILL``.

    Call this only after giving the process a chance to terminate gracefully.  It is a
    POSIX helper; if the process did not become its own session leader, only its PID is
    killed so an unrelated caller session is never targeted.
    """
    root = process.pid
    assert root is not None
    # The caller waits on the process sentinel without joining first.  An exited worker
    # therefore remains a zombie and keeps its PID/SID reserved until this cleanup joins it.
    members = _session_processes(root)
    if members is None:
        logger.error(
            "Cannot discover worker session %d; killing only the worker process", root
        )
        with contextlib.suppress(Exception):
            process.kill()
        process.join(timeout=timeout)
        return
    if root not in members:
        with contextlib.suppress(Exception):
            process.kill()
        process.join(timeout=timeout)
        return

    live = {pid for pid, is_live in members.items() if is_live}
    discovery_failures = 0
    deadline = time.monotonic() + timeout
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        try:
            while live and time.monotonic() < deadline:
                for pid in live:
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.kill(pid, signal.SIGSTOP)
                discovered = _session_processes(root)
                if discovered is None:
                    discovery_failures += 1
                    if discovery_failures < 3:
                        continue
                    logger.error("Could not verify stopped worker session %d", root)
                    break
                discovery_failures = 0
                current = {pid for pid, is_live in discovered.items() if is_live}
                if current <= live:
                    live = current
                    break
                live = current
            else:
                if live:
                    logger.warning("Timed out freezing worker session %d", root)
        finally:
            # Drain rather than kill one snapshot: a member that forked just before SIGSTOP or
            # the freeze deadline may have added a process that only the next scan can see.
            # Every successful SIGKILL round removes possible forkers, so this converges without
            # letting an unfreezable member keep the bounded freeze phase alive forever.
            try:
                kill_deadline = time.monotonic() + timeout
                while live and time.monotonic() < kill_deadline:
                    signaled = False
                    for pid in live:
                        with contextlib.suppress(ProcessLookupError, PermissionError):
                            if os.getsid(pid) == root:
                                os.kill(pid, signal.SIGKILL)
                                signaled = True
                    if not signaled:
                        logger.error("Could not kill worker session %d", root)
                        break
                    current = _session_processes(root)
                    if current is None:
                        logger.error("Could not verify killed worker session %d", root)
                        break
                    live = {pid for pid, is_live in current.items() if is_live}
                if live:
                    logger.error(
                        "Worker session %d still has %d live process(es) after SIGKILL",
                        root,
                        len(live),
                    )
            finally:
                with contextlib.suppress(BaseException):
                    process.kill()
                with contextlib.suppress(BaseException):
                    process.join(timeout=timeout)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def use_threading_tqdm_lock() -> None:
    """Pin tqdm to a threading lock so it never lazily creates an `mp.RLock` a killed worker would
    leak (a `resource_tracker` semaphore warning). No-op if tqdm isn't installed."""
    try:
        import threading

        import tqdm

        tqdm.tqdm.set_lock(threading.RLock())
    except Exception:
        pass
