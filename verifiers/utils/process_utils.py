"""Process lifecycle utilities."""

import contextlib
import logging
import os
import signal
import subprocess
import threading
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


def _session_processes(session: int) -> set[int]:
    """Return the current members of a POSIX session."""
    try:
        listing = subprocess.run(
            ["ps", "-A", "-o", "pid=,sid="],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return {
            pid
            for line in listing.splitlines()
            for pid, sid in [map(int, line.split())]
            if sid == session
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        logger.warning("Could not read POSIX session %d", session)
        return set()


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
    if root not in members:
        with contextlib.suppress(Exception):
            process.kill()
        process.join(timeout=timeout)
        return

    while members:
        for pid in members:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGSTOP)
        discovered = _session_processes(root)
        if discovered <= members:
            break
        members.update(discovered)

    for pid in members:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    process.join(timeout=timeout)


def use_threading_tqdm_lock() -> None:
    """Pin tqdm to a threading lock so it never lazily creates an `mp.RLock` a killed worker would
    leak (a `resource_tracker` semaphore warning). No-op if tqdm isn't installed."""
    try:
        import threading

        import tqdm

        tqdm.tqdm.set_lock(threading.RLock())
    except Exception:
        pass
