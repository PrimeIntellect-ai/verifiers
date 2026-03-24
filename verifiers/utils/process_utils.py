"""Process lifecycle utilities."""

import logging
import os
import signal
import threading
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

logger = logging.getLogger(__name__)


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

    Sends SIGTERM to all live processes first, then joins all with a shared
    timeout, escalating to SIGKILL for any that don't exit in time.
    """
    alive = [p for p in processes if p.is_alive()]
    for p in alive:
        p.terminate()
    for p in alive:
        p.join(timeout=timeout)
        if p.is_alive():
            p.kill()
            p.join(timeout=kill_timeout)
