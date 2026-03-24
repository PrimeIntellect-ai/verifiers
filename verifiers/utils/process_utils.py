"""Process lifecycle utilities."""

import signal
import sys
from multiprocessing.process import BaseProcess


def request_parent_death_signal() -> None:
    """Ask the Linux kernel to send SIGTERM when the parent process dies."""
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


def terminate_process(
    process: BaseProcess | None,
    timeout: float = 5.0,
    kill_timeout: float = 5.0,
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
