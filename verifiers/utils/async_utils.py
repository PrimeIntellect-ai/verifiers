import asyncio
import inspect
import logging
from collections import deque
from collections.abc import Coroutine
from time import perf_counter
from typing import Any, AsyncContextManager, Callable, Optional, TypeVar

import numpy as np
import tenacity as tc
from pydantic import BaseModel

import verifiers as vf
from verifiers.utils.error_utils import ErrorChain
from verifiers.utils.logging_utils import print_time

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def with_sem(sem: AsyncContextManager, coro: Coroutine[Any, Any, T]) -> T:
    """Wrap a coroutine with a context manager (typically a semaphore)."""
    try:
        async with sem:
            return await coro
    finally:
        # closes the coroutine if it was never awaited (e.g. cancelled while acquiring sem)
        coro.close()


async def maybe_await(func: Callable, *args, **kwargs):
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class NullAsyncContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


async def maybe_semaphore(
    limit: Optional[int] = None,
) -> AsyncContextManager:
    """
    Return either a real semaphore (if limit is set),
    or a no-op context manager (if limit is None or <= 0).

    Usage:
    maybe_sem = await maybe_semaphore(10)
    async with maybe_sem:
        await do_something()
    """
    if limit and limit > 0:
        return asyncio.Semaphore(limit)
    else:
        return NullAsyncContext()


class EventLoopLagMonitor:
    """A class to monitor how busy the main event loop is."""

    def __init__(
        self,
        measure_interval: float = 0.1,
        max_measurements: int = 1000,
    ):
        assert measure_interval > 0 and max_measurements > 0
        self.measure_interval = measure_interval
        self.max_measurements = max_measurements
        self.lags: deque[float] = deque(maxlen=max_measurements)

    async def measure_lag(self):
        """Measures event loop lag by asynchronously sleeping for interval seconds"""
        next_time = perf_counter() + self.measure_interval
        await asyncio.sleep(self.measure_interval)
        now = perf_counter()
        lag = now - next_time
        return lag

    async def run(self):
        """Loop to measure event loop lag. Should be started as background task."""
        while True:
            lag = await self.measure_lag()
            self.lags.append(lag)


class EventLoopLagStats(BaseModel):
    """Snapshot of event loop lag statistics."""

    min: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    max: float = 0.0
    n: int = 0

    def __str__(self) -> str:
        from verifiers.utils.logging_utils import print_time

        if self.n == 0:
            return "no samples"
        return (
            f"min={print_time(self.min)} mean={print_time(self.mean)} "
            f"median={print_time(self.median)} p90={print_time(self.p90)} "
            f"p99={print_time(self.p99)} max={print_time(self.max)} (n={self.n})"
        )

    @classmethod
    def from_monitor(cls, monitor: EventLoopLagMonitor) -> "EventLoopLagStats":
        lags = monitor.lags
        n = len(lags)
        if n == 0:
            return cls(n=0)
        arr = np.array(lags)
        return cls(
            min=float(arr.min()),
            mean=float(arr.mean()),
            median=float(np.median(arr)),
            p90=float(np.percentile(arr, 90)),
            p99=float(np.percentile(arr, 99)),
            max=float(arr.max()),
            n=n,
        )


class ProcessStats(BaseModel):
    """Snapshot of process-level resource usage (htop-style)."""

    res: int = 0  # current resident memory in bytes (htop RES)
    virt: int = 0  # virtual memory in bytes (htop VIRT)
    open_fds: int = 0

    @classmethod
    def snapshot(cls) -> "ProcessStats":
        """Capture current process stats. Works on Linux, macOS, and Windows."""
        import os
        import sys

        res = 0
        virt = 0
        open_fds = 0

        if sys.platform == "linux":
            try:
                # /proc/self/statm: fields are in pages
                page_size = os.sysconf("SC_PAGE_SIZE")
                with open("/proc/self/statm") as f:
                    parts = f.read().split()
                virt = int(parts[0]) * page_size
                res = int(parts[1]) * page_size
            except (OSError, IndexError):
                pass
            try:
                open_fds = len(os.listdir("/proc/self/fd"))
            except OSError:
                pass
        elif sys.platform == "darwin":
            import resource

            # macOS: ru_maxrss is in bytes and is peak RSS (best we can
            # do without ctypes into mach_task_basic_info)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            res = usage.ru_maxrss
            try:
                open_fds = len(os.listdir("/dev/fd"))
            except OSError:
                pass

        return cls(res=res, virt=virt, open_fds=open_fds)

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        if n >= 1024**3:
            return f"{n / 1024**3:.1f}G"
        if n >= 1024**2:
            return f"{n / 1024**2:.0f}M"
        if n >= 1024:
            return f"{n / 1024:.0f}K"
        return f"{n}B"

    def __str__(self) -> str:
        parts = []
        if self.res:
            parts.append(f"RES: {self._fmt_bytes(self.res)}")
        if self.virt:
            parts.append(f"VIRT: {self._fmt_bytes(self.virt)}")
        if self.open_fds:
            parts.append(f"FDs: {self.open_fds}")
        return " | ".join(parts) if parts else "no process stats"


def maybe_retry(
    func: Callable[..., Coroutine[Any, Any, T]],
    max_retries: int = 0,
    initial: float = 1.0,
    max_wait: float = 60.0,
    error_types: tuple[type[Exception], ...] = (
        vf.InfraError,
        vf.InvalidModelResponseError,
    ),
) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Return retry-wrapped function if max_retries > 0, else return func unchanged.
    Re-raises specified errors from state["error"] to trigger tenacity retry.
    Returns result with error in state if retries are exhausted (does not crash).

    Usage:
        state = await maybe_retry(self.run_rollout, max_retries=3)(input, client, ...)
    """
    if max_retries <= 0:
        return func

    def reraise_error_from_state(result, error_types: tuple[type[Exception], ...]):
        """Re-raise specified errors from state(s) to trigger tenacity retry."""
        if isinstance(result, dict):
            err = result.get("error")
            if err and any(isinstance(err, err_type) for err_type in error_types):
                raise err
        elif isinstance(result, list):
            for state in result:
                err = state.get("error")
                if err and any(isinstance(err, err_type) for err_type in error_types):
                    raise err

    def log_retry(retry_state: tc.RetryCallState) -> None:
        """Log a warning with the exception and the number of attempts."""
        caller = retry_state.fn.__name__ if retry_state.fn else "unknown function"
        error_chain = (
            repr(
                ErrorChain(
                    retry_state.outcome.exception() or Exception("Unknown exception")
                )
            )
            if retry_state.outcome
            else None
        )
        next_action = retry_state.next_action.sleep if retry_state.next_action else 0
        logger.warning(
            f"Caught {error_chain} in {caller}. Retrying in {print_time(next_action)} (retry {retry_state.attempt_number}/{max_retries})"
        )

    last_result = None

    def return_last_result(retry_state: tc.RetryCallState):
        """Return the last result when retries are exhausted (instead of raising)."""
        caller = retry_state.fn.__name__ if retry_state.fn else "unknown function"
        error_chain = (
            repr(
                ErrorChain(
                    retry_state.outcome.exception() or Exception("Unknown exception")
                )
            )
            if retry_state.outcome
            else None
        )
        logger.error(
            f"Retries exhausted for {caller} after {max_retries} attempts. "
            f"Last error: {error_chain}. Continuing with error in state."
        )
        return last_result

    async def wrapper(*args, **kwargs):
        nonlocal last_result
        result = await func(*args, **kwargs)
        last_result = result  # store result
        reraise_error_from_state(result, error_types)
        return result

    wrapper.__name__ = getattr(func, "__name__", "unknown")
    wrapper.__qualname__ = getattr(func, "__qualname__", "unknown")

    return tc.AsyncRetrying(
        retry=tc.retry_if_exception_type(error_types),
        stop=tc.stop_after_attempt(max_retries + 1),
        wait=tc.wait_exponential_jitter(initial=initial, max=max_wait),
        before_sleep=log_retry,
        retry_error_callback=return_last_result,
        reraise=True,
    ).wraps(wrapper)
