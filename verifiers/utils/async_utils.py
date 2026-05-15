import asyncio
import inspect
import logging
from collections import deque
from collections.abc import Mapping
from collections.abc import Coroutine
from time import perf_counter
from typing import Any, AsyncContextManager, Callable, Optional, TypeVar, cast

import numpy as np
import tenacity as tc
from pydantic import BaseModel

import verifiers as vf
from verifiers.utils.error_utils import ErrorChain
from verifiers.utils.error_utils import error_info_to_exception
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


async def maybe_call_with_named_args(func: Callable, **objects):
    sig = inspect.signature(func)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return await maybe_await(func, **objects)
    allowed = {key: value for key, value in objects.items() if key in sig.parameters}
    return await maybe_await(func, **allowed)


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


def maybe_retry(
    func: Callable[..., Coroutine[Any, Any, T]],
    max_retries: int = 0,
    initial: float = 1.0,
    max_wait: float = 60.0,
    retry_error_types: tuple[type[Exception], ...] = (
        vf.InfraError,
        vf.InvalidModelResponseError,
    ),
    terminal_error_types: tuple[type[Exception], ...] = (),
) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Return retry-wrapped function if max_retries > 0, else return func unchanged.
    Re-raises specified errors from state["error"] to trigger tenacity retry.
    Returns result with error in state if retries are exhausted (does not crash).

    Usage:
        state = await maybe_retry(self.run_rollout, max_retries=3)(input, client, ...)
    """
    if max_retries <= 0 and not terminal_error_types:
        return func

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

    def log_terminal(error: Exception) -> None:
        """Log a warning when a terminal error stops retries."""
        caller = getattr(func, "__name__", "unknown function")
        error_chain = repr(ErrorChain(error))
        logger.warning(
            f"Caught {error_chain} in {caller}. Returning result without retry."
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
        terminal_error = find_error_in_state(result, terminal_error_types)
        if terminal_error is not None:
            log_terminal(terminal_error)
            return result
        if max_retries > 0:
            retry_error = find_error_in_state(result, retry_error_types)
            if retry_error is not None:
                raise retry_error
        return result

    wrapper.__name__ = getattr(func, "__name__", "unknown")
    wrapper.__qualname__ = getattr(func, "__qualname__", "unknown")

    if max_retries <= 0:
        return wrapper

    return tc.AsyncRetrying(
        retry=tc.retry_if_exception_type(retry_error_types),
        stop=tc.stop_after_attempt(max_retries + 1),
        wait=tc.wait_exponential_jitter(initial=initial, max=max_wait),
        before_sleep=log_retry,
        retry_error_callback=return_last_result,
        reraise=True,
    ).wraps(wrapper)


def find_error_in_state(
    result: Any,
    error_types: tuple[type[Exception], ...],
) -> Exception | None:
    """Return the first matching error from a state or list of states."""

    def match_error(err: object) -> Exception | None:
        if isinstance(err, Exception) and any(
            isinstance(err, err_type) for err_type in error_types
        ):
            return err
        if isinstance(err, Mapping):
            return error_info_to_exception(cast(Mapping[str, object], err), error_types)
        return None

    if isinstance(result, dict):
        return match_error(result.get("error"))
    elif isinstance(result, list):
        for state in result:
            if isinstance(state, dict):
                matched = match_error(state.get("error"))
                if matched is not None:
                    return matched
    return None
