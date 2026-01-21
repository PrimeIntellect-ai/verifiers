import asyncio
import inspect
import logging
from time import perf_counter
from collections.abc import Coroutine
from typing import Any, AsyncContextManager, Callable, Optional, TypeVar

import tenacity as tc

import verifiers as vf

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def maybe_await(func: Callable, *args, **kwargs):
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


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
        max_measurements: int = int(1e5),
        logger: Any | None = None,
    ):
        assert measure_interval > 0 and max_measurements > 0
        self.measure_interval = measure_interval
        self.max_measurements = max_measurements
        self.logger = logger or logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.lags = []
        self.logger.debug(
            f"Event loop lag monitor initialized with measure_interval={self.measure_interval} and max_measurements={self.max_measurements}"
        )

    async def measure_lag(self):
        """Measures event loop lag by asynchronously sleeping for interval seconds"""
        next_time = perf_counter() + self.measure_interval
        await asyncio.sleep(self.measure_interval)
        now = perf_counter()
        lag = now - next_time
        return lag

    def get_lags(self) -> list[float]:
        """Get the list of measured event loop lags."""
        return self.lags

    def reset_lags(self):
        """Reset the list of measured event loop lags."""
        self.lags = []

    async def run(self):
        """Loop to measure event loop lag. Should be started as background task."""
        while True:
            lag = await self.measure_lag()
            self.lags.append(lag)
            if len(self.lags) > self.max_measurements:
                self.lags.pop(0)

    def run_in_background(self):
        """Run the event loop lag monitor as a background task."""
        return asyncio.create_task(self.run())


def _raise_error_from_state(result):
    """Re-raise InfraError from state(s) to trigger retry."""
    if isinstance(result, dict):
        err = result.get("error")
        if err and isinstance(err, vf.InfraError):
            raise err
    elif isinstance(result, list):
        for state in result:
            err = state.get("error")
            if err and isinstance(err, vf.InfraError):
                raise err


def maybe_retry(
    func: Callable[..., Coroutine[Any, Any, T]],
    max_retries: int = 0,
    initial: float = 1.0,
    max_wait: float = 60.0,
) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Return retry-wrapped function if max_retries > 0, else return func unchanged.
    Re-raises vf.InfraError from state["error"] to trigger retry.

    Usage:
        state = await maybe_retry(self.run_rollout, max_retries=3)(input, client, ...)
    """
    if max_retries <= 0:
        return func

    def log_retry(retry_state):
        logger.warning(
            "Retrying %s (attempt %s/%s): %s",
            retry_state.fn.__name__,
            retry_state.attempt_number + 1,
            max_retries + 1,
            retry_state.outcome.exception(),
        )

    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        _raise_error_from_state(result)
        return result

    wrapper.__name__ = getattr(func, "__name__", "unknown")
    wrapper.__qualname__ = getattr(func, "__qualname__", "unknown")

    return tc.AsyncRetrying(
        retry=tc.retry_if_exception_type(vf.InfraError),
        stop=tc.stop_after_attempt(max_retries + 1),
        wait=tc.wait_exponential_jitter(initial=initial, max=max_wait),
        before_sleep=log_retry,
        reraise=True,
    ).wraps(wrapper)
