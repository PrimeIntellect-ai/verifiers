import asyncio
import inspect
import logging
from collections.abc import Coroutine
from time import perf_counter
from typing import Any, AsyncContextManager, Callable, Optional, TypeVar

import tenacity as tc

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
