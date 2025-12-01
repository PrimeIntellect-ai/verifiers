import asyncio
from typing import Any, Callable, TypeVar
from logging import Logger
from verifiers.utils.async_utils import maybe_await

T = TypeVar("T")


async def with_retry(
    func: Callable[..., T],
    *args: Any,
    logger: Logger,
    max_retries: int = 5,
    base_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_backoff_seconds: float = 30.0,
    **kwargs: Any,
) -> T:
    delay = base_delay
    for attempt in range(max_retries):
        try:
            result = await maybe_await(func, *args, **kwargs)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            name = func.__name__ if hasattr(func, "__name__") else str(func)
            logger.error(f"Error calling {name}: {e}")
            await asyncio.sleep(min(delay, max_backoff_seconds))
            delay *= backoff_factor
    raise RuntimeError(f"Failed after {max_retries} attempts")
