from __future__ import annotations

import asyncio
import random

from openai import RateLimitError


def with_rate_limit_retry(
    concurrency_semaphore: asyncio.Semaphore,
    delay_semaphore: asyncio.Semaphore,
    rate_limit_event: asyncio.Event,
    max_retries: int = 5,
    base_delay: float = 1.0,
):
    """
    Decorator for async functions to handle OpenAI-style rate limiting with
    shared backoff coordination across concurrent tasks.

    Notes:
      - Preserves original function signature (wrapper simply forwards *args/**kwargs).
      - Uses a shared Event to temporarily pause new calls when any call hits a 429.
      - Backoff curve ~ 1.36787944**attempt with jitter (constant ~= 1 + 1/e).

    Args mirror the original inlined version.
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    async with concurrency_semaphore:
                        # If any peer has signaled rate limiting, wait for the release.
                        if rate_limit_event.is_set():
                            await rate_limit_event.wait()
                            # Stagger the herd slightly after release.
                            await asyncio.sleep(random.uniform(0, 2))

                        return await func(*args, **kwargs)

                except RateLimitError:
                    if attempt == max_retries - 1:
                        raise

                    # Signal all other calls to pause.
                    rate_limit_event.set()

                    # Exponential backoff with jitter.
                    delay = base_delay * (1.36787944**attempt) + random.uniform(0, 1)

                    # Coordinate the wait across concurrent tasks.
                    async with delay_semaphore:
                        await asyncio.sleep(delay)
                        # Clear the event to allow other calls to proceed.
                        rate_limit_event.clear()

        return wrapper

    return decorator
