import inspect
from typing import Callable

import asyncio


async def maybe_await(func: Callable, *args, **kwargs):
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def tqdm_gather_with_metrics(tasks: list, total: int, desc: str) -> list:
    """
    Gather async tasks with tqdm progress bar showing running averages
    for reward and completion token count.
    """
    from tqdm.asyncio import tqdm as async_tqdm

    results = [None] * total
    progress_bar = async_tqdm(total=total, desc=desc)

    count = 0
    sum_reward = 0.0
    sum_tokens = 0

    async def track(idx: int, task):
        nonlocal count, sum_reward, sum_tokens
        results[idx] = result = await task
        count += 1

        # we expect a tuple of (reward, state)
        if result and len(result) == 2:
            sum_reward += result[0]

            # extract token count from state using OpenAI standard format
            state = result[1]
            if "responses" in state:
                for response in state["responses"]:
                    if hasattr(response, "usage") and response.usage:
                        sum_tokens += response.usage.completion_tokens or 0

        description_parts = [desc]
        if sum_reward:
            description_parts.append(f"reward={sum_reward / count:.3f}")
        if sum_tokens:
            description_parts.append(f"completion_length={sum_tokens / count:.0f}")

        if len(description_parts) > 1:
            progress_bar.set_description(" | ".join(description_parts))
        progress_bar.update(1)

    await asyncio.gather(*[track(i, t) for i, t in enumerate(tasks)])
    progress_bar.close()
    return results
