"""Run and group-score all rollouts for one task."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING

from verifiers.v1.decorators import discover_decorated
from verifiers.v1.retries import run_with_retry
from verifiers.v1.rollout import Phase, Rollout
from verifiers.v1.trace import Trace
from verifiers.v1.utils.memory import trim_memory_periodically

if TYPE_CHECKING:
    from verifiers.v1.retries import RolloutRetryConfig


class Episode:
    def __init__(self, rollouts: list[Rollout], retry: RolloutRetryConfig) -> None:
        if not rollouts:
            raise ValueError("an episode needs at least one rollout (n >= 1)")
        self.rollouts = rollouts
        self.task = rollouts[0].task
        self.retry = retry

    async def run(
        self,
        semaphore: asyncio.Semaphore | None = None,
        on_complete: Callable[[Trace], Awaitable[None]] | None = None,
    ) -> list[Trace]:
        """Run rollouts; delay completion callbacks only when group scoring needs all of them."""
        group_scored = bool(discover_decorated(self.task, "group_reward"))

        async def run_one(rollout: Rollout) -> Trace:
            async with semaphore or nullcontext():
                trace = await run_with_retry(rollout, self.retry)
            if not group_scored:  # reward already final → don't wait for the group
                rollout.phase = Phase.DONE
                if on_complete is not None:
                    await on_complete(trace)
            # hand freed per-turn request bodies (base64 images) back to the OS
            await trim_memory_periodically()
            return trace

        traces = await asyncio.gather(*(run_one(r) for r in self.rollouts))
        if group_scored:
            await self.task.score_group(traces)  # cross-rollout @group_rewards
            for rollout in self.rollouts:
                rollout.phase = Phase.DONE
            for trace in traces:
                if on_complete is not None:
                    await on_complete(trace)
        return traces
