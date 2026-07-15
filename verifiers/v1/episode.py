"""Run and group-score all rollouts for one task."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

from verifiers.v1.decorators import discover_decorated
from verifiers.v1.retries import run_with_retry
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.utils.memory import trim_memory_periodically

if TYPE_CHECKING:
    from verifiers.v1.agent import Agent
    from verifiers.v1.retries import RolloutRetryConfig


@dataclass
class RunSlot:
    """One planned run of the episode's task, observable while it happens: `trace`
    points at the current attempt's live trace from the moment the engine mints it
    (a retry repoints it at the fresh attempt's), and `done` flips once that trace is
    final — scored, including the task's cross-rollout `@group_reward`s. The `--rich`
    dashboard renders slots (deriving the live stage from the trace's timing spans);
    `--resume` preloads the previous session's kept traces as `finished` slots."""

    task: Task
    trace: Trace | None = None
    done: bool = False

    @classmethod
    def finished(cls, trace: Trace) -> RunSlot:
        return cls(task=Task(trace.task.data), trace=trace, done=True)


class Episode:
    """The `n` rollouts of one task: each runs through `agent.run` (with whole-rollout
    retries), then the task's `@group_reward`s run across their traces."""

    def __init__(
        self, agent: Agent, task: Task, n: int, retry: RolloutRetryConfig
    ) -> None:
        if n < 1:
            raise ValueError("an episode needs at least one rollout (n >= 1)")
        # A task with @group_rewards compares its rollouts, so it needs >=2 of them —
        # refuse n < 2 there (rather than silently scoring a group of one).
        if n < 2 and discover_decorated(task, "group_reward"):
            raise ValueError(
                f"task {task.data.idx!r} defines @group_reward(s), which compare a task's rollouts "
                f"and need >=2; got n={n} (pass -r/--num-rollouts >= 2)"
            )
        self.agent = agent
        self.task = task
        self.retry = retry
        self.slots = [RunSlot(task) for _ in range(n)]

    async def run(
        self,
        semaphore: asyncio.Semaphore | None = None,
        on_complete: Callable[[Trace], Awaitable[None]] | None = None,
    ) -> list[Trace]:
        """Run rollouts; delay completion callbacks only when group scoring needs all of them."""
        group_scored = bool(discover_decorated(self.task, "group_reward"))

        async def run_one(slot: RunSlot) -> Trace:
            def watch(trace: Trace) -> None:
                slot.trace = trace

            async with semaphore or nullcontext():
                trace = await run_with_retry(
                    lambda: self.agent.run(self.task, on_trace=watch), self.retry
                )
            if not group_scored:  # reward already final → don't wait for the group
                slot.done = True
                if on_complete is not None:
                    await on_complete(trace)
            # hand freed per-turn request bodies (base64 images) back to the OS
            await trim_memory_periodically()
            return trace

        traces = await asyncio.gather(*(run_one(s) for s in self.slots))
        if group_scored:
            await self.task.score_group(traces)  # cross-rollout @group_rewards
            for slot in self.slots:
                slot.done = True
            for trace in traces:
                if on_complete is not None:
                    await on_complete(trace)
        return traces
