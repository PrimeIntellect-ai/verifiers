"""Run and group-score all env-rollouts for one task."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from verifiers.v1.decorators import discover_decorated
from verifiers.v1.retries import run_record_with_retry
from verifiers.v1.task import Task
from verifiers.v1.trace import RolloutRecord, Trace
from verifiers.v1.utils.memory import trim_memory_periodically

if TYPE_CHECKING:
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.env import Environment
    from verifiers.v1.retries import RolloutRetryConfig


@dataclass
class RunSlot:
    """One planned env-rollout of the episode's task, observable while it happens:
    `traces` collects the current attempt's live traces from the moment the engine
    mints them (a retry restarts the list with the fresh attempt's; a single-agent
    rollout has exactly one), `record` is the finished rollout's record, and `done`
    flips once that record is final — scored, including the task's cross-rollout
    `@group_reward`s. The `--rich` dashboard renders slots (deriving each trace's live
    stage from its timing spans); `--resume` preloads the previous session's kept
    records as `finished` slots."""

    task: Task
    traces: list[Trace] = field(default_factory=list)
    record: RolloutRecord | None = None
    done: bool = False

    @classmethod
    def finished(cls, record: RolloutRecord) -> RunSlot:
        return cls(
            task=Task(record.task.data),
            traces=list(record.traces),
            record=record,
            done=True,
        )


class Episode:
    """The `n` env-rollouts of one task: each runs through `env.run_record` (with
    whole-record retries), then the task's `@group_reward`s run across their traces."""

    def __init__(
        self,
        env: Environment,
        task: Task,
        ctx: ModelContext,
        n: int,
        retry: RolloutRetryConfig,
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
        self.env = env
        self.task = task
        self.ctx = ctx
        self.retry = retry
        self.slots = [RunSlot(task) for _ in range(n)]

    async def run(
        self,
        semaphore: asyncio.Semaphore | None = None,
        on_complete: Callable[[RolloutRecord], Awaitable[None]] | None = None,
    ) -> list[RolloutRecord]:
        """Run the env-rollouts; delay completion callbacks only when group scoring
        needs all of them."""
        group_scored = bool(discover_decorated(self.task, "group_reward"))

        async def run_one(slot: RunSlot) -> RolloutRecord:
            async def attempt() -> RolloutRecord:
                slot.traces = []  # a retry shows the fresh attempt's traces
                return await self.env.run_record(
                    self.task, self.ctx, on_trace=slot.traces.append
                )

            async with semaphore or nullcontext():
                record = await run_record_with_retry(attempt, self.retry)
            # The record is authoritative: the hook's returned order, or (on a hook
            # failure) the completed subset.
            slot.traces = list(record.traces)
            slot.record = record
            if not group_scored:  # reward already final → don't wait for the group
                slot.done = True
                if on_complete is not None:
                    await on_complete(record)
            # hand freed per-turn request bodies (base64 images) back to the OS
            await trim_memory_periodically()
            return record

        records = await asyncio.gather(*(run_one(s) for s in self.slots))
        if group_scored:
            # cross-rollout @group_rewards, over every trace of the task's records
            await self.task.score_group([t for r in records for t in r.traces])
            for slot in self.slots:
                slot.done = True
            for record in records:
                if on_complete is not None:
                    await on_complete(record)
        return records
