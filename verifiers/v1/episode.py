"""An episode: evaluate one task — its rollout(s) and all scoring across them.

An Episode is the largest unit the *evaluator* knows about: it runs `n` Rollouts of one
task (each a single trajectory) under a shared concurrency limit, then scores across
them. Per-rollout `@reward`/`@metric` already ran inside each Rollout; the Episode adds
the cross-rollout `@group_reward` stage — pairwise/preference rewards (declared on the
task's class) that compare a task's rollouts. n=1 is just an episode with a single
rollout. It is also the trivial topology: one agent, one node — the topology layer
(`verifiers.v1.topology`) generalizes exactly this shape across agents.

These are env-level *rewards*. Training-time transforms of rewards — advantages (GRPO,
RLOO), on-policy distillation — are deliberately NOT modeled here; they sit a level
above, in a trainer that consumes episodes. That's why this is an "Episode" and not a
"group": nothing here computes an advantage.

Each Rollout tears its own runtime down (see rollout.py), so the Episode owns no
runtimes — only the rollouts and the scoring across them.
"""

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
    from verifiers.v1.task import Task


def requires_group_scoring(tasks: "list[Task]") -> bool:
    """Whether any of `tasks` declares `@group_reward`s — probed once per distinct task
    class, since a factory may emit heterogeneous classes (only some group-scored).
    The runner/server coarse-grained switch; the Episode itself decides per task."""
    seen: set[type] = set()
    for task in tasks:
        if type(task) not in seen:
            seen.add(type(task))
            if discover_decorated(task, "group_reward"):
                return True
    return False


class Episode:
    def __init__(self, rollouts: list[Rollout], retry: RolloutRetryConfig) -> None:
        self.rollouts = rollouts
        self.retry = retry
        self.task = rollouts[0].task
        """The one task all rollouts share — the owner of the `@group_reward`s scored
        across their traces."""

    async def run(
        self,
        semaphore: asyncio.Semaphore | None = None,
        on_complete: Callable[[Trace], Awaitable[None]] | None = None,
    ) -> list[Trace]:
        """Run all rollouts (each under `semaphore`), then group-score across their
        traces. Without `@group_reward`s a rollout's reward is final the moment its own
        scoring ends, so it's marked DONE then (no waiting for slower siblings); with
        them, the whole group is marked DONE together after `score_group` — the reward
        isn't final until every rollout is in. Each rollout already carries the run-level
        shared tool servers / interception pool (injected by `Environment.episode`).
        `on_complete` (the runner's persist hook) is called with each trace the instant
        it's finalized (DONE) — per rollout without group rewards, or once per trace after
        group scoring with them."""
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
