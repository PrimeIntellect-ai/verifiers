"""An episode: evaluate one task — its rollout(s) and all scoring across them.

An Episode is the largest unit the *evaluator* knows about: it runs `n` Rollouts of one
task (each a single trajectory) under a shared concurrency limit, then scores across
them. Per-rollout `@reward`/`@metric` already ran inside each Rollout; the Episode adds
the cross-rollout `@group_reward` stage — pairwise/preference rewards that compare a
task's rollouts. n=1 is just an episode with a single rollout.

These are env-level *rewards*. Training-time transforms of rewards — advantages (GRPO,
RLOO), on-policy distillation — are deliberately NOT modeled here; they sit a level
above, in a trainer that consumes episodes. That's why this is an "Episode" and not a
"group": nothing here computes an advantage.

Each Rollout tears its own runtime down (see rollout.py), so the Episode owns no
runtimes — only the rollouts and the scoring across them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING

from verifiers.v1.decorators import discover_decorated
from verifiers.v1.retries import run_with_retry
from verifiers.v1.rollout import Phase, Rollout
from verifiers.v1.taskset import Taskset
from verifiers.v1.trace import Trace

if TYPE_CHECKING:
    from verifiers.v1.retries import RetryConfig


class Episode:
    def __init__(
        self, rollouts: list[Rollout], taskset: Taskset, retry: RetryConfig
    ) -> None:
        self.rollouts = rollouts
        self.taskset = taskset
        self.retry = retry

    async def run(
        self,
        semaphore: asyncio.Semaphore | None = None,
        shared_urls: dict[str, str] | None = None,
        on_complete: Callable[[Trace], None] = lambda _trace: None,
    ) -> list[Trace]:
        """Run all rollouts (each under `semaphore`), then group-score across their
        traces. Without `@group_reward`s a rollout's reward is final the moment its own
        scoring ends, so it's marked DONE then (no waiting for slower siblings); with
        them, the whole group is marked DONE together after `score_group` — the reward
        isn't final until every rollout is in. `shared_urls` are eval-level shared tool
        servers passed through to each rollout. `on_complete` (the runner's persist hook)
        is called with each trace the instant it's finalized (DONE) — per rollout without
        group rewards, or once per trace after group scoring with them."""
        group_scored = bool(discover_decorated(self.taskset, "group_reward"))

        async def run_one(rollout: Rollout) -> Trace:
            async with semaphore or nullcontext():
                trace = await run_with_retry(rollout, shared_urls, self.retry)
            if not group_scored:  # reward already final → don't wait for the group
                rollout.phase = Phase.DONE
                on_complete(trace)
            return trace

        traces = await asyncio.gather(*(run_one(r) for r in self.rollouts))
        if group_scored:
            await self.taskset.score_group(traces)  # cross-rollout @group_rewards
            for rollout in self.rollouts:
                rollout.phase = Phase.DONE
            for trace in traces:
                on_complete(trace)
        return traces
