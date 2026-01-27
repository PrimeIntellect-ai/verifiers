"""
Multi-agent rubric with per-actor rewards.

Extends Rubric with:
- Per-actor reward functions (different rewards for different actors)
- Per-actor GRPO advantages (within-actor normalization)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import AsyncContextManager

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import RewardFunc, State


class MultiAgentRubric(Rubric):
    """
    Rubric with per-actor rewards.

    GRPO advantages are computed within actor groups (solver vs solver),
    not across all actors, preventing unfair comparisons.
    """

    def __init__(
        self,
        funcs: list[RewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: vf.Parser | None = None,
    ):
        super().__init__(funcs=funcs, weights=weights, parser=parser)

        # Per-actor reward functions: actor_id -> [(func, weight), ...]
        self.actor_reward_funcs: dict[str, list[tuple[RewardFunc, float]]] = defaultdict(list)

    def add_actor_reward_func(
        self,
        actor_id: str,
        func: RewardFunc,
        weight: float = 1.0,
    ) -> None:
        """Add a reward function specific to an actor."""
        self.actor_reward_funcs[actor_id].append((func, weight))

    def add_actor_metric(
        self,
        actor_id: str,
        func: RewardFunc,
    ) -> None:
        """Add a metric (zero-weight reward) for logging without affecting reward."""
        self.add_actor_reward_func(actor_id, func, weight=0.0)

    def get_actor_id_from_state(self, state: State) -> str | None:
        """Extract actor ID from state (checks extras, actor_history, trajectory)."""
        # Check extras first (primary location)
        extras = state.get("extras", {})
        if "current_actor_id" in extras:
            return extras["current_actor_id"]

        # Check actor_history (take first actor if available)
        actor_history = extras.get("actor_history", [])
        if actor_history:
            return actor_history[0][0]

        # Check trajectory steps
        trajectory = state.get("trajectory", [])
        for step in trajectory:
            step_extras = step.get("extras", {})
            if "actor_id" in step_extras:
                return step_extras["actor_id"]

        return None

    async def _compute_actor_reward(
        self,
        state: State,
        actor_id: str,
        score_sem: AsyncContextManager,
    ) -> tuple[float, dict[str, float]]:
        """Compute reward using actor-specific + global reward functions."""
        total_reward = 0.0
        metrics: dict[str, float] = {}

        # Compute rewards for the current actor
        actor_funcs = self.actor_reward_funcs.get(actor_id, [])
        for func, weight in actor_funcs:
            try:
                score = await self._call_individual_reward_func(func, state, score_sem)
                score = score if score is not None else 0.0
                metrics[func.__name__] = score
                total_reward += score * weight
            except Exception as e:
                self.logger.error(f"Error in actor reward func {func.__name__}: {e}")
                metrics[func.__name__] = 0.0

        # Also compute global reward functions
        for func, weight in zip(self.funcs, self.weights):
            if not self._is_group_func(func):
                try:
                    score = await self._call_individual_reward_func(func, state, score_sem)
                    score = score if score is not None else 0.0
                    total_reward += score * weight
                    metrics[func.__name__] = score
                except Exception as e:
                    self.logger.error(f"Error in global reward func {func.__name__}: {e}")
                    metrics[func.__name__] = 0.0

        return total_reward, metrics

    async def score_group(
        self,
        states: list[State],
        score_sem: AsyncContextManager,
    ) -> None:
        """Score with per-actor GRPO advantages (solver vs solver, not vs proposer)."""
        if not states:
            self.logger.warning("No states to score")
            return

        # Extract actor_ids once
        actor_ids = [self.get_actor_id_from_state(s) or "default" for s in states]

        # Compute individual rewards in parallel
        reward_tasks = [
            self._compute_actor_reward(state, actor_id, score_sem)
            for state, actor_id in zip(states, actor_ids)
        ]
        results = await asyncio.gather(*reward_tasks)

        # Apply rewards and group by actor
        actor_groups: dict[str, list[State]] = defaultdict(list)
        for state, actor_id, (reward, metrics) in zip(states, actor_ids, results):
            state["reward"] = reward
            state["metrics"] = metrics
            actor_groups[actor_id].append(state)

        # Compute GRPO advantages per-actor group
        for actor_id, actor_states in actor_groups.items():
            # Compute mean reward for this actor group
            actor_rewards = [s["reward"] for s in actor_states]
            mean_reward = sum(actor_rewards) / len(actor_rewards)

            # Compute advantages relative to actor mean
            for state in actor_states:
                advantage = state["reward"] - mean_reward
                state["advantage"] = advantage

                # Propagate to trajectory steps
                for step in state.get("trajectory", []):
                    if step.get("advantage") is None:
                        step["advantage"] = advantage
                    if step.get("reward") is None:
                        step["reward"] = state["reward"]
