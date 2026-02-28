"""
Multi-agent rubric with per-actor rewards.

Extends Rubric with:
- Per-actor reward functions (different rewards for different actors)
- Per-actor GRPO advantages (within-actor normalization)
- Group funcs run per-actor-group (solvers vs solvers, not vs proposers)
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import cast

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import GroupRewardFunc, RewardFunc, State


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

        # Actor metadata: actor_id -> {"is_trainable": bool}
        # Populated by MultiAgentEnv.register_actor() during setup
        self.actors: dict[str, dict] = {}

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

    def register_actor(self, actor_id: str, is_trainable: bool = True) -> None:
        """Register an actor with trainability metadata.

        Called automatically by MultiAgentEnv during setup.
        """
        self.actors[actor_id] = {"is_trainable": is_trainable}

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
    ) -> tuple[float, dict[str, float]]:
        """Compute reward using actor-specific + global reward functions."""
        total_reward = 0.0
        metrics: dict[str, float] = {}

        # Compute rewards for the current actor
        actor_funcs = self.actor_reward_funcs.get(actor_id, [])
        for func, weight in actor_funcs:
            try:
                score = await self._call_individual_reward_func(func, state)
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
                    score = await self._call_individual_reward_func(func, state)
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
    ) -> None:
        """
        Score with per-actor GRPO advantages (solver vs solver, not vs proposer).

        Features:
        - Children scored first (so parents can read child rewards)
        - Per-actor reward functions
        - Group funcs run per-actor-group
        - Per-actor GRPO advantage normalization
        """
        if not states:
            self.logger.warning("No states to score")
            return

        start_time = time.time()

        # Score children first, then parents (so parents can read child rewards)
        children = [s for s in states if not s.get("child_states")]
        parents = [s for s in states if s.get("child_states")]

        await self._score_states(children)
        await self._score_states(parents)

        # Run group funcs per-actor-group
        await self._run_group_funcs_per_actor(states)

        # Compute GRPO advantages per-actor group
        actor_groups: dict[str, list[State]] = defaultdict(list)
        for state in states:
            actor_id = self.get_actor_id_from_state(state) or "default"
            actor_groups[actor_id].append(state)

        for actor_id, actor_states in actor_groups.items():
            # Skip GRPO advantage computation for non-trainable actors
            # (they still get scored for logging, just no advantage for training)
            if actor_states and not actor_states[0].get("is_trainable", True):
                for state in actor_states:
                    state["advantage"] = 0.0
                    for step in state.get("trajectory", []):
                        if step.get("advantage") is None:
                            step["advantage"] = 0.0
                        if step.get("reward") is None:
                            step["reward"] = state["reward"]
                continue

            actor_rewards = [s["reward"] for s in actor_states]
            mean_reward = sum(actor_rewards) / len(actor_rewards)

            for state in actor_states:
                advantage = state["reward"] - mean_reward
                state["advantage"] = advantage

                for step in state.get("trajectory", []):
                    if step.get("advantage") is None:
                        step["advantage"] = advantage
                    if step.get("reward") is None:
                        step["reward"] = state["reward"]

        # Timing tracking (match parent)
        end_time = time.time()
        scoring_ms = (end_time - start_time) * 1000
        for state in states:
            if "timing" in state:
                state["timing"]["scoring_ms"] = scoring_ms
                state["timing"]["total_ms"] += scoring_ms

    async def _score_states(
        self,
        states: list[State],
    ) -> None:
        """Score a list of states with individual reward funcs."""
        if not states:
            return

        actor_ids = [self.get_actor_id_from_state(s) or "default" for s in states]

        reward_tasks = [
            self._compute_actor_reward(state, actor_id)
            for state, actor_id in zip(states, actor_ids)
        ]
        results = await asyncio.gather(*reward_tasks)

        for state, actor_id, (reward, metrics) in zip(states, actor_ids, results):
            state["reward"] = reward
            state["metrics"] = metrics

    async def _run_group_funcs_per_actor(
        self,
        states: list[State],
    ) -> None:
        """Run group reward funcs per-actor-group (solvers vs solvers, not vs proposers)."""
        group_funcs = [(f, w) for f, w in zip(self.funcs, self.weights) if self._is_group_func(f)]
        if not group_funcs:
            return

        # Group states by actor
        actor_groups: dict[str, list[State]] = defaultdict(list)
        for state in states:
            actor_id = self.get_actor_id_from_state(state) or "default"
            actor_groups[actor_id].append(state)

        # Run group funcs on each actor group
        for actor_id, actor_states in actor_groups.items():
            for func, weight in group_funcs:
                group_func = cast(GroupRewardFunc, func)
                scores = await self._call_group_reward_func(group_func, actor_states)

                for state, score in zip(actor_states, scores):
                    state["reward"] += score * weight
                    if state["metrics"] is None:
                        state["metrics"] = {}
                    state["metrics"][func.__name__] = score
