"""
MultiAgentRubric: Rubric that propagates rewards to child_states.

Extends Rubric.score_group to traverse the state tree and apply
the root state's reward/advantage to all nested trajectory steps.
"""
import asyncio
import time
from typing import AsyncContextManager, cast

from verifiers.types import GroupRewardFunc, RewardFunc, State

from .rubric import Rubric


class MultiAgentRubric(Rubric):
    """
    Rubric that propagates rewards to child_states.

    When scoring a group, after computing rewards for top-level states,
    traverses into child_states and applies the same reward/advantage
    to all trajectory steps in the tree.
    """

    async def score_group(self, states: list[State], score_sem: AsyncContextManager):
        """
        Score a group of rollouts together, propagating to child_states.

        All reward functions are executed in order, parallelizing across states.
        After scoring, propagates the reward/advantage to all nested child_states.
        """
        start_time = time.time()
        num_states = len(states)
        if num_states == 0:
            self.logger.warning("No states to score")
            return
        aggregated_rewards = [0.0] * num_states
        aggregated_metrics: dict[str, list[float]] = {}

        # process functions in order
        for func, weight in zip(self.funcs, self.weights):
            is_group = self._is_group_func(func)
            if is_group:
                # GroupRewardFunc: score all states together
                group_func = cast(GroupRewardFunc, func)
                scores = await self._call_group_reward_func(
                    group_func, states, score_sem=score_sem
                )
                func_name = func.__name__
                if func_name not in aggregated_metrics:
                    aggregated_metrics[func_name] = [0.0] * num_states
                for i in range(num_states):
                    score_value = scores[i]
                    aggregated_rewards[i] += score_value * weight
                    aggregated_metrics[func_name][i] = score_value
            else:
                reward_func = cast(RewardFunc, func)
                score_tasks = [
                    self._call_individual_reward_func(
                        reward_func, state, score_sem=score_sem
                    )
                    for state in states
                ]
                scores = await asyncio.gather(*score_tasks)

                func_name = func.__name__
                if func_name not in aggregated_metrics:
                    aggregated_metrics[func_name] = [0.0] * num_states
                for i in range(num_states):
                    score_value = scores[i]
                    aggregated_rewards[i] += score_value * weight
                    aggregated_metrics[func_name][i] = score_value

        # update states with aggregated results
        end_time = time.time()
        scoring_ms = (end_time - start_time) * 1000
        avg_reward = sum(aggregated_rewards) / num_states
        for i, state in enumerate(states):
            state["reward"] = aggregated_rewards[i]
            state["advantage"] = aggregated_rewards[i] - avg_reward
            for t in state["trajectory"]:
                if t["advantage"] is None:
                    t["advantage"] = state["advantage"]
                if t["reward"] is None:
                    t["reward"] = state["reward"]
            state["metrics"] = {
                func_name: values[i] for func_name, values in aggregated_metrics.items()
            }
            state["timing"]["scoring_ms"] = scoring_ms
            state["timing"]["total_ms"] += state["timing"]["scoring_ms"]

            # Propagate reward/advantage to all child_states
            self._propagate_to_children(
                state.get("child_states", []),
                state["reward"],
                state["advantage"],
            )

    def _propagate_to_children(
        self,
        child_states: list[State],
        reward: float,
        advantage: float,
    ):
        """
        Recursively apply reward/advantage to all trajectory steps in child_states.
        """
        for child in child_states:
            child["reward"] = reward
            child["advantage"] = advantage
            for t in child.get("trajectory", []):
                if t["advantage"] is None:
                    t["advantage"] = advantage
                if t["reward"] is None:
                    t["reward"] = reward
            # Recurse into nested children
            self._propagate_to_children(
                child.get("child_states", []),
                reward,
                advantage,
            )
