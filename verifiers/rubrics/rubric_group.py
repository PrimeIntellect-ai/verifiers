from typing import AsyncContextManager

from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    RewardFunc,
    State,
)


class RubricGroup(Rubric):
    """
    Class for aggregating multiple rubrics.
    """

    def __init__(self, rubrics: list[Rubric], **kwargs):
        if not rubrics:
            raise ValueError("RubricGroup must have at least one rubric")
        super().__init__(**kwargs)
        self.rubrics = rubrics
        self.logger.info(f"Initialized RubricGroup with {len(rubrics)} rubrics")

    def _get_reward_func_names(self) -> list[str]:
        names = []
        for rubric in self.rubrics:
            names.extend(rubric._get_reward_func_names())
        return names

    def _get_reward_funcs(self) -> list[RewardFunc]:
        funcs = []
        for rubric in self.rubrics:
            funcs.extend(rubric._get_reward_funcs())
        return funcs

    def _get_reward_weights(self) -> list[float]:
        weights = []
        for rubric in self.rubrics:
            weights.extend(rubric._get_reward_weights())
        return weights

    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        assert len(self.rubrics) > 0, "RubricGroup must have at least one rubric"
        self.logger.warning("Adding reward function to the first rubric in the group.")
        self.rubrics[0].add_reward_func(func, weight)

    async def score_rollout(self, state: State, score_sem: AsyncContextManager):
        """
        Evaluate all reward functions in-place for a single rollout.
        """
        total_reward = 0.0
        aggregated_metrics: dict[str, float] = {}
        original_reward = state.get("reward", 0.0)
        original_metrics = (
            state.get("metrics", {}).copy() if state.get("metrics") else {}
        )
        for rubric in self.rubrics:
            await rubric.score_rollout(state, score_sem=score_sem)
            rubric_reward = state.get("reward", 0.0)
            rubric_metrics = (
                state.get("metrics", {}).copy() if state.get("metrics") else {}
            )
            total_reward += rubric_reward
            for key, value in rubric_metrics.items():
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + value
            # restore original values for next rubric
            state["reward"] = original_reward
            state["metrics"] = original_metrics.copy()
        state["reward"] = total_reward
        state["metrics"] = aggregated_metrics

    async def score_group(self, states: list[State], score_sem: AsyncContextManager):
        """
        Evaluate all reward functions in-place for a group of rollouts.
        """
        aggregated_rewards = [0.0] * len(states)
        aggregated_metrics: dict[str, list[float]] = {}
        original_rewards = [state.get("reward", 0.0) for state in states]
        original_metrics = [
            state.get("metrics", {}).copy() if state.get("metrics") else {}
            for state in states
        ]
        for rubric in self.rubrics:
            await rubric.score_group(states, score_sem=score_sem)
            for i, state in enumerate(states):
                rubric_reward = state.get("reward", 0.0)
                rubric_metrics = (
                    state.get("metrics", {}).copy() if state.get("metrics") else {}
                )
                aggregated_rewards[i] += rubric_reward
                for key, value in rubric_metrics.items():
                    if key not in aggregated_metrics:
                        aggregated_metrics[key] = [0.0] * len(states)
                    aggregated_metrics[key][i] += value
                state["reward"] = original_rewards[i]
                state["metrics"] = original_metrics[i].copy()
        for i, state in enumerate(states):
            state["reward"] = aggregated_rewards[i]
            if aggregated_metrics:
                if "metrics" not in state:
                    state["metrics"] = {}
                for key, values in aggregated_metrics.items():
                    state["metrics"][key] = values[i]
