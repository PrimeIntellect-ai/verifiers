import asyncio
import inspect
import logging
import time
from typing import AsyncContextManager, cast

import verifiers as vf
from verifiers.types import (
    GroupRewardFunc,
    RewardFunc,
    RolloutScore,
    State,
)
from verifiers.utils.async_utils import maybe_await


class Rubric:
    """
    Rubric class for reward functions.

    Each reward function takes:
    - prompt: list[dict[str, str]] | str
    - completion: list[dict[str, str]] | str
    - answer: Any (metadata for scoring)
    - task (optional): str (type of task)
    - **kwargs: additional kwargs

    Returns:
    - float | list[float] | dict[str, float]
    """

    def __init__(
        self,
        funcs: list[RewardFunc | GroupRewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: vf.Parser | None = None,
    ):
        self.logger = logging.getLogger(f"verifiers.rubrics.{self.__class__.__name__}")

        self.funcs = funcs or []
        self.weights = weights or []
        if not self.weights:
            self.weights = [1.0] * len(self.funcs)
        elif len(self.weights) != len(self.funcs):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match number of functions ({len(self.funcs)})"
            )

        self.parser = parser or vf.Parser()

        # class objects for reward functions
        self.class_objects = {}
        if self.parser:
            self.class_objects["parser"] = self.parser

    # public helpers
    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        self.funcs.append(func)
        self.weights.append(weight)

    # private helpers
    def _get_reward_func_names(self) -> list[str]:
        return [func.__name__ for func in self.funcs]  # type: ignore[possibly-missing-attribute]

    def _get_reward_funcs(self) -> list[RewardFunc]:
        return [func for func in self.funcs]

    def _get_reward_weights(self) -> list[float]:
        return self.weights

    def _is_group_func(self, func: RewardFunc) -> bool:
        """Check if a function is a GroupRewardFunc by inspecting its signature."""
        sig = inspect.signature(func)
        # GroupRewardFunc has plural parameters: states, prompts, completions, etc.
        param_names = set(sig.parameters.keys())
        group_indicators = {
            "states",
            "prompts",
            "completions",
            "answers",
            "tasks",
            "infos",
        }
        returns_list = inspect.signature(func).return_annotation is list
        return bool(param_names & group_indicators) or returns_list

    # individual-level reward helpers
    def _get_individual_reward_func_names(self) -> list[str]:
        return [func.__name__ for func in self.funcs if not self._is_group_func(func)]  # type: ignore[possibly-missing-attribute]

    def _get_individual_reward_funcs(self) -> list[RewardFunc]:
        return [func for func in self.funcs if not self._is_group_func(func)]  # type: ignore[possibly-missing-attribute]

    def _get_individual_reward_weights(self) -> list[float]:
        return [
            weight
            for func, weight in zip(self.funcs, self.weights)
            if not self._is_group_func(func)
        ]

    async def _call_individual_reward_func(
        self,
        func: RewardFunc,
        state: State,
        score_sem: AsyncContextManager,
    ) -> float:
        """
        Invoke `func` with only the required arguments.

        Example:
        ```
        def func(completion, answer, **kwargs):
            ...
        ``
        """

        async def _call():
            sig = inspect.signature(func)

            merged = dict(
                prompt=state["prompt"],
                completion=state["completion"],
                answer=state.get("answer", ""),
                state=state,
                task=state["task"],
                info=state.get("info", {}),
            )
            merged.update(self.class_objects)
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                try:
                    ans = float(await maybe_await(func, **merged))
                except Exception as e:
                    self.logger.error(
                        f"Error calling reward function {func.__name__}: {e}"  # type: ignore[unresolved-attribute]
                    )
                    ans = 0.0
            else:
                allowed = {k: v for k, v in merged.items() if k in sig.parameters}
                try:
                    ans = float(await maybe_await(func, **allowed))
                except Exception as e:
                    self.logger.error(
                        f"Error calling reward function {func.__name__}: {e}"  # type: ignore[unresolved-attribute]
                    )
                    ans = 0.0
            return ans

        async with score_sem:
            return await _call()

    # group-level reward helpers
    def _get_group_reward_func_names(self) -> list[str]:
        return [func.__name__ for func in self.funcs if self._is_group_func(func)]  # type: ignore[possibly-missing-attribute]

    def _get_group_reward_funcs(self) -> list[GroupRewardFunc]:
        return [func for func in self.funcs if self._is_group_func(func)]  # type: ignore[possibly-missing-attribute]

    def _get_group_reward_weights(self) -> list[float]:
        return [
            weight
            for func, weight in zip(self.funcs, self.weights)
            if self._is_group_func(func)
        ]

    async def _call_group_reward_func(
        self,
        func: GroupRewardFunc,
        states: list[State],
        score_sem: AsyncContextManager,
    ) -> list[float]:
        """
        Invoke `func` with only the required arguments.
        """

        async def _call():
            sig = inspect.signature(func)
            merged = dict(
                prompts=[state["prompt"] for state in states],
                completions=[state["completion"] for state in states],
                answers=[state.get("answer", "") for state in states],
                states=states,
                tasks=[state["task"] for state in states],
                infos=[state.get("info", {}) for state in states],
            )
            merged.update(self.class_objects)
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                try:
                    ans = await maybe_await(func, **merged)
                except Exception as e:
                    self.logger.error(
                        f"Error calling group reward function {func.__name__}: {e}"  # type: ignore[unresolved-attribute]
                    )
                    ans = [0.0] * len(states)
            else:
                allowed = {k: v for k, v in merged.items() if k in sig.parameters}
                try:
                    ans = await maybe_await(func, **allowed)
                except Exception as e:
                    self.logger.error(
                        f"Error calling group reward function {func.__name__}: {e}"  # type: ignore[unresolved-attribute]
                    )
                    ans = [0.0] * len(states)
            return ans

        async with score_sem:
            return await _call()

    async def score_rollout(self, state: State, score_sem: AsyncContextManager):
        """
        Evaluate all reward functions for a single rollout.
        """
        reward_funcs = self._get_individual_reward_funcs()
        group_reward_funcs = self._get_group_reward_funcs()
        assert len(reward_funcs) > 0 and len(group_reward_funcs) == 0, (
            "Rubric.score_rollout requires at least one individual-level reward function and no group-level reward functions"
        )
        start_time = time.time()
        reward_scores = []
        for func in reward_funcs:
            reward_scores.append(
                await self._call_individual_reward_func(
                    func=func,
                    state=state,
                    score_sem=score_sem,
                )
            )
        rewards = RolloutScore(
            metrics={
                func.__name__: reward
                for func, reward in zip(reward_funcs, reward_scores)
            },
            reward=sum(
                [
                    reward * weight
                    for reward, weight in zip(
                        reward_scores, self._get_individual_reward_weights()
                    )
                ]
            ),
        )
        end_time = time.time()
        state["timing"]["scoring_ms"] = (end_time - start_time) * 1000
        state["timing"]["total_ms"] += state["timing"]["scoring_ms"]
        state["reward"] = rewards["reward"]
        state["metrics"] = rewards["metrics"]

    async def score_group(self, states: list[State], score_sem: AsyncContextManager):
        """
        Score a group of rollouts together.

        All reward functions are executed in order, parallelizing across states.
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
