import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any, cast, get_origin

import verifiers as vf
from verifiers.decorators import discover_decorated
from verifiers.types import (
    GroupRewardFunc,
    MultiAgentRewardFunc,
    RewardFunc,
    RolloutScore,
    State,
    TASK_INPUT_FIELDS,
)
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.async_utils import maybe_call_with_named_args

ScoreObjectProvider = Callable[[State], Mapping[str, object]]
GroupScoreObjectProvider = Callable[[list[State]], Mapping[str, object]]


class Rubric:
    """
    Rubric class for reward functions.

    Each reward function takes:
    - prompt: list[dict[str, str]] | str
    - completion: list[dict[str, str]] | str
    - answer: Any (metadata for scoring)
    - task (optional): vf.Task for taskset-backed environments
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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

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
        self.score_object_providers: list[ScoreObjectProvider] = []
        self.group_score_object_providers: list[GroupScoreObjectProvider] = []

        self._cleanup_handlers = discover_decorated(self, "cleanup")
        self._teardown_handlers = discover_decorated(self, "teardown")

    # public helpers
    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        self.funcs.append(func)
        self.weights.append(weight)

    def add_metric(self, func: RewardFunc, weight: float = 0.0):
        self.funcs.append(func)
        self.weights.append(weight)

    def add_class_object(self, name: str, obj: Any):
        self.class_objects[name] = obj

    def add_score_object_provider(self, provider: ScoreObjectProvider):
        self.score_object_providers.append(provider)

    def add_group_score_object_provider(self, provider: GroupScoreObjectProvider):
        self.group_score_object_providers.append(provider)

    def add_cleanup_handler(self, handler: Callable[..., Any]) -> None:
        self._cleanup_handlers.append(handler)
        self._cleanup_handlers.sort(
            key=lambda h: (
                -getattr(h, "cleanup_priority", 0),
                str(getattr(h, "__name__", "")),
            )
        )

    # private helpers
    def _get_reward_func_names(self) -> list[str]:
        return [getattr(func, "__name__", repr(func)) for func in self.funcs]

    def _get_reward_funcs(self) -> list[RewardFunc]:
        return [func for func in self.funcs]

    def _get_reward_weights(self) -> list[float]:
        return self.weights

    def _is_group_func(self, func: RewardFunc) -> bool:
        """Check if a function is a GroupRewardFunc by inspecting its signature."""
        if self._is_multiagent_func(func):
            return False
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
        return_annotation = inspect.signature(func).return_annotation
        returns_list = (
            return_annotation is list or get_origin(return_annotation) is list
        )
        return bool(param_names & group_indicators) or returns_list

    def score_objects(self, state: State) -> dict[str, Any]:
        task = self.task_for_state(state, self.class_objects.get("resources"))
        objects = self.task_score_fields(state, task)
        objects.update(
            {
                "prompt": state["prompt"],
                "completion": state["completion"],
                "answer": state.get("answer", ""),
                "state": state,
                "info": state.get("info", {}),
                **self.class_objects,
            }
        )
        objects["task"] = task
        for provider in self.score_object_providers:
            objects.update(provider(state))
        return objects

    def task_score_fields(self, state: State, task: object) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        input_data = state.get("input")
        if isinstance(input_data, Mapping):
            row = cast(Mapping[str, Any], input_data)
            fields.update(
                {
                    key: value
                    for key, value in row.items()
                    if key not in TASK_INPUT_FIELDS
                }
            )
        if isinstance(task, Mapping):
            fields.update(cast(Mapping[str, Any], task))
        return fields

    def group_score_objects(self, states: list[State]) -> dict[str, Any]:
        state_objects = [self.score_objects(state) for state in states]
        objects = dict(
            prompts=[state["prompt"] for state in states],
            completions=[state["completion"] for state in states],
            answers=[state.get("answer", "") for state in states],
            states=states,
            tasks=[state_object.get("task") for state_object in state_objects],
            infos=[state_object.get("info", {}) for state_object in state_objects],
            **self.class_objects,
        )
        for provider in self.group_score_object_providers:
            objects.update(provider(states))
        return objects

    def task_for_state(self, state: State, resources: object | None) -> object:
        if "task" in state:
            return state.get("task")
        taskset = getattr(resources, "taskset", None)
        to_task = getattr(taskset, "to_task", None)
        if callable(to_task) and "input" in state:
            return to_task(state["input"])
        return None

    def _is_multiagent_func(self, func: RewardFunc) -> bool:
        """Check if a function is a MultiAgentRewardFunc by inspecting its return annotation."""
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation
        return return_annotation is dict or get_origin(return_annotation) is dict

    # individual-level reward helpers
    def _get_individual_reward_func_names(self) -> list[str]:
        return [
            getattr(func, "__name__", repr(func))
            for func in self.funcs
            if not self._is_group_func(func) and not self._is_multiagent_func(func)
        ]

    def _get_individual_reward_funcs(self) -> list[RewardFunc]:
        return [
            func
            for func in self.funcs
            if not self._is_group_func(func) and not self._is_multiagent_func(func)
        ]

    def _get_individual_reward_weights(self) -> list[float]:
        return [
            weight
            for func, weight in zip(self.funcs, self.weights)
            if not self._is_group_func(func) and not self._is_multiagent_func(func)
        ]

    async def _call_individual_reward_func(
        self,
        func: RewardFunc,
        state: State,
    ) -> float:
        """
        Invoke `func` with only the required arguments.

        Example:
        ```
        def func(completion, answer, **kwargs):
            ...
        ``
        """

        sig = inspect.signature(func)
        merged = self.score_objects(state)
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

    # group-level reward helpers
    def _get_group_reward_func_names(self) -> list[str]:
        return [
            getattr(func, "__name__", repr(func))
            for func in self.funcs
            if self._is_group_func(func)
        ]

    def _get_group_reward_funcs(self) -> list[GroupRewardFunc]:
        return cast(
            list[GroupRewardFunc],
            [func for func in self.funcs if self._is_group_func(func)],
        )

    @property
    def has_group_rewards(self) -> bool:
        return bool(self._get_group_reward_funcs())

    @property
    def has_advantages(self) -> bool:
        return False

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
    ) -> list[float]:
        """
        Invoke `func` with only the required arguments.
        """

        sig = inspect.signature(func)
        merged = self.group_score_objects(states)
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

    # multi-agent reward helpers
    def _get_multiagent_reward_funcs(self) -> list[MultiAgentRewardFunc]:
        return cast(
            list[MultiAgentRewardFunc],
            [func for func in self.funcs if self._is_multiagent_func(func)],
        )

    async def _call_multiagent_reward_func(
        self,
        func: MultiAgentRewardFunc,
        state: State,
    ) -> dict[str, float]:
        """Invoke a multi-agent reward function that returns per-agent rewards."""
        sig = inspect.signature(func)
        merged = self.score_objects(state)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            try:
                ans = await maybe_await(func, **merged)
            except Exception as e:
                self.logger.error(
                    f"Error calling multi-agent reward function {func.__name__}: {e}"
                )
                ans = {}
        else:
            allowed = {k: v for k, v in merged.items() if k in sig.parameters}
            try:
                ans = await maybe_await(func, **allowed)
            except Exception as e:
                self.logger.error(
                    f"Error calling multi-agent reward function {func.__name__}: {e}"
                )
                ans = {}
        return ans

    async def cleanup(self, state: State):
        """Run all @vf.cleanup-decorated methods on this rubric."""
        for handler in self._cleanup_handlers:
            await self._call_cleanup_handler(handler, state)

    async def _call_cleanup_handler(self, handler: Callable[..., Any], state: State):
        objects = self.cleanup_objects(handler, state)
        await maybe_call_with_named_args(handler, **objects)

    def cleanup_objects(
        self, handler: Callable[..., Any], state: State
    ) -> dict[str, Any]:
        sig = inspect.signature(handler)
        parameters = sig.parameters.values()
        wants_kwargs = any(p.kind == p.VAR_KEYWORD for p in parameters)
        requested = {
            name
            for name, parameter in sig.parameters.items()
            if parameter.kind
            in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        }
        objects: dict[str, Any] = {"state": state, **self.class_objects}
        known = {"state", "prompt", "completion", "answer", "info", "task"} | set(
            self.class_objects
        )
        if wants_kwargs or "prompt" in requested:
            objects["prompt"] = state.get("prompt")
        if wants_kwargs or "completion" in requested:
            objects["completion"] = state.get("completion")
        if wants_kwargs or "answer" in requested:
            objects["answer"] = state.get("answer", "")
        if wants_kwargs or "info" in requested:
            objects["info"] = state.get("info", {})
        if wants_kwargs or "task" in requested:
            objects["task"] = self.task_for_state(state, objects.get("resources"))
        if wants_kwargs or not requested <= known:
            for name, value in self.task_score_fields(
                state, objects.get("task")
            ).items():
                objects.setdefault(name, value)
            for provider in self.score_object_providers:
                objects.update(provider(state))
        return objects

    async def teardown(self):
        """Run all @vf.teardown-decorated methods on this rubric."""
        for handler in self._teardown_handlers:
            await handler()

    async def dummy_score_rollout(self, state: State):
        """Score a single rollout with dummy rewards."""
        state["reward"] = 0.0
        state["metrics"] = {}

    async def score_rollout(self, state: State):
        """
        Evaluate all reward functions for a single rollout.
        """
        reward_funcs = self._get_individual_reward_funcs()
        group_reward_funcs = self._get_group_reward_funcs()
        assert len(reward_funcs) > 0 and len(group_reward_funcs) == 0, (
            "Rubric.score_rollout requires at least one individual-level reward function and no group-level reward functions"
        )
        reward_scores = []
        for func in reward_funcs:
            reward_scores.append(
                await self._call_individual_reward_func(
                    func=func,
                    state=state,
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
        state["reward"] = rewards["reward"]
        state["metrics"] = rewards["metrics"]

    async def dummy_score_group(self, states: list[State]):
        """Score a group of rollouts together with dummy rewards."""
        for state in states:
            await self.dummy_score_rollout(state)

    async def score_group(self, states: list[State]):
        """
        Score a group of rollouts together.

        All reward functions are executed in order, parallelizing across states.
        Supports multi-agent reward functions that return per-agent rewards.
        """
        num_states = len(states)
        if num_states == 0:
            self.logger.warning("No states to score")
            return
        aggregated_rewards = [0.0] * num_states
        # Per-agent rewards for multi-agent envs: list of dict[agent_id, reward]
        aggregated_agent_rewards: list[dict[str, float]] = [
            {} for _ in range(num_states)
        ]
        aggregated_metrics: dict[str, list[float]] = {}

        # process functions in order
        for func, weight in zip(self.funcs, self.weights):
            is_group = self._is_group_func(func)
            is_multiagent = self._is_multiagent_func(func)

            if is_multiagent:
                # MultiAgentRewardFunc: returns dict[str, float] per state
                multiagent_func = cast(MultiAgentRewardFunc, func)
                score_tasks = [
                    self._call_multiagent_reward_func(multiagent_func, state)
                    for state in states
                ]
                agent_scores_list = await asyncio.gather(*score_tasks)

                func_name = func.__name__
                for i, agent_scores in enumerate(agent_scores_list):
                    # Aggregate per-agent rewards
                    for agent_id, score_value in agent_scores.items():
                        if agent_id not in aggregated_agent_rewards[i]:
                            aggregated_agent_rewards[i][agent_id] = 0.0
                        aggregated_agent_rewards[i][agent_id] += score_value * weight
                    # Also compute a rollout-level reward (mean of agent rewards)
                    if agent_scores:
                        mean_score = sum(agent_scores.values()) / len(agent_scores)
                        aggregated_rewards[i] += mean_score * weight
                        if func_name not in aggregated_metrics:
                            aggregated_metrics[func_name] = [0.0] * num_states
                        aggregated_metrics[func_name][i] = mean_score
            elif is_group:
                # GroupRewardFunc: score all states together
                group_func = cast(GroupRewardFunc, func)
                scores = await self._call_group_reward_func(group_func, states)
                func_name = func.__name__
                if func_name not in aggregated_metrics:
                    aggregated_metrics[func_name] = [0.0] * num_states
                for i in range(num_states):
                    score_value = scores[i]
                    aggregated_rewards[i] += score_value * weight
                    aggregated_metrics[func_name][i] = score_value
                    # Also add to each agent's rewards (for multi-agent compatibility)
                    agent_ids = set(
                        t.get("extras", {}).get("agent_id")
                        for t in states[i]["trajectory"]
                        if t.get("extras", {}).get("agent_id")
                    )
                    for agent_id in agent_ids:
                        if agent_id not in aggregated_agent_rewards[i]:
                            aggregated_agent_rewards[i][agent_id] = 0.0
                        aggregated_agent_rewards[i][agent_id] += score_value * weight
            else:
                reward_func = cast(RewardFunc, func)
                score_tasks = [
                    self._call_individual_reward_func(reward_func, state)
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
                    # Also add to each agent's rewards (for multi-agent compatibility)
                    agent_ids = set(
                        t.get("extras", {}).get("agent_id")
                        for t in states[i]["trajectory"]
                        if t.get("extras", {}).get("agent_id")
                    )
                    for agent_id in agent_ids:
                        if agent_id not in aggregated_agent_rewards[i]:
                            aggregated_agent_rewards[i][agent_id] = 0.0
                        aggregated_agent_rewards[i][agent_id] += score_value * weight

        avg_reward = sum(aggregated_rewards) / num_states
        for i, state in enumerate(states):
            state["reward"] = aggregated_rewards[i]
            state["advantage"] = aggregated_rewards[i] - avg_reward

            # Store per-agent rewards if any multi-agent funcs were used
            agent_rewards = aggregated_agent_rewards[i]
            if agent_rewards:
                state["agent_rewards"] = agent_rewards

            # Assign per-step rewards and advantages based on agent_id (for multi-agent)
            for t in state["trajectory"]:
                if t["reward"] is None:
                    if agent_rewards:
                        agent_id = t.get("extras", {}).get("agent_id")
                        t["reward"] = agent_rewards.get(agent_id, state["reward"])
                    else:
                        t["reward"] = state["reward"]
                # Compute per-agent advantage: agent's reward - shared baseline
                if t["advantage"] is None:
                    t["advantage"] = t["reward"] - avg_reward

            state["metrics"] = {
                func_name: values[i] for func_name, values in aggregated_metrics.items()
            }
