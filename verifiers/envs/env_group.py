from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast, final

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import (
    ClientConfig,
    Messages,
    RolloutInput,
    SamplingArgs,
)
from verifiers.serve import EnvClient

if TYPE_CHECKING:
    from datasets import Dataset


ENV_GROUP_INFO_KEY = "env_id"


def _info_dict(info: object) -> dict[str, Any]:
    if info is None:
        return {}
    if isinstance(info, str):
        parsed = json.loads(info)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("RolloutInput info must decode to a dict for EnvGroup.")
    if isinstance(info, Mapping):
        return dict(cast(Mapping[str, Any], info))
    raise ValueError("RolloutInput info must be a dict for EnvGroup.")


def _normalize_route(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    if value is None:
        return ()
    raise ValueError("EnvGroup info['env_id'] must be a string or list of strings.")


def _route_value(route: tuple[str, ...]) -> str | list[str] | None:
    if not route:
        return None
    if len(route) == 1:
        return route[0]
    return list(route)


def _set_info_route(row: Mapping[str, Any], route: tuple[str, ...]) -> dict[str, Any]:
    routed = dict(row)
    info = _info_dict(routed.get("info"))
    info[ENV_GROUP_INFO_KEY] = _route_value(route)
    routed["info"] = info
    return routed


class EnvGroupRubric(vf.Rubric):
    """
    Custom rubric for EnvGroup that routes scoring to appropriate environment rubrics.
    """

    def __init__(self, env_map: Mapping[str, vf.Environment]):
        super().__init__()
        self.env_map = env_map

        # Collect all unique reward function names across all environments
        all_names_set = set()
        for env in env_map.values():
            all_names_set.update(env.rubric._get_reward_func_names())
        self.all_reward_names = sorted(list(all_names_set))

        self.logger.info(
            f"EnvGroupRubric tracking {len(self.all_reward_names)} unique reward functions"
        )

    def _get_reward_func_names(self) -> list[str]:
        """Return all unique reward function names across all environments."""
        return self.all_reward_names

    async def score_rollout(
        self,
        state: vf.State,
    ) -> None:
        """
        Evaluate all reward functions in-place for a single rollout.

        Routes scoring to the appropriate environment's rubric based on info["env_id"].
        """
        route = _normalize_route((state.get("info") or {}).get(ENV_GROUP_INFO_KEY))
        env_name = route[0] if route else None
        metrics = {name: 0.0 for name in self.all_reward_names}
        reward = 0.0

        # get the appropriate environment
        env = self.env_map.get(env_name) if env_name is not None else None
        if env is None:
            self.logger.warning(f"No environment found for EnvGroup route '{env_name}'")
            state["reward"] = reward
            state["metrics"] = metrics
            return

        await env.rubric.score_rollout(state)
        env_reward = state.get("reward", 0.0)
        env_metrics = state.get("metrics", {}).copy() if state.get("metrics") else {}

        for reward_name, score in env_metrics.items():
            if reward_name in metrics:
                metrics[reward_name] = score

        reward = env_reward
        state["reward"] = reward
        state["metrics"] = metrics

    async def score_group(
        self,
        states: list[vf.State],
    ) -> None:
        """
        Score a group of rollouts, routing to appropriate environment rubrics based on info["env_id"].

        All states in a group have the same environment route, so we route once to the appropriate
        environment's rubric. Ensures all states have metrics for all reward function names
        across all environments.
        """
        num_states = len(states)
        route = _normalize_route((states[0].get("info") or {}).get(ENV_GROUP_INFO_KEY))
        env_name = route[0] if route else None
        env = self.env_map.get(env_name) if env_name is not None else None
        if env is None:
            self.logger.warning(f"No environment found for EnvGroup route '{env_name}'")
            for state in states:
                state["reward"] = 0.0
                state["metrics"] = {name: 0.0 for name in self.all_reward_names}
            return

        # Score all states using the environment's rubric
        await env.rubric.score_group(states)

        # Initialize metrics dict with all reward function names
        aggregated_metrics: dict[str, list[float]] = {
            name: [0.0] * num_states for name in self.all_reward_names
        }

        # Extract metrics from each state and ensure all reward function names are present
        for i, state in enumerate(states):
            env_metrics = state.get("metrics", {}) or {}
            for reward_name, score in env_metrics.items():
                if reward_name in aggregated_metrics:
                    aggregated_metrics[reward_name][i] = score

        for i, state in enumerate(states):
            state["metrics"] = {
                func_name: values[i] for func_name, values in aggregated_metrics.items()
            }

    async def cleanup(self, state: vf.State) -> None:
        route = _normalize_route((state.get("info") or {}).get(ENV_GROUP_INFO_KEY))
        env_name = route[0] if route else None
        env = self.env_map.get(env_name) if env_name is not None else None
        if env is not None:
            await env.rubric.cleanup(state)
        await super().cleanup(state)


class EnvGroup(vf.Environment):
    """
    Environment group that acts as a mixture of multiple environments.

    Routes operations to sub-environments based on ``info["env_id"]``.
    """

    def __init__(
        self,
        envs: list[vf.Environment],
        env_names: list[str] | None = None,
        map_kwargs: dict = {},
        **kwargs,
    ):
        """
        Initialize EnvGroup with a list of environments.

        Args:
            envs: list of Environment instances
            env_names: Optional list of names for each environment.
                      If not provided, uses "env_0", "env_1", etc.
            **kwargs: Additional arguments passed to parent Environment
        """
        from datasets import concatenate_datasets

        if not envs:
            raise ValueError("EnvGroup requires at least one environment")

        self.envs = envs
        self.env_names = env_names or [f"env_{i}" for i in range(len(envs))]

        if len(self.env_names) != len(self.envs):
            raise ValueError("Number of env_names must match number of envs")

        # create mapping for quick lookup
        self.env_map = {name: env for name, env in zip(self.env_names, self.envs)}

        # concatenate datasets and add EnvGroup routing metadata under info["env_id"]
        datasets = []
        eval_datasets = []

        def make_add_env_route_fn(env_name: str):
            def add_env_route(example):
                info = _info_dict(example.get("info"))
                child_route = _normalize_route(info.get(ENV_GROUP_INFO_KEY))
                route = (env_name, *child_route)
                info[ENV_GROUP_INFO_KEY] = _route_value(route)
                example["info"] = info
                return example

            return add_env_route

        for env, name in zip(self.envs, self.env_names):
            add_env_route = make_add_env_route_fn(name)

            # Build dataset if using DatasetBuilder, returns None if not available
            env_dataset = env.build_dataset()
            if env_dataset is not None:
                remove_cols = [
                    col for col in ("env_id",) if col in env_dataset.column_names
                ]
                if remove_cols:
                    env_dataset = env_dataset.remove_columns(remove_cols)
                env_dataset = env_dataset.map(add_env_route, **map_kwargs)
                datasets.append(env_dataset)
            # Build eval_dataset if using DatasetBuilder, returns None if not available
            env_eval_dataset = env.build_eval_dataset()
            if env_eval_dataset is not None:
                remove_cols = [
                    col for col in ("env_id",) if col in env_eval_dataset.column_names
                ]
                if remove_cols:
                    env_eval_dataset = env_eval_dataset.remove_columns(remove_cols)
                env_eval_dataset = env_eval_dataset.map(add_env_route, **map_kwargs)
                eval_datasets.append(env_eval_dataset)
        dataset = concatenate_datasets(datasets) if datasets else None
        eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None
        # wrap rubrics in EnvGroupRubric
        rubric = EnvGroupRubric(self.env_map)

        # don't set tool_defs at the group level since different sub-environments
        # may have different tools. Instead, set them per-task in rollout().
        # initialize parent Environment
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            tool_defs=None,
            map_kwargs=map_kwargs,
            **kwargs,
        )
        self.logger.info(
            f"Initialized EnvGroup with {len(envs)} environments: {self.env_names}"
        )

    def _format_dataset(
        self,
        dataset: Dataset,
        system_prompt: str | None = None,
        few_shot: Messages | None = None,
        question_key: str = "question",
        answer_key: str = "answer",
        map_kwargs: dict = {},
    ) -> Dataset:
        """Ensure unique example_ids across concatenated datasets."""
        # use parent's prompt handling
        dataset = self._ensure_prompt(
            dataset, system_prompt, few_shot, question_key, answer_key, map_kwargs
        )

        # ensure unique example_ids across concatenated datasets
        if "example_id" in dataset.column_names:
            dataset = dataset.remove_columns(["example_id"])

        def add_example_id(example, i):
            example["example_id"] = i
            return example

        dataset = dataset.map(add_example_id, with_indices=True, **map_kwargs)

        assert "example_id" in dataset.column_names
        assert "prompt" in dataset.column_names
        return dataset

    def _format_completion_dataset(
        self, dataset: Dataset, map_kwargs: dict = {}
    ) -> Dataset:
        """Ensure unique example_ids across concatenated datasets."""
        # ensure unique example_ids across concatenated datasets
        if "example_id" in dataset.column_names:
            dataset = dataset.remove_columns(["example_id"])

        def add_example_id(example, i):
            example["example_id"] = i
            return example

        dataset = dataset.map(add_example_id, with_indices=True, **map_kwargs)
        assert "example_id" in dataset.column_names
        return dataset

    @final
    async def run_rollout(  # type: ignore[override]
        self,
        input: RolloutInput,
        client: Client | ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
        env_client: EnvClient | None = None,
    ) -> vf.RolloutOutput:
        env_name, child_input, route = self._route_child_input(input)
        env = self.get_env_for_name(env_name)
        env_client = env_client or env.env_client or self.env_client
        output = await env.run_rollout(
            child_input,
            client,
            model,
            sampling_args,
            max_retries,
            state_columns,
            env_client,
        )
        return _set_info_route(output, route)  # type: ignore[return-value]

    @final
    async def run_group(  # type: ignore[override]
        self,
        group_inputs: list[RolloutInput],
        client: Client | ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
        env_client: EnvClient | None = None,
    ) -> list[vf.RolloutOutput]:
        env_name, first_child_input, route = self._route_child_input(group_inputs[0])
        child_inputs = [first_child_input]
        for group_input in group_inputs[1:]:
            input_env_name, child_input, input_route = self._route_child_input(
                group_input
            )
            if input_env_name != env_name:
                raise ValueError(
                    "All EnvGroup inputs in a group must route to the same environment."
                )
            if input_route != route:
                raise ValueError(
                    "All EnvGroup inputs in a group must have the same route."
                )
            child_inputs.append(child_input)
        env = self.get_env_for_name(env_name)
        env_client = env_client or env.env_client or self.env_client
        outputs = await env.run_group(
            child_inputs,
            client,
            model,
            sampling_args,
            max_retries,
            state_columns,
            env_client,
        )
        return [_set_info_route(output, route) for output in outputs]  # type: ignore[return-value]

    @final
    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> vf.State:
        env_name, child_input, route = self._route_child_input(input)
        env = self.get_env_for_name(env_name)
        state = await env.rollout(child_input, client, model, sampling_args)
        info = _info_dict(state.get("info"))
        info[ENV_GROUP_INFO_KEY] = _route_value(route)
        state["info"] = info
        return state

    def _route_child_input(
        self, input: RolloutInput
    ) -> tuple[str, RolloutInput, tuple[str, ...]]:
        route = self._input_env_route(input)
        env_name = route[0]
        remaining = route[1:]
        child_input = dict(input)
        info = _info_dict(child_input.get("info"))
        if remaining:
            info[ENV_GROUP_INFO_KEY] = _route_value(remaining)
        else:
            info.pop(ENV_GROUP_INFO_KEY, None)
        if info:
            child_input["info"] = info
        else:
            child_input.pop("info", None)
        return env_name, child_input, route  # type: ignore[return-value]

    def _input_env_route(self, input: RolloutInput) -> tuple[str, ...]:
        info = _info_dict(input.get("info"))
        route = _normalize_route(info.get(ENV_GROUP_INFO_KEY))
        if route:
            return route
        if len(self.envs) == 1:
            return (self.env_names[0],)
        raise ValueError(
            "EnvGroup input is missing info['env_id']; use rows from "
            "EnvGroup.get_dataset()/get_eval_dataset() or provide that routing value."
        )

    def get_env_for_name(self, name: str) -> vf.Environment:
        if name not in self.env_map:
            raise ValueError(f"No environment found for info['env_id']={name!r}")
        return self.env_map[name]

    def set_max_seq_len(self, max_seq_len: int | None) -> None:
        """Set the max_seq_len value for this environment group and all sub-environments."""
        self.max_seq_len = max_seq_len
        for env in self.envs:
            env.set_max_seq_len(max_seq_len)

    def set_score_rollouts(self, score_rollouts: bool) -> None:
        """Set the score_rollouts flag for this environment group and all sub-environments."""
        self.score_rollouts = score_rollouts
        for env in self.envs:
            env.set_score_rollouts(score_rollouts)
