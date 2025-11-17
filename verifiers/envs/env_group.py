import time
from typing import TYPE_CHECKING, AsyncContextManager, Mapping

from datasets import Dataset, concatenate_datasets
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import RolloutInput, SamplingArgs

if TYPE_CHECKING:
    pass


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
        score_sem: AsyncContextManager,
    ) -> None:
        """
        Evaluate all reward functions in-place for a single rollout.

        Routes scoring to the appropriate environment's rubric based on task.
        """
        task = state.get("task", "default")
        metrics = {name: 0.0 for name in self.all_reward_names}
        reward = 0.0

        # get the appropriate environment
        env = self.env_map.get(task)
        if env is None:
            self.logger.warning(f"No environment found for task '{task}'")
            state["reward"] = reward
            state["metrics"] = metrics
            return

        await env.rubric.score_rollout(state, score_sem=score_sem)
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
        score_sem: AsyncContextManager,
    ) -> None:
        """
        Score a group of rollouts, routing to appropriate environment rubrics based on task.

        All states in a group have the same task, so we route once to the appropriate
        environment's rubric. Ensures all states have metrics for all reward function names
        across all environments.
        """
        start_time = time.time()
        num_states = len(states)
        # get task from first state (all states in a group have the same task)
        task = states[0].get("task", "default")
        env = self.env_map.get(task)
        if env is None:
            self.logger.warning(f"No environment found for task '{task}'")
            for state in states:
                state["reward"] = 0.0
                state["metrics"] = {name: 0.0 for name in self.all_reward_names}
                state["timing"]["scoring_ms"] = 0.0
            return

        # Score all states using the environment's rubric
        await env.rubric.score_group(states, score_sem=score_sem)

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

        # Update all states with aggregated metrics (ensuring all reward names are present)
        end_time = time.time()
        scoring_ms = (end_time - start_time) * 1000
        for i, state in enumerate(states):
            state["metrics"] = {
                func_name: values[i] for func_name, values in aggregated_metrics.items()
            }
            state["timing"]["scoring_ms"] = scoring_ms
            state["timing"]["total_ms"] += state["timing"]["scoring_ms"]


class EnvGroup(vf.Environment):
    """
    Environment group that acts as a mixture of multiple environments.

    Routes operations to appropriate sub-environments based on the 'task' column.
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
        if not envs:
            raise ValueError("EnvGroup requires at least one environment")

        self.envs = envs
        self.env_names = env_names or [f"env_{i}" for i in range(len(envs))]

        if len(self.env_names) != len(self.envs):
            raise ValueError("Number of env_names must match number of envs")

        # create mapping for quick lookup
        self.env_map = {name: env for name, env in zip(self.env_names, self.envs)}

        # concatenate datasets - override task column to use env_names for routing
        datasets = []
        eval_datasets = []
        for env, name in zip(self.envs, self.env_names):

            def add_task(example):
                example["task"] = name
                return example

            env_dataset = env.get_dataset()
            if env_dataset is not None:
                # override task column to use env_name for routing
                if "task" in env_dataset.column_names:
                    env_dataset = env_dataset.remove_columns(["task"])
                env_dataset = env_dataset.map(add_task, **map_kwargs)
                datasets.append(env_dataset)
            env_eval_dataset = env.get_eval_dataset()
            if env_eval_dataset is not None:
                # override task column to use env_name for routing
                if "task" in env_eval_dataset.column_names:
                    env_eval_dataset = env_eval_dataset.remove_columns(["task"])
                env_eval_dataset = env_eval_dataset.map(add_task, **map_kwargs)
                eval_datasets.append(env_eval_dataset)
        dataset = concatenate_datasets(datasets) if datasets else None
        eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None
        # wrap rubrics in EnvGroupRubric
        rubric = EnvGroupRubric(self.env_map)

        # don't set oai_tools at the group level since different sub-environments
        # may have different tools. Instead, set them per-task in rollout().
        # initialize parent Environment
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            oai_tools=None,
            map_kwargs=map_kwargs,
            **kwargs,
        )
        self.logger.info(
            f"Initialized EnvGroup with {len(envs)} environments: {self.env_names}"
        )

    def format_dataset(
        self,
        dataset: Dataset,
        system_prompt: str | None = None,
        few_shot: vf.ChatMessages | None = None,
        question_key: str = "question",
        answer_key: str = "answer",
        map_kwargs: dict = {},
    ) -> Dataset:
        """
        Ensure unique example_ids and mapped tasks across concatenated datasets.
        """
        # use parent's prompt handling
        dataset = self._ensure_prompt(
            dataset, system_prompt, few_shot, question_key, answer_key, map_kwargs
        )
        # task is already set during concatenation, so skip _ensure_task

        # ensure unique example_ids across concatenated datasets
        if "example_id" in dataset.column_names:
            dataset = dataset.remove_columns(["example_id"])

        def add_example_id(example, i):
            example["example_id"] = i
            return example

        dataset = dataset.map(add_example_id, with_indices=True, **map_kwargs)

        assert "example_id" in dataset.column_names
        assert "prompt" in dataset.column_names
        assert "task" in dataset.column_names, (
            "Task column should be set during concatenation in __init__"
        )
        return dataset

    async def init_state(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> vf.State:
        env = self.get_env_for_task(input["task"])
        return await env.init_state(input, client, model, sampling_args)

    async def setup_state(self, state: vf.State) -> vf.State:
        env = self.get_env_for_task(state["task"])
        return await env.setup_state(state)

    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> vf.State:
        env = self.get_env_for_task(input["task"])
        return await env.rollout(input, client, model, sampling_args)

    def get_env_for_task(self, task: str) -> vf.Environment:
        return self.env_map.get(task, self.envs[0])

    def set_max_seq_len(self, max_seq_len: int | None) -> None:
        """Set the maximum sequence length for this environment group and all sub-environments."""
        self.max_seq_len = max_seq_len
        for env in self.envs:
            env.set_max_seq_len(max_seq_len)
