from collections import defaultdict
from typing import TYPE_CHECKING, Mapping

from datasets import Dataset, concatenate_datasets
from openai import AsyncOpenAI

from verifiers import (
    ChatMessage,
    Info,
    Messages,
    SamplingArgs,
    State,
)
from verifiers.envs.environment import Environment
from verifiers.rubrics.rubric import Rubric
from verifiers.types import MessageType, ProcessedOutputs, RolloutScore

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class EnvGroupRubric(Rubric):
    """
    Custom rubric for EnvGroup that routes scoring to appropriate environment rubrics.
    """

    def __init__(self, env_map: Mapping[str, Environment]):
        super().__init__()
        self.env_map = env_map

        # Collect all unique reward function names across all environments
        all_names_set = set()
        for env in env_map.values():
            all_names_set.update(env.rubric.get_reward_func_names())
        self.all_reward_names = sorted(list(all_names_set))

        self.logger.info(
            f"EnvGroupRubric tracking {len(self.all_reward_names)} unique reward functions"
        )

    def get_reward_func_names(self) -> list[str]:
        """Return all unique reward function names across all environments."""
        return self.all_reward_names

    async def score_rollout(
        self,
        prompt: str | list[ChatMessage],
        completion: str | list[ChatMessage],
        answer: str = "",
        state: State | None = None,
        task: str = "default",
        info: dict | None = None,
        example_id: int | None = None,
        **kwargs,
    ) -> RolloutScore:
        """
        Route scoring to the appropriate environment's rubric based on task.

        Returns a RolloutScore with all reward function names, using 0.0 for functions
        not applicable to this sample's environment.
        """
        state = state or {}
        info = info if info is not None else {}

        # Initialize metrics with all reward names set to 0.0
        metrics = {name: 0.0 for name in self.all_reward_names}
        reward = 0.0

        # Get the appropriate environment
        env = self.env_map.get(task)
        if env is None:
            self.logger.warning(f"No environment found for task '{task}'")
            return RolloutScore(reward=reward, metrics=metrics)

        # Score with the environment's rubric
        env_results = await env.rubric.score_rollout(
            prompt, completion, answer, state, task, info, example_id, **kwargs
        )

        # Update metrics with individual metric scores from the environment
        for reward_name, score in env_results.metrics.items():
            if reward_name in metrics:
                metrics[reward_name] = score

        # The overall reward is from the environment's rubric
        reward = env_results.reward

        return RolloutScore(reward=reward, metrics=metrics)


class EnvGroup(Environment):
    """
    Environment group that acts as a mixture of multiple environments.

    Routes operations to appropriate sub-environments based on the 'task' column.
    """

    def __init__(
        self,
        envs: list[Environment],
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

        # Create mapping for quick lookup
        self.env_map = {name: env for name, env in zip(self.env_names, self.envs)}

        # concatenate datasets with task labels
        datasets = []
        eval_datasets = []
        for env, name in zip(self.envs, self.env_names):

            def add_task(example):
                example["task"] = name
                return example

            env_dataset = env.get_dataset()
            if env_dataset is not None:
                env_dataset = env_dataset.map(add_task, **map_kwargs)
                datasets.append(env_dataset)
            env_eval_dataset = env.get_eval_dataset()
            if env_eval_dataset is not None:
                env_eval_dataset = env_eval_dataset.map(add_task, **map_kwargs)
                eval_datasets.append(env_eval_dataset)
        dataset = concatenate_datasets(datasets) if datasets else None
        eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None
        # wrap rubrics
        rubric = EnvGroupRubric(self.env_map)

        # Don't set oai_tools at the group level since different sub-environments
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
        few_shot: list[ChatMessage] | None = None,
        question_key: str = "question",
        answer_key: str = "answer",
        map_kwargs: dict = {},
    ) -> Dataset:
        """
        Ensures that the (eval) dataset of an env group has unique example ids.

        Explanation: Each individual env creates example id column unique to the
        env upon init. However, these are not unique across splits (e.g. they
        will have an example id `0`) Upon initting the env group, this method is
        called (via super().__init__) and forcefully removes the example id
        column in the concatenated dataset and replaces it with a new one with
        globally unique example ids.
        """

        # Remove the example_id column present in the individual env datasets and add global ids
        if "example_id" in dataset.column_names:
            dataset = dataset.remove_columns(["example_id"])

        def add_example_id(example, i):
            example["example_id"] = i
            return example

        dataset = dataset.map(add_example_id, with_indices=True, **map_kwargs)

        assert "example_id" in dataset.column_names
        assert "prompt" in dataset.column_names
        return dataset

    async def init_state(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        task: str,
        info: Info,
        example_id: int,
        **kwargs,
    ) -> State:
        """
        Initialize state for a rollout.
        """
        env = self.env_map.get(task)
        if env and hasattr(env, "oai_tools") and env.oai_tools:
            if "oai_tools" not in info:
                info["oai_tools"] = env.oai_tools
        return await super().init_state(
            prompt, completion, answer, task, info, example_id
        )

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        completion: Messages | None = None,
        answer: str = "",
        state: State = {},
        task: str = "default",
        info: Info | None = None,
        example_id: int = 0,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        """
        Route rollout to the appropriate sub-environment based on task.

        The task is determined from (in order of priority):
        1. kwargs['task']
        2. info['task']
        3. First environment name (default)
        """
        info = info if info is not None else {}
        sampling_args = sampling_args or {}

        # Route to appropriate environment
        env = self.env_map[task]

        # Set tools for this task's environment if not already set in info
        if "oai_tools" not in info and hasattr(env, "oai_tools") and env.oai_tools:
            info["oai_tools"] = env.oai_tools
            state["info"]["oai_tools"] = env.oai_tools

        # Pass through all arguments
        completion, state = await env.rollout(
            client,
            model,
            prompt,
            completion,
            answer,
            state,
            task,
            info,
            example_id,
            sampling_args,
            **kwargs,
        )

        return completion, state

    def process_env_results_vllm(
        self,
        prompts: list[Messages],
        completions: list[Messages],
        states: list[State],
        rewards: list[float],
        processing_class: "PreTrainedTokenizerBase",
        max_seq_len: int = -1,
        mask_env_responses: bool = False,
        mask_truncated_completions: bool = False,
        zero_truncated_completions: bool = False,
        message_type: MessageType | None = "chat",
    ) -> ProcessedOutputs:
        """Route vLLM result processing to the appropriate sub-environment."""
        num_samples = len(prompts)
        assert (
            len(completions) == num_samples
            and len(states) == num_samples
            and len(rewards) == num_samples
        ), (
            f"Mismatch in lengths of prompts, completions, states, or rewards: {len(prompts)}, {len(completions)}, {len(states)}, {len(rewards)}"
        )
        all_prompt_ids = [[] for _ in range(num_samples)]
        all_prompt_masks = [[] for _ in range(num_samples)]
        all_completion_ids = [[] for _ in range(num_samples)]
        all_completion_masks = [[] for _ in range(num_samples)]
        all_completion_logprobs = [[] for _ in range(num_samples)]
        all_rewards = [0.0] * num_samples
        all_is_truncated = [False] * num_samples

        # keep track of indices for each task by grouping indices by task
        env_indices = defaultdict(list)
        for idx, state in enumerate(states):
            task = state.get("task")
            env_indices[task].append(idx)

        # process results for each task
        for task, indices in env_indices.items():
            env = self.get_env_for_task(task)
            env_processed_outputs = env.process_env_results_vllm(
                [prompts[i] for i in indices],
                [completions[i] for i in indices],
                [states[i] for i in indices],
                [rewards[i] for i in indices],
                processing_class,
                max_seq_len=max_seq_len,
                mask_env_responses=mask_env_responses,
                mask_truncated_completions=mask_truncated_completions,
                zero_truncated_completions=zero_truncated_completions,
                message_type=message_type,
            )
            # map processed outputs back to original indices
            for i, original_idx in enumerate(indices):
                all_prompt_ids[original_idx] = env_processed_outputs.prompt_ids[i]
                all_prompt_masks[original_idx] = env_processed_outputs.prompt_mask[i]
                all_completion_ids[original_idx] = env_processed_outputs.completion_ids[
                    i
                ]
                all_completion_masks[original_idx] = (
                    env_processed_outputs.completion_mask[i]
                )
                all_completion_logprobs[original_idx] = (
                    env_processed_outputs.completion_logprobs[i]
                )
                all_rewards[original_idx] = env_processed_outputs.rewards[i]
                all_is_truncated[original_idx] = env_processed_outputs.is_truncated[i]
        return ProcessedOutputs(
            prompt_ids=all_prompt_ids,
            prompt_mask=all_prompt_masks,
            completion_ids=all_completion_ids,
            completion_mask=all_completion_masks,
            completion_logprobs=all_completion_logprobs,
            rewards=all_rewards,
            is_truncated=all_is_truncated,
        )

    def get_env_for_task(self, task: str) -> Environment:
        """Get the environment instance for a given task name."""
        return self.env_map.get(task, self.envs[0])
