from typing import Mapping

from datasets import concatenate_datasets
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
from verifiers.types import RolloutScore


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
        info = info or {}

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


class EnvGroupSparseRubric(EnvGroupRubric):
    """
    enhanced EnvGroup rubric with domain-specific sparse tracking.

    this rubric extends EnvGroupRubric to support sparse metrics for multi-domain environments.
    when routing scoring to domain-specific environments, it automatically marks metrics
    that weren't computed by the target environment as sparse (excluded from averaging).

    Key differences from standard EnvGroupRubric:
    - marks uncomputed domain metrics as sparse (e.g., chemistry_reward=0.0 becomes sparse)
    - enables mathematically correct domain averaging by excluding irrelevant zeros
    - Only used when EnvGroup is initialized with enable_sparse_metrics=True

    Example: For a chemistry task in ProfBench, physics/finance/consulting rewards are marked
    sparse, ensuring chemistry_reward averages only over actual chemistry evaluations.
    """

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
        Route scoring with sparse metrics support for multi-domain environments.

        This method handles scoring by:
        1. Routing the task to the appropriate domain-specific environment
        2. Computing metrics using that environment's rubric
        3. Filling uncomputed metrics with 0.0 and marking them as sparse
        4. Returning results with sparse flags for proper averaging

        Only used when EnvGroup has enable_sparse_metrics=True.
        """
        state = state or {}
        info = info or {}

        # pre-initialize all known metrics to 0.0
        # this ensures consistent metric structure across all rollouts
        # uncomputed metrics will remain 0.0 and be marked sparse
        metrics = {name: 0.0 for name in self.all_reward_names}
        reward = 0.0

        # Route to appropriate domain environment based on task
        env = self.env_map.get(task)
        if env is None:
            self.logger.warning(f"No environment found for task '{task}'")
            return RolloutScore(reward=reward, metrics=metrics)

        # Score using the domain-specific environment's rubric
        # this computes only the metrics relevant to this domain
        env_results = await env.rubric.score_rollout(
            prompt, completion, answer, state, task, info, example_id, **kwargs
        )

        # update metrics with computed values from domain environment
        # metrics not computed by this environment remain at 0.0
        for reward_name, score in env_results.metrics.items():
            if reward_name in metrics:
                metrics[reward_name] = score

        # mark uncomputed metrics as sparse for exclusion from averaging
        # example: for chemistry task, physics/finance/consulting rewards marked sparse
        # this enables mathematically correct domain averaging
        uncomputed_metrics = set(self.all_reward_names) - set(
            env_results.metrics.keys()
        )
        sparse_metrics = uncomputed_metrics if uncomputed_metrics else None

        # Overall reward comes from the domain environment
        reward = env_results.reward

        return RolloutScore(
            reward=reward, metrics=metrics, sparse_metrics=sparse_metrics
        )


class EnvGroup(Environment):
    """
    Environment group that acts as a mixture of multiple environments.

    Routes operations to appropriate sub-environments based on the 'task' column.
    """

    def __init__(
        self,
        envs: list[Environment],
        env_names: list[str] | None = None,
        enable_sparse_metrics: bool = False,
        **kwargs,
    ):
        """
        Initialize EnvGroup with a list of environments.

        Args:
            envs: list of Environment instances
            env_names: Optional list of names for each environment.
                      If not provided, uses "env_0", "env_1", etc.
            enable_sparse_metrics: Enable sparse metrics for mathematically correct domain averaging
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
                env_dataset = env_dataset.map(add_task)
                datasets.append(env_dataset)
            env_eval_dataset = env.get_eval_dataset()
            if env_eval_dataset is not None:
                env_eval_dataset = env_eval_dataset.map(add_task)
                eval_datasets.append(env_eval_dataset)
        dataset = concatenate_datasets(datasets) if datasets else None
        eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None
        # choose rubric type based on enable_sparse_metrics flag
        # this is the key decision point for sparse metrics activation
        if enable_sparse_metrics:
            # use sparse-aware rubric that marks uncomputed domain metrics as sparse
            # enables mathematically correct averaging by excluding irrelevant zeros
            rubric = EnvGroupSparseRubric(self.env_map)
        else:
            # use standard rubric that includes all values in averaging (backwards compatible)
            # this preserves existing behavior for environments without sparse metrics
            rubric = EnvGroupRubric(self.env_map)

        # don't set oai_tools at the group level since different sub-environments
        # may have different tools. Instead, set them per-task in rollout().
        # initialize parent Environment
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            oai_tools=None,
            **kwargs,
        )
        self.logger.info(
            f"Initialized EnvGroup with {len(envs)} environments: {self.env_names}"
        )

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
        info = info or {}
        sampling_args = sampling_args or {}

        # Route to appropriate environment
        env = self.env_map[task]

        # Set tools for this task's environment if not already set in info
        if "oai_tools" not in info and hasattr(env, "oai_tools") and env.oai_tools:
            info["oai_tools"] = env.oai_tools

        # Pass through all arguments
        return await env.rollout(
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

    def get_env_for_task(self, task: str) -> Environment:
        """Get the environment instance for a given task name."""
        return self.env_map.get(task, self.envs[0])
