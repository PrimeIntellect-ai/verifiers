"""
Protocol: Orchestrates multiple environments and actors for composable training.

The Protocol holds actors, environments, and the dataset. It enables cross-environment
calls via spawn() and handling initial scheduling via generate().

Dataset registration: The Protocol owns the dataset, not individual environments.
This allows multi-agent scenarios where the dataset is shared across environments.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List

from datasets import Dataset
from openai import AsyncOpenAI

from verifiers.types import DatasetBuilder, RolloutInput, SamplingArgs, State
from verifiers.utils.async_utils import NullAsyncContext, maybe_semaphore

from .actor import Actor

if TYPE_CHECKING:
    from .environment import Environment


class Protocol:
    """
    Holds actors, environments, and the dataset. Enables cross-env calls and handles generation scheduling.

    The Protocol owns the dataset, not individual environments. This allows multi-agent
    scenarios where the dataset is shared across environments.

    Usage:
        # Define actors
        solver = Actor(id="solver", system_prompt="You are a math solver...")
        verifier = Actor(id="verifier", system_prompt="You verify solutions...")

        # Define environments with their actors (no dataset needed)
        math_env = MathEnv(actors=["solver"])
        verify_env = VerifyEnv(actors=["verifier"])

        # Create protocol with dataset
        protocol = Protocol(
            actors=[solver, verifier],
            envs=[math_env, verify_env],
            dataset=my_dataset,  # Dataset registered here
        )

        # Get inputs from protocol's dataset
        inputs = protocol.get_inputs(n=100, rollouts_per_example=4)
        states = await protocol.generate(inputs, client, model)

        # Within math_env's env_response, spawn child rollouts:
        child_states = await self.protocol.spawn([sub_input1, sub_input2], score=True)
        state["child_states"].extend(child_states)
    """

    def __init__(
        self,
        actors: list[Actor],
        envs: list["Environment"],
        dataset: Dataset | DatasetBuilder | None = None,
        eval_dataset: Dataset | DatasetBuilder | None = None,
    ):
        """
        Initialize protocol with actors, environments, and optional dataset.

        - Builds actor dict: {actor.id: actor}
        - Builds env dict: {env.name: env}
        - Validates each env's actors exist in protocol
        - Injects self into each environment
        - Registers dataset (owned by Protocol, not environments)

        Args:
            actors: List of Actor instances to register
            envs: List of Environment instances to register
            dataset: Training dataset (Dataset or callable that returns Dataset)
            eval_dataset: Evaluation dataset (Dataset or callable that returns Dataset)
        """
        # Register actors
        self._actors: dict[str, Actor] = {}
        for actor in actors:
            if actor.id in self._actors:
                raise ValueError(f"Duplicate actor id: {actor.id}")
            self._actors[actor.id] = actor

        # Register environments
        self._envs: dict[str, "Environment"] = {}
        for env in envs:
            name = getattr(env, "name", env.__class__.__name__)
            if name in self._envs:
                raise ValueError(f"Duplicate environment name: {name}")

            # Validate env's actors exist in protocol
            env_actors = getattr(env, "actors", [])
            for actor_id in env_actors:
                if actor_id not in self._actors:
                    raise ValueError(
                        f"Environment '{name}' references unknown actor '{actor_id}'. "
                        f"Available actors: {list(self._actors.keys())}"
                    )

            self._envs[name] = env
            # Inject protocol reference
            env.protocol = self

        # Dataset registration (owned by Protocol)
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None

        if dataset is not None:
            if callable(dataset):
                self._dataset_source: DatasetBuilder | None = dataset
            else:
                self._dataset_source = lambda ds=dataset: ds
                self._build_dataset()  # Eagerly build for raw datasets
        else:
            self._dataset_source = None

        if eval_dataset is not None:
            if callable(eval_dataset):
                self._eval_dataset_source: DatasetBuilder | None = eval_dataset
            else:
                self._eval_dataset_source = lambda ds=eval_dataset: ds
                self._build_eval_dataset()  # Eagerly build for raw datasets
        else:
            self._eval_dataset_source = None

        # Context stored during generate() for spawn() to use
        self._client: AsyncOpenAI | None = None
        self._model: str | None = None
        self._sampling_args: SamplingArgs | None = None
        self._gen_sem = None
        self._score_sem = None

    def get_actor(self, actor_id: str) -> Actor:
        """Get actor by id."""
        if actor_id not in self._actors:
            raise KeyError(
                f"Actor '{actor_id}' not found. Available: {list(self._actors.keys())}"
            )
        return self._actors[actor_id]

    def get_env(self, name: str) -> "Environment":
        """Get environment by name."""
        if name not in self._envs:
            raise KeyError(
                f"Environment '{name}' not found. Available: {list(self._envs.keys())}"
            )
        return self._envs[name]

    @property
    def actors(self) -> dict[str, Actor]:
        """All registered actors."""
        return self._actors

    @property
    def envs(self) -> dict[str, "Environment"]:
        """All registered environments."""
        return self._envs

    # Dataset management methods

    def _build_dataset(self) -> Dataset | None:
        """Build and cache the training dataset from source if needed."""
        if self._dataset is not None:
            return self._dataset
        if self._dataset_source is None:
            return None
        self._dataset = self._dataset_source()
        return self._dataset

    def _build_eval_dataset(self) -> Dataset | None:
        """Build and cache the evaluation dataset from source if needed."""
        if self._eval_dataset is not None:
            return self._eval_dataset
        if self._eval_dataset_source is None:
            return None
        self._eval_dataset = self._eval_dataset_source()
        return self._eval_dataset

    def get_dataset(self, n: int = -1, seed: int | None = None) -> Dataset:
        """Get the training dataset, optionally shuffled and truncated."""
        self._build_dataset()
        if self._dataset is None:
            raise ValueError("dataset is not set on Protocol")
        dataset = self._dataset
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if n > 0:
            n = min(n, len(dataset))
            return dataset.select(range(n))
        return dataset

    def get_eval_dataset(self, n: int = -1, seed: int | None = None) -> Dataset:
        """Get the evaluation dataset, optionally shuffled and truncated."""
        self._build_eval_dataset()
        if self._eval_dataset is None:
            # Fall back to train dataset
            return self.get_dataset(n, seed)
        dataset = self._eval_dataset
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if n > 0:
            n = min(n, len(dataset))
            return dataset.select(range(n))
        return dataset

    def get_inputs(
        self, n: int = -1, rollouts_per_example: int = 1, seed: int | None = None
    ) -> List[RolloutInput]:
        """Get training inputs from the dataset."""
        dataset = self.get_dataset(n=n, seed=seed)
        if rollouts_per_example > 1:
            dataset = dataset.repeat(rollouts_per_example)
        return dataset.to_list()

    def get_eval_inputs(
        self, n: int = -1, rollouts_per_example: int = 1, seed: int | None = None
    ) -> List[RolloutInput]:
        """Get evaluation inputs from the dataset."""
        dataset = self.get_eval_dataset(n=n, seed=seed)
        if rollouts_per_example > 1:
            dataset = dataset.repeat(rollouts_per_example)
        return dataset.to_list()

    async def generate(
        self,
        inputs: list[RolloutInput],
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        max_concurrent: int = -1,
    ) -> list[State]:
        """
        Generate rollouts for a batch of inputs.

        This is where initial scheduling happens - determines which env
        handles each input and dispatches accordingly.

        Called by MultiAgentOrchestrator.generate_batch().
        """
        # Store context for spawn() calls during this generation
        self._client = client
        self._model = model
        self._sampling_args = sampling_args
        self._gen_sem = await maybe_semaphore(max_concurrent)
        self._score_sem = await maybe_semaphore(max_concurrent)

        # Group inputs by target environment
        by_env: dict[str, list[RolloutInput]] = {}
        for inp in inputs:
            env_name = inp.get("task") or self._get_default_env()
            by_env.setdefault(env_name, []).append(inp)

        # Run each environment's generate()
        all_states: list[State] = []
        for env_name, env_inputs in by_env.items():
            env = self.get_env(env_name)
            results = await env.generate(
                env_inputs,
                client=client,
                model=model,
                sampling_args=sampling_args,
                max_concurrent=max_concurrent,
            )
            all_states.extend(results["state"])

        # Flatten: collect trainable child_states recursively
        return self._flatten_trainable(all_states)

    def _get_default_env(self) -> str:
        """Return first registered environment as default."""
        return next(iter(self._envs.keys()))

    def _flatten_trainable(self, states: list[State]) -> list[State]:
        """Recursively collect all trainable states including children."""
        result: list[State] = []
        for state in states:
            result.append(state)
            child_states = state.get("child_states", [])
            if child_states:
                result.extend(self._flatten_trainable(child_states))
        return result

    async def spawn(
        self,
        inputs: list[RolloutInput],
        score: bool = True,
    ) -> list[State]:
        """
        Spawn child rollouts in sibling environments.

        Routes each input to its target environment based on the `task` field,
        then runs rollouts in parallel using asyncio.gather.
        Uses context stored by the enclosing generate() call.

        Args:
            inputs: List of rollout inputs (task field determines target env)
            score: Whether to score the rollouts after completion

        Returns:
            List of completed states from the child rollouts
        """
        if self._client is None or self._model is None:
            raise RuntimeError(
                "spawn() can only be called within a generate() context. "
                "Ensure you're calling spawn() from within an env_response or rollout."
            )

        # Use NullAsyncContext for children to allow parallel execution
        null_sem = NullAsyncContext()

        # Run all rollouts in parallel
        all_states = await asyncio.gather(*(
            self.get_env(inp.get("task") or self._get_default_env()).run_rollout(
                inp,
                client=self._client,
                model=self._model,
                gen_sampling_args=self._sampling_args or {},
                gen_sem=null_sem,
                score_sem=null_sem,
                score=score,
            )
            for inp in inputs
        ))

        return list(all_states)

    async def evaluate(
        self,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        max_concurrent: int = -1,
        seed: int | None = None,
    ) -> list[State]:
        """
        Evaluate model on the Protocol's evaluation dataset.

        Gets inputs from the Protocol's eval dataset (or train dataset if no eval
        dataset is set) and runs generate() to produce rollout states.
        """
        inputs = self.get_eval_inputs(
            n=num_examples,
            rollouts_per_example=rollouts_per_example,
            seed=seed,
        )
        return await self.generate(
            inputs=inputs,
            client=client,
            model=model,
            sampling_args=sampling_args,
            max_concurrent=max_concurrent,
        )
