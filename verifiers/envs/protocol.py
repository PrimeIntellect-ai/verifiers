"""
Protocol: Orchestrates multiple environments and actors for composable training.

- Protocol: Top-level coordinator owning actors, environments, and dataset
- Enables cross-environment spawning via spawn()
- Manages dataset for multi-environment training
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import TYPE_CHECKING, List

from datasets import Dataset
from openai import AsyncOpenAI

from verifiers.types import DatasetBuilder, RolloutInput, SamplingArgs, State
from verifiers.utils.async_utils import maybe_semaphore

from .actor import Actor

if TYPE_CHECKING:
    from .environment import Environment

# Context variables for task-local storage during generate().
# Enables: (1) concurrent generate() calls without interference,
# (2) simplified spawn() API - environments can call self.protocol.spawn(inputs)
#     without passing client/model explicitly (retrieved from context).
_ctx_client: contextvars.ContextVar[AsyncOpenAI | None] = contextvars.ContextVar("client", default=None)
_ctx_model: contextvars.ContextVar[str | None] = contextvars.ContextVar("model", default=None)
_ctx_sampling_args: contextvars.ContextVar[SamplingArgs | None] = contextvars.ContextVar("sampling_args", default=None)


class Protocol:
    """Top-level coordinator owning actors, environments, and dataset."""

    def __init__(
        self,
        actors: list[Actor],
        envs: list["Environment"],
        dataset: Dataset | DatasetBuilder | None = None,
        eval_dataset: Dataset | DatasetBuilder | None = None,
    ):
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
            self._envs[name] = env
            # Inject protocol reference into environment
            env.protocol = self

        # Dataset registration (owned by Protocol)
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None

        if dataset is not None:
            if callable(dataset):
                self._dataset_source: DatasetBuilder | None = dataset
            else:
                self._dataset_source = lambda ds=dataset: ds
                self._build_dataset()
        else:
            self._dataset_source = None

        if eval_dataset is not None:
            if callable(eval_dataset):
                self._eval_dataset_source: DatasetBuilder | None = eval_dataset
            else:
                self._eval_dataset_source = lambda ds=eval_dataset: ds
                self._build_eval_dataset()
        else:
            self._eval_dataset_source = None

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

    # Dataset management

    def _build_dataset(self) -> Dataset | None:
        """Build and cache the training dataset from source."""
        if self._dataset is not None:
            return self._dataset
        if self._dataset_source is None:
            return None
        self._dataset = self._dataset_source()
        return self._dataset

    def _build_eval_dataset(self) -> Dataset | None:
        """Build and cache the evaluation dataset from source."""
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
            raise ValueError("Dataset is not set on Protocol")
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
            return self.get_dataset(n, seed)
        dataset = self._eval_dataset
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if n > 0:
            n = min(n, len(dataset))
            return dataset.select(range(n))
        return dataset

    def get_eval_inputs(
        self, n: int = -1, rollouts_per_example: int = 1, seed: int | None = None
    ) -> List[RolloutInput]:
        """Get evaluation inputs from the dataset."""
        dataset = self.get_eval_dataset(n=n, seed=seed)
        inputs = dataset.to_list()
        if rollouts_per_example > 1:
            inputs = inputs * rollouts_per_example
        return inputs

    async def generate(
        self,
        inputs: list[RolloutInput],
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        max_concurrent: int = -1,
    ) -> list[State]:
        """Generate rollouts, dispatching to environments based on input['task']."""
        # Store context in task-local variables for spawn() calls
        # Using contextvars allows concurrent generate() calls on same Protocol
        _ctx_client.set(client)
        _ctx_model.set(model)
        _ctx_sampling_args.set(sampling_args)

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
        return self._flatten_states(all_states)

    def _get_default_env(self) -> str:
        """Return first registered environment as default."""
        return next(iter(self._envs.keys()))

    def _flatten_states(self, states: list[State]) -> list[State]:
        """Recursively collect all states including children."""
        result: list[State] = []
        for state in states:
            result.append(state)
            child_states = state.get("child_states", [])
            if child_states:
                result.extend(self._flatten_states(child_states))
        return result

    async def spawn(
        self,
        inputs: list[RolloutInput],
        score: bool = True,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> list[State]:
        """Spawn child rollouts from within an environment.

        Can pass client/model explicitly, or they'll be read from context vars
        (set by Protocol.generate()).
        """
        # Try explicit params first, then context vars
        client = client or _ctx_client.get()
        model = model or _ctx_model.get()
        sampling_args = sampling_args or _ctx_sampling_args.get()

        if client is None or model is None:
            raise RuntimeError(
                "spawn() requires client and model. Either pass them explicitly "
                "or call spawn() from within a Protocol.generate() context."
            )

        # Run all rollouts in parallel
        tasks = []
        for inp in inputs:
            env_name = inp.get("task") or self._get_default_env()
            env = self.get_env(env_name)
            tasks.append(
                env.rollout(
                    inp,
                    client=client,
                    model=model,
                    sampling_args=sampling_args,
                )
            )

        all_states = await asyncio.gather(*tasks)

        # Score rollouts if requested
        if score:
            from verifiers.utils.async_utils import maybe_semaphore
            score_sem = await maybe_semaphore(-1)  # No concurrency limit
            for inp, state in zip(inputs, all_states):
                env_name = inp.get("task") or self._get_default_env()
                env = self.get_env(env_name)
                if env.rubric:
                    await env.rubric.score_rollout(state, score_sem=score_sem)

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
        """Evaluate model on the Protocol's evaluation dataset."""
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
