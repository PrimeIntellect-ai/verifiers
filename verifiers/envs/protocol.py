"""
Protocol: Wires actors to environments and enables cross-environment spawning.

Protocol is the glue that connects:
- Actors (trainable entities with system prompts)
- Environments (where rollouts happen)

Key functionality:
- Actor registry: Look up actors by ID
- Env registry: Look up environments by name
- spawn(): Run child rollouts in other environments (e.g., Proposer spawns Solvers)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from verifiers.types import RolloutInput, SamplingArgs, State
from verifiers.utils.async_utils import maybe_semaphore

from .actor import Actor

if TYPE_CHECKING:
    from .environment import Environment


class Protocol:
    """
    Wires actors to environments. Enables spawn() for cross-env communication.

    """

    def __init__(
        self,
        actors: list[Actor],
        envs: list["Environment"],
    ):
        """
        Register actors and environments.

        Args:
            actors: List of Actor instances to register
            envs: List of Environment instances to register
        """
        # Register actors by ID
        self._actors: dict[str, Actor] = {}
        for actor in actors:
            if actor.id in self._actors:
                raise ValueError(f"Duplicate actor id: {actor.id}")
            self._actors[actor.id] = actor

        # Register environments by name
        self._envs: dict[str, "Environment"] = {}
        for env in envs:
            name = getattr(env, "name", env.__class__.__name__)
            if name in self._envs:
                raise ValueError(f"Duplicate environment name: {name}")
            self._envs[name] = env
            # Inject protocol reference so env can call self.protocol.spawn()
            env.protocol = self

    def get_actor(self, actor_id: str) -> Actor:
        """Get actor by ID."""
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

    async def spawn(
        self,
        inputs: list[RolloutInput],
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        score: bool = True,
    ) -> list[State]:
        """
        Spawn child rollouts in target environments.

        Routes each input to its environment based on input["task"],
        runs rollouts in parallel, and optionally scores them.

        Args:
            inputs: List of rollout inputs, each with "task" field for routing
            client: AsyncOpenAI client (required)
            model: Model name (required)
            sampling_args: Optional sampling parameters
            score: Whether to score rollouts with env's rubric (default True)

        Returns:
            List of completed states from child rollouts

            )
        """
        # Run all rollouts in parallel
        tasks = []
        for inp in inputs:
            env_name = inp.get("task")
            if not env_name:
                raise ValueError("spawn() requires 'task' field in each input")
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

        # Mark spawned states as children (for progress tracking)
        for state in all_states:
            if "extras" not in state:
                state["extras"] = {}
            state["extras"]["parent_episode_id"] = "spawned"

        # Score rollouts if requested
        # Use score_group() instead of score_rollout() to properly handle:
        # - MultiAgentRubric per-actor reward functions
        # - GRPO advantage computation per-actor group
        if score:
            score_sem = await maybe_semaphore(-1)
            # Group states by environment for proper score_group() semantics
            states_by_env: dict[str, list[State]] = {}
            for inp, state in zip(inputs, all_states):
                env_name = inp.get("task")
                if env_name not in states_by_env:
                    states_by_env[env_name] = []
                states_by_env[env_name].append(state)

            # Score each group with its environment's rubric
            for env_name, env_states in states_by_env.items():
                env = self.get_env(env_name)
                if env.rubric:
                    await env.rubric.score_group(env_states, score_sem=score_sem)

        return list(all_states)
