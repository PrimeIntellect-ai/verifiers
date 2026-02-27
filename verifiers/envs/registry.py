"""
Registry: Wires agents to environments and enables cross-environment spawning.

Registry is the glue that connects:
- Agents (entities with model config and system prompts)
- Environments (where rollouts happen)

Key functionality:
- Agent registry: Look up agents by ID
- Env registry: Look up environments by name
- spawn(): Run child rollouts in other environments (e.g., Orchestrator spawns Solvers)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from verifiers.types import RolloutInput, SamplingArgs, State

if TYPE_CHECKING:
    from .environment import Environment


class Registry:
    """
    Wires agents to environments. Enables spawn() for cross-env communication.
    """

    def __init__(
        self,
        agents: list[Any],
        envs: list["Environment"],
    ):
        """
        Register agents and environments.

        Injects registry reference into each env (env.registry = self)
        and shares agents with envs that support inject_agents().

        Args:
            agents: List of Agent instances to register
            envs: List of Environment instances to register
        """
        # Register agents by ID
        self._agents: dict[str, Any] = {}
        for agent in agents:
            if agent.id in self._agents:
                raise ValueError(f"Duplicate agent id: {agent.id}")
            self._agents[agent.id] = agent

        # Register environments by name
        self._envs: dict[str, "Environment"] = {}
        for env in envs:
            name = getattr(env, "name", env.__class__.__name__)
            if name in self._envs:
                raise ValueError(f"Duplicate environment name: {name}")
            self._envs[name] = env
            # Inject registry reference so env can call self.registry.spawn()
            env.registry = self
            # Share agents with env so get_actor() works
            if hasattr(env, "inject_agents"):
                env.inject_agents(self._agents)

    def get_agent(self, agent_id: str) -> Any:
        """Get agent by ID."""
        if agent_id not in self._agents:
            raise KeyError(
                f"Agent '{agent_id}' not found. Available: {list(self._agents.keys())}"
            )
        return self._agents[agent_id]

    def get_env(self, name: str) -> "Environment":
        """Get environment by name."""
        if name not in self._envs:
            raise KeyError(
                f"Environment '{name}' not found. Available: {list(self._envs.keys())}"
            )
        return self._envs[name]

    @property
    def agents(self) -> dict[str, Any]:
        """All registered agents."""
        return self._agents

    @property
    def envs(self) -> dict[str, "Environment"]:
        """All registered environments."""
        return self._envs

    async def spawn(
        self,
        inputs: list[RolloutInput],
        client: Any,
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
            client: Client instance
            model: Model name
            sampling_args: Optional sampling parameters
            score: Whether to score rollouts with env's rubric (default True)

        Returns:
            List of completed states from child rollouts
        """
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

        # Mark spawned states as children
        for state in all_states:
            if "extras" not in state:
                state["extras"] = {}
            state["extras"]["parent_episode_id"] = "spawned"

        # Score rollouts if requested
        if score:
            states_by_env: dict[str, list[State]] = {}
            for inp, state in zip(inputs, all_states):
                env_name = inp.get("task")
                if env_name not in states_by_env:
                    states_by_env[env_name] = []
                states_by_env[env_name].append(state)

            for env_name, env_states in states_by_env.items():
                env = self.get_env(env_name)
                if env.rubric:
                    await env.rubric.score_group(env_states)

        return list(all_states)
