"""The single-agent case — the env every plain taskset resolves to.

Not paired by id like its siblings here: `loaders.environment_class` falls back to
it whenever neither `--env.id` nor the taskset names an `Environment`, so a plain
eval is exactly this env (one `agent` seat, one nameless trace per episode).
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from verifiers.v1.env import (
    AgentConfig,
    EnvConfig,
    Environment,
    validate_pairing,
)
from verifiers.v1.task import Task

if TYPE_CHECKING:
    from verifiers.v1.agent import Agent


class SingleAgentEnvConfig(EnvConfig):
    """`SingleAgentEnv`'s config: the one `agent` seat over the seed taskset."""

    agent: AgentConfig = AgentConfig()
    """The one seat — the policy under evaluation/training; pin
    `--env.agent.harness.*` to choose its program or runtime."""


class SingleAgentEnv(Environment[SingleAgentEnvConfig]):
    """The single-agent case — the env every plain taskset resolves to: one `agent`
    seat playing the seed taskset (`--env.agent.*`). Its one trace per episode
    carries no seat name, so the wire matches a plain eval's."""

    _stamp_roles = False

    def __init__(self, config: SingleAgentEnvConfig) -> None:
        super().__init__(config)
        # The one seat definitionally plays the seed taskset, so an impossible
        # pairing is knowable from class facts alone — refuse at construction,
        # before any work (multi-agent envs validate per run instead, on the
        # task each agent actually receives).
        harness = self._harnesses["agent"]
        validate_pairing(
            harness,
            self._task_cls,
            harness.config.runtime,
            shared_tools=type(self.taskset).tools,
        )

    async def rollout(self, task: Task, agents: Mapping[str, "Agent"]) -> None:
        await agents["agent"].run(task)
