"""The single-agent case — the env every plain taskset resolves to.

Not paired by id like its siblings here: `loaders.environment_class` falls back to
it whenever neither `--env.id` nor the taskset names an `Environment`, so a plain
eval is exactly this env (one `agent`, one trace per episode).
"""

from verifiers.v1.agents import AgentConfig, Agents
from verifiers.v1.env import EnvConfig, Environment
from verifiers.v1.task import Task


class SingleAgentEnvConfig(EnvConfig):
    """`SingleAgentEnv`'s config: the one `agent` over the seed taskset."""

    agent: AgentConfig = AgentConfig()
    """The one agent — the policy under evaluation/training; pin
    `--env.agent.harness.*` to choose its program or runtime."""


class SingleAgentEnv(Environment[SingleAgentEnvConfig]):
    """The single-agent case — the env every plain taskset resolves to: one
    `agent` playing the seed taskset (`--env.agent.*`), one trace per episode."""

    async def run(self, task: Task, agents: Agents) -> None:
        await agents.agent.run(task)
