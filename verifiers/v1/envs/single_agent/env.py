"""The single-agent case — the env every plain taskset resolves to."""

from verifiers.v1.agents import AgentConfig, Agents
from verifiers.v1.env import Env, EnvConfig
from verifiers.v1.task import Task


class SingleAgentEnvConfig(EnvConfig):
    agent: AgentConfig = AgentConfig()
    """The one agent — the policy under evaluation/training; pin
    `--env.agent.harness.*` to choose its program or runtime."""


class SingleAgentEnv(Env[SingleAgentEnvConfig]):
    """One `agent` playing the seed taskset, one trace per episode. Not paired by
    id like its siblings here: `loaders.environment_class` falls back to it whenever
    neither `--env.id` nor the taskset names an `Env`."""

    async def run(self, task: Task, agents: Agents):
        await agents.agent.run(task)
