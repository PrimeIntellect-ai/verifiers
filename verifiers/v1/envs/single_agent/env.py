"""The single-agent case — the env every plain taskset resolves to."""

from verifiers.v1.agents import AgentConfig, Agents
from verifiers.v1.env import Env, EnvConfig
from verifiers.v1.task import Task


class SingleAgentEnvConfig(EnvConfig):
    agent: AgentConfig = AgentConfig()


class SingleAgentEnv(Env[SingleAgentEnvConfig]):
    async def run(self, task: Task, agents: Agents):
        await agents.agent.run(task)
