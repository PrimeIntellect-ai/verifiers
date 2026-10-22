"""The single-agent case — the env every plain taskset resolves to.

Not paired by id like its siblings here: `loaders.environment_class` falls back to
it whenever neither `--env.id` nor the taskset names an `Env`, so a plain
eval is exactly this env (one `agent` seat, one nameless trace per episode).
"""

from verifiers.v1.agent import Agents
from verifiers.v1.configs.env import SingleAgentEnvConfig
from verifiers.v1.env import Env
from verifiers.v1.task import Task


class SingleAgentEnv(Env[SingleAgentEnvConfig]):
    """The single-agent case — the env every plain taskset resolves to: one `agent`
    seat playing the seed taskset (`--env.agent.*`). Its one trace per episode
    carries no seat name, so the wire matches a plain eval's."""

    async def run(self, task: Task, agents: Agents) -> None:
        await agents.agent.run(task)
