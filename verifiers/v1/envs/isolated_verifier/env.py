"""Single-agent preset for task-owned isolated verification."""

import verifiers.v1 as vf
from verifiers.v1.agent import resolve_rollout_timeouts
from verifiers.v1.configs.verifier import VerifierConfig


class IsolatedVerifierEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()
    verifier: VerifierConfig = VerifierConfig()


class IsolatedVerifierEnv(vf.Env[IsolatedVerifierEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        async with task.open(
            placement=agents.agent.runtime_config,
            timeouts=resolve_rollout_timeouts(agents.agent.timeout, task),
            verifier=self.config.verifier,
        ) as attempt:
            solution = await agents.agent.run(attempt)
            await attempt.grade(solution)
