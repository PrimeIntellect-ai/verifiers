"""Single-agent preset for Harbor task attempts."""

import copy

import verifiers.v1 as vf
from verifiers.v1.envs.isolated_verifier import IsolatedVerifierEnvConfig
from verifiers.v1.tasksets.harbor.taskset import HarborTask


class HarborEnvConfig(IsolatedVerifierEnvConfig):
    trust_compose: bool = False
    """Allow local Compose tasks to use host files and Docker privileges."""


class HarborEnv(vf.Env[HarborEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, HarborTask):
            raise TypeError(
                f"the harbor env runs harbor tasks; got {type(task).__name__}"
            )
        task = copy.deepcopy(task)
        task.config = task.config.model_copy(
            update={
                "trust_compose": task.config.trust_compose or self.config.trust_compose,
                "verifier": task.config.verifier
                or (self.config.verifier if task.data.verifier is not None else None),
            }
        )
        await agents.agent.run(task)
