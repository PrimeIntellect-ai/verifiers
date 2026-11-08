"""format-guard: one episode of echo attempts, some ending as format violations.

The fixture for the framework's zero-reward guarantee (#3072), resolved by id
`format-guard`: attempts listed in `invalid` stop on a `format_violation` stop
hook on the model's reply, the stop condition a harness sets for an invalid
terminal response.
Every reward path then writes to every attempt: task scoring (`echoed`), a
scoring callback, `run()` writing straight into `rewards`, and `finalize()`.
Only the valid attempts may keep them.
"""

import asyncio

from echo import EchoTask, EchoTaskset
from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.runtimes import SubprocessConfig


class InvalidEchoTask(EchoTask):
    @vf.stop
    def format_violation(self, response: vf.Response) -> bool:
        return True


class FormatGuardEnvConfig(vf.EnvConfig):
    # The null chat loop needs no container: run it on the host.
    agent: vf.AgentConfig = vf.AgentConfig(
        harness=vf.HarnessConfig(id="null"), runtime=SubprocessConfig()
    )
    n: int = 3
    invalid: list[int] = Field(default_factory=list)
    """Indexes of the attempts that end as format violations."""


class FormatGuardEnv(vf.Env[FormatGuardEnvConfig]):
    async def run(self, task, agents):
        async def callback(trace, runtime):
            trace.record_reward("callback", 1.0)

        async def attempt(i: int) -> None:
            cls = InvalidEchoTask if i in self.config.invalid else EchoTask
            trace = await agents.agent.run(cls(task.data), on_score=callback)
            trace.rewards["direct"] = vf.Reward(score=1.0)

        await asyncio.gather(*(attempt(i) for i in range(self.config.n)))

    async def finalize(self, task, episode):
        for trace in episode.traces:
            trace.record_metric("seen_reward", trace.reward)
            trace.record_reward("group", 1.0)
            trace.record_metric("group", 1.0)


class FormatGuardTaskset(EchoTaskset):
    pass


__all__ = ["FormatGuardEnv", "FormatGuardTaskset"]
