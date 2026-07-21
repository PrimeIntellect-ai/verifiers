"""duet: the smallest multi-agent env fixture — two agents echo the same phrase."""

import asyncio

import verifiers.v1 as vf

from echo_v1 import EchoTaskset, lenient_match


class DuetEnvConfig(vf.EnvConfig):
    a: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="null"))
    b: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="null"))


class DuetEnv(vf.Env[DuetEnvConfig]):
    def setup(self, agents):
        # "b" plays a fixed, untrainable participant.
        agents.b.trainable = False

    async def run(self, task, agents):
        async with asyncio.TaskGroup() as group:
            group.create_task(agents.a.run(task))
            group.create_task(agents.b.run(task))

    async def finalize(self, task, traces):
        # A sibling-dependent signal: did every agent echo the phrase?
        echoed = all(
            lenient_match(task.data.answer, t.last_reply) and not t.has_error
            for t in traces
        )
        for trace in traces:
            trace.record_metric("duet", float(echoed))


class DuetTaskset(EchoTaskset):
    pass


__all__ = ["DuetTaskset", "DuetEnv"]
