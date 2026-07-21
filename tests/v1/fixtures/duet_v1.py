"""duet: the smallest multi-agent environment — two agents echo the same phrase.

The multi-agent fixture for the v1 e2e suite (resolved by id `duet-v1`): an
`Environment` subclass exported alongside its taskset, with two agents ("a", a
trainable agent on the run's model; "b", pinned untrainable), a `run()` that fans
both out on the task, and a `score()` that records a sibling-dependent signal. One
eval rollout should land two agent-stamped traces sharing one episode id.
"""

import asyncio

import verifiers.v1 as vf

from echo_v1 import EchoTaskset, lenient_match


class DuetEnvConfig(vf.EnvConfig):
    # Both agents pin the lean `null` chat loop.
    a: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="null"))
    b: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="null"))


class DuetEnv(vf.Environment[DuetEnvConfig]):
    def setup(self, agents):
        # "b" plays a fixed, untrainable participant.
        agents.b.trainable = False

    async def run(self, task, agents):
        await asyncio.gather(agents.a.run(task), agents.b.run(task))

    async def score(self, task, traces):
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
