"""duet: the smallest multi-agent environment — two roles echo the same phrase.

The multi-agent fixture for the v1 integration suite (resolved by id `duet`): an
`Env` subclass exported alongside its taskset, with two roles ("a", a trainable
seat on the run's model; "b", pinned untrainable), a `run()` that fans both out on
the task, and a `finalize()` that records a sibling-dependent signal. One episode
should land one record carrying two role-stamped traces.
"""

import asyncio

from echo import EchoTaskset, lenient_match

import verifiers.v1 as vf


class DuetEnvConfig(vf.EnvConfig):
    # Both seats pin the lean `null` chat loop.
    a: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="null"))
    b: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="null"))


class DuetEnv(vf.Env[DuetEnvConfig]):
    async def setup(self, agents):
        # "b" plays a fixed, untrainable participant.
        agents.b.trainable = False

    async def run(self, task, agents):
        await asyncio.gather(agents.a.run(task), agents.b.run(task))

    async def finalize(self, task, episode):
        # A sibling-dependent signal: did every seat echo the phrase?
        echoed = all(
            lenient_match(task.data.answer, t.last_reply) and not t.has_error
            for t in episode.traces
        )
        for trace in episode.traces:
            trace.record_metric("duet", float(echoed))


class DuetTaskset(EchoTaskset):
    pass


__all__ = ["DuetEnv", "DuetTaskset"]
