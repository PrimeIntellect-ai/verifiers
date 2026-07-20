"""duet: the smallest multi-agent environment — two roles echo the same phrase.

The multi-agent fixture for the v1 e2e suite (resolved by id `duet-v1`): an
`Environment` subclass exported alongside its taskset, with two roles ("a", a trainable
seat on the run's model; "b", pinned untrainable), a `rollout()` that fans both out on
the task, and a `score()` that records a sibling-dependent signal. One eval rollout
should land one record carrying two role-stamped traces.
"""

import asyncio

import verifiers.v1 as vf

from echo_v1 import EchoTaskset, lenient_match


class DuetEnvConfig(vf.EnvConfig):
    # Both seats pin the lean `null` chat loop.
    a: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="null"))
    b: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="null"))


class DuetEnv(vf.Environment[DuetEnvConfig]):
    def brief(self, agents):
        # "b" plays a fixed, untrainable participant.
        agents["b"].trainable = False

    async def rollout(self, task, agents):
        a, b = await asyncio.gather(agents["a"].run(task), agents["b"].run(task))
        return {"a": a, "b": b}

    async def score(self, task, views):
        # A sibling-dependent signal: did every seat echo the phrase?
        echoed = all(
            lenient_match(task.data.answer, t.last_reply) and not t.has_error
            for t in views.values()
        )
        for trace in views.values():
            trace.record_metric("duet", float(echoed))


class DuetTaskset(EchoTaskset):
    pass


__all__ = ["DuetTaskset", "DuetEnv"]
