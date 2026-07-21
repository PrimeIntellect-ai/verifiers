"""best-of-n: n independent attempts at the task, scored against each other."""

import asyncio

from pydantic import Field

import verifiers.v1 as vf


class BestOfNEnvConfig(vf.EnvConfig):
    # A single-agent-shaped env keeps the `agent` name: `--env.id best-of-n`
    # composes with a plain run's `--env.agent.*` flags unchanged.
    agent: vf.AgentConfig = vf.AgentConfig()
    n: int = Field(4, ge=1)
    """Independent attempts per episode, scored against each other."""
    threshold: float = 1.0
    """A sibling counts as solved when its task reward reaches this (`pass_at_n`)."""


class BestOfNEnv(vf.Env[BestOfNEnvConfig]):
    async def run(self, task, agents):
        # TaskGroup, not gather: a raising attempt cancels and awaits its
        # siblings, so no straggler keeps running past the episode.
        async with asyncio.TaskGroup() as group:
            for _ in range(self.config.n):
                group.create_task(agents.agent.run(task))

    async def finalize(self, task, traces):
        # `best` marks the argmax-reward attempt (ties share); `pass_at_n` is an
        # episode-level fact, recorded identically on every sibling so flat
        # consumers see it without reconstructing the group.
        best = max(t.reward for t in traces)
        for trace in traces:
            trace.record_metric("best", float(trace.reward == best))
            trace.record_metric("pass_at_n", float(best >= self.config.threshold))
