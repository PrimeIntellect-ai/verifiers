"""best-of-n: n independent attempts at the task, scored against each other.

One "agent" attempts the task `--env.n` times (`--env.id best-of-n` over any
taskset); each attempt is judged by the task's own rewards as usual, then two
cross-trace metrics compare the finished siblings: `best` marks the argmax-reward
attempt, `pass_at_n` whether any sibling reached `--env.threshold`.
"""

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


class BestOfNEnv(vf.Environment[BestOfNEnvConfig]):
    async def run(self, task, agents):
        await asyncio.gather(*(agents.agent.run(task) for _ in range(self.config.n)))

    @vf.metric
    async def best(self, trace, traces):
        """Marks the argmax-reward attempt; ties share (a degenerate all-equal
        episode marks every sibling)."""
        return float(trace.reward == max(t.reward for t in traces))

    @vf.metric
    async def pass_at_n(self, trace, traces):
        """Whether any sibling reached the threshold — an episode-level fact,
        recorded identically on every sibling so flat consumers see it without
        reconstructing the group."""
        return float(max(t.reward for t in traces) >= self.config.threshold)
