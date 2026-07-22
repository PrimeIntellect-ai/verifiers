"""best-of-n: n independent attempts at the task, scored against each other.

One "agent" role attempts the task `--env.n` times (`--env.id best-of-n` over any
taskset); each attempt is judged by the task's own rewards as usual, then
`finalize()` compares the finished siblings: `best` marks the argmax-reward
attempt, `pass_at_n` whether any sibling reached `--env.threshold`.
"""

import asyncio

from pydantic import Field

import verifiers.v1 as vf


class BestOfNEnvConfig(vf.EnvConfig):
    # A single-role env keeps the single-agent seat name: `--env.id best-of-n`
    # composes with a plain run's `--env.agent.*` flags unchanged.
    agent: vf.AgentConfig = vf.AgentConfig()
    n: int = Field(4, ge=1)
    """Independent attempts per episode, scored against each other."""
    threshold: float = 1.0
    """A sibling counts as solved when its task reward reaches this (`pass_at_n`)."""


class BestOfNEnv(vf.Env[BestOfNEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        # TaskGroup: a raising attempt cancels and awaits its siblings, so no
        # straggler keeps burning tokens past the episode.
        async with asyncio.TaskGroup() as tg:
            for _ in range(self.config.n):
                tg.create_task(agents.agent.run(task))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        """The sibling comparison: `best` marks the argmax-reward attempt (ties
        share; a degenerate all-equal rollout marks every sibling); `pass_at_n` —
        whether any sibling reached the threshold — is a rollout-level fact,
        recorded identically on every sibling so flat consumers see it without
        reconstructing the group."""
        top = max(t.reward for t in episode.traces)
        solved = float(top >= self.config.threshold)
        for trace in episode.traces:
            trace.record_metric("best", float(trace.reward == top))
            trace.record_metric("pass_at_n", solved)
