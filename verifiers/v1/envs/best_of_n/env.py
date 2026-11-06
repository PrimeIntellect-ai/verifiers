"""best-of-n: n independent attempts at the task, scored against each other.

One "agent" role attempts the task `--env.n` times (`--env.id best-of-n` over any
taskset); each attempt is judged by the task's own rewards as usual, then
`finalize()` compares the finished siblings: `best` marks the argmax-reward
attempt, `pass_at_n` whether any sibling reached `--env.threshold` (by default the
task's pass threshold).
"""

import asyncio

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.configs.scoring import resolve_scoring


class BestOfNEnvConfig(vf.EnvConfig):
    # A single-role env keeps the single-agent seat name: `--env.id best-of-n`
    # composes with a plain run's `--env.agent.*` flags unchanged.
    agent: vf.AgentConfig = vf.AgentConfig()
    n: int = Field(4, ge=1)
    """Independent attempts per episode, scored against each other."""
    threshold: float | None = None
    """A sibling counts as solved when its task reward reaches this (`pass_at_n`).
    None uses the task's pass threshold (`ScoringConfig.pass_threshold`)."""


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
        threshold = self.solved_threshold(task)
        solved = float(threshold is not None and top >= threshold)
        for trace in episode.traces:
            trace.record_metric("best", float(trace.reward == top))
            trace.record_metric("pass_at_n", solved)

    def solved_threshold(self, task: vf.Task) -> float | None:
        """`--env.threshold`, else the task's pass threshold under the agent's
        scoring policy; None when neither defines one."""
        if self.config.threshold is not None:
            return self.config.threshold
        return resolve_scoring(task.scoring, self.config.agent.scoring).threshold
