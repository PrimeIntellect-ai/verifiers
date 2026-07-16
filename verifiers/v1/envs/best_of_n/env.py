"""best-of-n: n independent attempts at the task, scored against each other.

The sibling-selection wrapper (`--env.id best-of-n` over any taskset): one "solver"
role attempts the task `--env.n` times; each attempt is judged by the task's own
rewards as usual, then `score()` compares the finished siblings — `best` marks the
argmax-reward attempt (rejection sampling / RFT selection reads it), `pass_at_n`
records whether any sibling reached `--env.threshold` (the pass@k signal, identical
on every sibling of the rollout).
"""

import asyncio

import verifiers.v1 as vf


class BestOfNParams(vf.EnvParams):
    solver: vf.AgentConfig = vf.AgentConfig()
    n: int = 4
    """Independent attempts per env-rollout, scored against each other."""
    threshold: float = 1.0
    """A sibling counts as solved when its task reward reaches this (`pass_at_n`)."""


class BestOfNEnv(vf.Environment[BestOfNParams]):
    def roles(self):
        return {"solver": self.params.solver}

    async def rollout(self, task, agents):
        return list(
            await asyncio.gather(
                *(agents["solver"].run(task) for _ in range(self.params.n))
            )
        )

    async def score(self, task, traces):
        """The sibling comparison. Ties share `best` (degenerate all-equal rollouts
        mark every sibling); `pass_at_n` is a rollout-level fact recorded on each
        trace so it survives flat consumers."""
        rewards = [trace.reward for trace in traces]
        best = max(rewards)
        solved = float(best >= self.params.threshold)
        for trace, reward in zip(traces, rewards):
            trace.record_metric("best", float(reward == best))
            trace.record_metric("pass_at_n", solved)
