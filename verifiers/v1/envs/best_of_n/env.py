"""best-of-n: n independent attempts at the task, scored against each other.

The sibling-selection wrapper (`--env.id best-of-n` over any taskset): one "solver"
role attempts the task `--env.n` times; each attempt is judged by the task's own
rewards as usual, then two decorated cross-agent metrics compare the finished
siblings — `best` marks the argmax-reward attempt (rejection sampling / RFT
selection reads it), `pass_at_n` records whether any sibling reached
`--env.threshold` (the pass@k signal, identical on every sibling of the rollout).
"""

import asyncio

from pydantic import Field

import verifiers.v1 as vf


class BestOfNParams(vf.EnvParams):
    solver: vf.AgentConfig = vf.AgentConfig()
    n: int = Field(4, ge=1)
    """Independent attempts per env-rollout, scored against each other."""
    threshold: float = 1.0
    """A sibling counts as solved when its task reward reaches this (`pass_at_n`)."""


class BestOfNEnv(vf.Environment[BestOfNParams]):
    # No roles() override: the default 1:1 plays the declared `solver` field
    # as a dataset role.
    async def rollout(self, task, agents):
        return list(
            await asyncio.gather(
                *(agents["solver"].run(task) for _ in range(self.params.n))
            )
        )

    @vf.metric
    async def best(self, trace, traces):
        """The sibling comparison: marks the argmax-reward attempt (ties share —
        degenerate all-equal rollouts mark every sibling). Like every decorated env
        signal it runs once per target trace, and the return value is recorded on
        that trace's `metrics` under the method's name: `trace.metrics["best"]`."""
        return float(trace.reward == max(t.reward for t in traces))

    @vf.metric
    async def pass_at_n(self, trace, traces):
        """Whether any sibling reached the threshold — a rollout-level fact, so the
        same value is recorded as `pass_at_n` on every sibling's `metrics` (flat
        consumers see it without reconstructing the group)."""
        return float(max(t.reward for t in traces) >= self.params.threshold)
