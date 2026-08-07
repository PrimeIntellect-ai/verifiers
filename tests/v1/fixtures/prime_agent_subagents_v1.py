"""Prime Agent subagent lifecycle and token accounting over preserved ACP `_meta`."""

from prime_agent_meta_guards import (
    child_tokens_attributed,
    no_outstanding_subagents,
    spawned_and_finished,
)

import verifiers.v1 as vf

TASK = (
    "Use IPython to spawn one subagent with rlm(). Ask it to reply to you, wait "
    "for its message, then reply with exactly DONE."
)


class PrimeAgentSubagentTask(vf.Task):
    @vf.reward(weight=1.0)
    async def subagent_lifecycle(self, trace: vf.Trace) -> float:
        # Each clause is independently necessary: a child that never finished, or
        # finished with no usage, or was still outstanding at scoring time, all
        # mean the rollout was scored against incomplete work.
        return float(
            spawned_and_finished(trace)
            and child_tokens_attributed(trace)
            and no_outstanding_subagents(trace)
        )


class PrimeAgentSubagentEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            await interaction.turn(TASK)
