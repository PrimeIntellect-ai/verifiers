"""Prime Agent goal and continual-harness state over preserved ACP `_meta`."""

from prime_agent_meta_guards import goal_progressed, refinement_applied

import verifiers.v1 as vf

TASK = (
    "Run /refine to record that this environment prefers concise answers, then "
    "reply with exactly DONE."
)


class PrimeAgentHarnessStateTask(vf.Task):
    @vf.reward(weight=1.0)
    async def harness_state(self, trace: vf.Trace) -> float:
        # Either surface proves continual-harness observability; requiring both
        # would couple this fixture to whether the task also set a goal.
        return float(refinement_applied(trace) or goal_progressed(trace))


class PrimeAgentHarnessStateEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            await interaction.turn(TASK)
