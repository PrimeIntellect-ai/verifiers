"""Prime Agent autonomous gates observed through preserved ACP `_meta`."""

from prime_agent_meta_guards import autonomous_continued, gate_attempted

import verifiers.v1 as vf

# The gate must FAIL at least once, or autonomous never continues: a gate that
# passes immediately emits continuationsUsed 0 and no gateAttempt at all
# (verified against real gpt-5.6-luna), so this fixture would score 0 forever.
# The agent creates the file the gate demands only on a later pass.
TASK = "Reply with exactly DONE. Do not create any files unless a gate tells you to."


class PrimeAgentAutonomousGateTask(vf.Task):
    @vf.reward(weight=1.0)
    async def autonomous_gate(self, trace: vf.Trace) -> float:
        # A gate configured but never run scores zero: that is the silent-inert
        # failure this fixture exists to catch.
        return float(autonomous_continued(trace) and gate_attempted(trace))


class PrimeAgentAutonomousGateEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            await interaction.turn(TASK)


class PrimeAgentAutonomousGateTaskset(
    vf.Taskset[PrimeAgentAutonomousGateTask, vf.TasksetConfig]
):
    def load(self) -> list[PrimeAgentAutonomousGateTask]:
        return [
            PrimeAgentAutonomousGateTask(
                vf.TaskData(
                    idx=0,
                    prompt=None,
                    system_prompt="Follow instructions exactly; create files when asked.",
                )
            )
        ]


__all__ = ["PrimeAgentAutonomousGateEnv", "PrimeAgentAutonomousGateTaskset"]
