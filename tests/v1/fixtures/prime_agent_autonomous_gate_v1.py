"""Prime Agent autonomous gates observed through preserved ACP `_meta`."""

from prime_agent_meta_guards import autonomous_continued, gate_attempted

import verifiers.v1 as vf

TASK = "Create a file named gate.txt containing ok, then reply with exactly DONE."


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
