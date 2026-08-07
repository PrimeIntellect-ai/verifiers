"""Prime Agent failures that must be loud.

Every bug this integration hit scored *wrongly* rather than erroring: a starved
follow-up, an ACP turn that reported a clean `end_turn` after failing, an agent
inheriting the runner's uv interpreter. Each looked like a weak model instead of
broken infrastructure.

So these guards convert "the evidence is missing" into an exception. A fixture
that silently returns 0.0 when metadata is absent is indistinguishable from a
fixture whose agent genuinely failed, and that ambiguity is what hides bugs.
"""

from prime_agent_meta_guards import (
    MissingAcpMeta,
    field_history,
    gate_failure_reported,
    observed_child_statuses,
)

import verifiers.v1 as vf


def child_error_reported(trace) -> bool:
    """A killed child surfaced a terminal error rather than vanishing."""
    return any(
        seen[-1] in {"error", "cancelled"}
        for seen in observed_child_statuses(trace).values()
    )


def quiescence_blocked_scoring(trace) -> bool:
    """Quiescence reported outstanding work at some point.

    This is the signal a consumer needs in order to refuse to score early; if it
    never appears, a harness cannot tell a finished turn from a busy one.
    """
    return any(
        isinstance(event.get("outstandingSubagents"), int)
        and event["outstandingSubagents"] > 0
        for event in field_history(trace, "quiescence")
    )


class PrimeAgentKilledChildTask(vf.Task):
    @vf.reward(weight=1.0)
    async def killed_child_is_loud(self, trace: vf.Trace) -> float:
        # Absent metadata raises out of the reward rather than scoring 0.0, so a
        # harness regression fails the run instead of blaming the model.
        return float(child_error_reported(trace))


class PrimeAgentKilledChildEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            await interaction.turn(
                "Use IPython to spawn a subagent with rlm() that sleeps for a long "
                "time. Then delete it with rlm.delete_subagent() before it finishes "
                "and reply with exactly DONE."
            )


class PrimeAgentFailingGateTask(vf.Task):
    @vf.reward(weight=1.0)
    async def failing_gate_is_loud(self, trace: vf.Trace) -> float:
        return float(gate_failure_reported(trace))


class PrimeAgentFailingGateEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            await interaction.turn(
                "Create a file named gate.txt containing ok, then reply with exactly DONE."
            )


__all__ = [
    "MissingAcpMeta",
    "PrimeAgentFailingGateEnv",
    "PrimeAgentFailingGateTask",
    "PrimeAgentKilledChildEnv",
    "PrimeAgentKilledChildTask",
    "child_error_reported",
    "quiescence_blocked_scoring",
]
