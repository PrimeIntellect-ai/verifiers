"""Prime Agent goal and continual-harness state over preserved ACP `_meta`."""

from prime_agent_meta_guards import MissingAcpMeta, goal_progressed, refinement_applied

import verifiers.v1 as vf

TASK = (
    "Run /refine to record that this environment prefers concise answers, then "
    "reply with exactly DONE."
)


def _any_metadata(trace) -> bool:
    """Whether the ACP envelope arrived at all, regardless of which keys it held."""
    recorded = (trace.info or {}).get("acp_meta") or {}
    return bool(recorded)


class PrimeAgentHarnessStateTask(vf.Task):
    @vf.reward(weight=1.0)
    async def harness_state(self, trace: vf.Trace) -> float:
        # Either surface proves continual-harness observability, so a missing
        # `refinement` envelope must fall through to `goal` rather than raising:
        # the guards raise on absent metadata, which would make this `or` dead.
        # Only raise when NEITHER surface is present, since that means the harness
        # preserved nothing and the run cannot be scored either way.
        seen = []
        for guard in (refinement_applied, goal_progressed):
            try:
                seen.append(guard(trace))
            except MissingAcpMeta:
                seen.append(False)
        if not any(seen) and not _any_metadata(trace):
            raise MissingAcpMeta(
                "neither refinement nor goal metadata was preserved for this rollout"
            )
        return float(any(seen))


class PrimeAgentHarnessStateEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            await interaction.turn(TASK)


class PrimeAgentHarnessStateTaskset(
    vf.Taskset[PrimeAgentHarnessStateTask, vf.TasksetConfig]
):
    def load(self) -> list[PrimeAgentHarnessStateTask]:
        return [
            PrimeAgentHarnessStateTask(
                vf.TaskData(
                    idx=0,
                    prompt=None,
                    system_prompt="Use /refine when asked to record a durable preference.",
                )
            )
        ]


__all__ = ["PrimeAgentHarnessStateEnv", "PrimeAgentHarnessStateTaskset"]
