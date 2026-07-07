"""echo-chain: two echo agents in sequence (fixture topology for the v1 e2e suite).

The smallest topology exercising every contract: agent `first` echoes the seed phrase
(seeds come from the `echo-v1` factory, pinned as the config default); a task for agent
`second` is derived from the first trace (forward arrow); `second` echoes the same phrase;
and `first` earns a deferred `relay` reward — declared, instance-end — equal to `second`'s
success (backward arrow). Both agents run the `null` harness, so a live run is fast and
deterministic: every trace should score 1.0 twice over.
"""

from pydantic import SerializeAsAny

import verifiers.v1 as vf
from echo_v1 import EchoConfig, EchoTask, lenient_match


class NullAgentConfig(vf.AgentConfig):
    """Pinned to the `null` chat loop — on a subclass, so partial overrides tune the pin
    (and a base-typed `HarnessConfig` pin is still detected, by value)."""

    harness: SerializeAsAny[vf.HarnessConfig] = vf.HarnessConfig(id="null")


class EchoChainConfig(vf.TopologyConfig):
    taskset: SerializeAsAny[vf.TasksetConfig] = EchoConfig(id="echo-v1")
    first: NullAgentConfig = NullAgentConfig()
    second: NullAgentConfig = NullAgentConfig()


class EchoChainTopology(vf.Topology[EchoChainConfig]):
    async def go(self, task: EchoTask, run: vf.TopologyRun) -> None:
        first = await run.rollout("first", task)
        # Forward arrow: the second agent must echo the same phrase — a task derived
        # from (and linked under) the first trace.
        derived = EchoTask(
            idx=task.idx,
            prompt=(
                "Do not call tools or execute code. Reply immediately and include "
                f"this exact phrase in your final response: {task.answer}"
            ),
            answer=task.answer,
        )
        await run.rollout("second", derived, parents=[first])

    @vf.reward(agent="first")
    async def relay(self, trace: vf.Trace, graph: vf.AgentGraph) -> float:
        """Backward arrow, declared: the first agent is rewarded for a relayable echo —
        its phrase made it through itself AND its second-agent child. Task-recorded rewards
        (`echoed`) are final by the time instance scoring runs, so reading the child's
        `reward` here is safe."""
        second = graph.children(trace, agent="second")
        relayed = lenient_match(trace.task.answer, trace.last_reply) and bool(
            second and second[0].reward > 0
        )
        return float(relayed)


__all__ = ["EchoChainTopology"]
