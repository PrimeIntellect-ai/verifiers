"""agent-user-harness-v1: one turn of any harness running as a non-root agent user.

Root setup creates the task's `agent_user` (images normally ship one) and hands it the
workdir; the harness then installs, launches as that user, and answers a one-line prompt.
With the e2e suite's scripted upstream answering, the run is deterministic: it fails only if
the harness can't set up or run without root.
"""

import verifiers.v1 as vf
from verifiers.v1.runtimes import Runtime

__all__ = ["AgentUserHarnessTaskset"]

AGENT_USER = "vf-agent"


class AgentUserHarnessTask(vf.Task):
    async def setup(self, trace: vf.Trace, runtime: Runtime) -> None:
        setup = f"useradd -m {AGENT_USER} && chown -R {AGENT_USER} ."
        result = await runtime.run(["sh", "-c", setup], {})
        if result.exit_code != 0:
            raise RuntimeError(f"root setup failed: {result.stderr}")

    @vf.reward(weight=1.0)
    async def answered(self, trace: vf.Trace) -> float:
        return float(trace.num_turns >= 1)


class AgentUserHarnessTaskset(vf.Taskset[AgentUserHarnessTask, vf.TasksetConfig]):
    def load(self) -> list[AgentUserHarnessTask]:
        return [
            AgentUserHarnessTask(
                vf.TaskData(idx=0, prompt="Reply with OK.", agent_user=AGENT_USER)
            )
        ]
