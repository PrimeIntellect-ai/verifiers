"""shared-runtime-v1 - two agents borrow one provisioned runtime.

This example is intentionally small: the writer runs first and, in its task `finalize`,
writes the model's reply into the live runtime. The reader then runs in the same borrowed
runtime; its task `setup` reads the file before the model turn. The reader reward verifies
that the artifact it saw in setup matches what the writer wrote, and a deferred writer
reward mirrors that handoff result back onto the writer trace.
"""

from pydantic import SerializeAsAny

import verifiers.v1 as vf

NOTE_PATH = "shared/note.txt"

WRITE_PROMPT = """Reply with exactly this sentence: shared runtime handoff ready."""

READ_PROMPT = """A previous agent wrote an artifact into your borrowed runtime.

Reply with one concise sentence acknowledging that the handoff was inspected."""


class DirectAgentConfig(vf.AgentConfig):
    """Both agents use the cheap in-process direct harness.

    The shared piece here is not the harness process; it is the runtime borrowed by both
    rollouts. Change `--topology.writer.harness.runtime.*` to move the shared world.
    """

    harness: SerializeAsAny[vf.HarnessConfig] = vf.HarnessConfig(id="direct")


class WriteTask(vf.Task):
    """The seed task. Its `finalize` hook persists the writer's reply into the runtime."""

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        note = trace.last_reply.strip()
        await runtime.write(NOTE_PATH, note.encode())
        trace.info["shared_runtime"] = {
            "path": NOTE_PATH,
            "wrote": note,
            "runtime": runtime.descriptor,
        }


class ReadTask(vf.Task):
    """A downstream task that reads the writer's artifact from the borrowed runtime."""

    expected: str
    path: str = NOTE_PATH

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        observed = (await runtime.read(self.path)).decode().strip()
        trace.info["shared_runtime"] = {
            "path": self.path,
            "read": observed,
            "runtime": runtime.descriptor,
        }

    @vf.reward
    async def read_shared_note(self, trace: vf.Trace) -> float:
        observed = trace.info.get("shared_runtime", {}).get("read")
        return float(observed == self.expected)


class SharedRuntimeConfig(vf.TopologyConfig):
    writer: DirectAgentConfig = DirectAgentConfig()
    reader: DirectAgentConfig = DirectAgentConfig()


class SharedRuntimeTopology(vf.Topology[SharedRuntimeConfig]):
    def load_tasks(self) -> list[vf.Task]:
        return [WriteTask(idx=0, prompt=WRITE_PROMPT)]

    async def go(self, task: WriteTask, run: vf.TopologyRun) -> None:
        writer = run.agent("writer")
        reader = run.agent("reader")
        async with writer.provision(task) as runtime:
            written = await writer.run(task, runtime=runtime)
            note = written.info["shared_runtime"]["wrote"]
            await reader.run(
                ReadTask(idx=task.idx, prompt=READ_PROMPT, expected=note),
                parents=[written],
                runtime=runtime,
            )

    @vf.reward(agent="writer")
    async def handoff_succeeded(self, trace: vf.Trace, graph: vf.AgentGraph) -> float:
        children = graph.children(trace, agent="reader")
        return children[0].rewards.get("read_shared_note", 0.0) if children else 0.0


__all__ = ["SharedRuntimeTopology"]
