"""subagent-restore: a task whose agent hands two jobs to two sub-agents that run at the
same time, so a box snapshotted in the middle holds two running sub-agents.

`setup` starts one process per sub-agent that holds a nonce only in its memory and
hands it to whoever reads its FIFO. Each sub-agent reads the nonce twice, one command
per turn. The second command first waits for the other sub-agent's first file and then
sleeps, so from the moment both sub-agents have a turn on the trace, neither finishes
for `HOLD_SECONDS`: a snapshot taken then catches both at work. The second read comes
from the process, so it works only in a box that kept its memory.
"""

import uuid

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime

NAMES = ("alpha", "bravo")
ROOT_MARK = "COORDINATOR-TASK"
HOLD_SECONDS = 20
SERVER = 'rm -f "$2"; mkfifo "$2"; while :; do printf %s "$1" > "$2"; sleep 0.5; done'


def mark(name: str) -> str:
    """The text only `name`'s brief holds."""
    return f"SUBAGENT-{name.upper()}"


def brief(name: str) -> str:
    other = NAMES[1 - NAMES.index(name)]
    return (
        f"{mark(name)}. Run these shell commands in the working directory, one "
        "command per turn, and wait for each result before the next.\n"
        f"1. `timeout 10 cat {name}.fifo > {name}.1`\n"
        f"2. `timeout 120 sh -c 'until [ -f {other}.1 ]; do sleep 1; done'; "
        f"sleep {HOLD_SECONDS}; timeout 10 cat {name}.fifo > {name}.done`\n"
        "3. Reply with the single word: done."
    )


def sub_agent_turns(trace: vf.Trace) -> dict[str, list[int]]:
    """The sampled nodes of each sub-agent: those whose history holds the sub-agent's
    brief and not the coordinator's task."""
    turns: dict[str, list[int]] = {name: [] for name in NAMES}
    for index, node in enumerate(trace.nodes):
        if not node.sampled:
            continue
        history, cursor = "", node.parent
        while cursor is not None:
            history += str(trace.nodes[cursor].message.model_dump())
            cursor = trace.nodes[cursor].parent
        if ROOT_MARK in history:
            continue
        for name in NAMES:
            if mark(name) in history:
                turns[name].append(index)
    return turns


class SubagentRestoreTask(vf.Task):
    async def setup(self, trace: vf.Trace, runtime: Runtime) -> None:
        for name in NAMES:
            nonce = uuid.uuid4().hex
            trace.info[name] = nonce
            await runtime.run_background(
                ["sh", "-c", SERVER, "sh", nonce, f"{name}.fifo"], {}, f"{name}.log"
            )

    async def _read(self, runtime: Runtime, command: str) -> str:
        try:
            result = await runtime.run(["sh", "-c", command], {})
        except (SandboxError, OSError):
            return ""
        return result.stdout.strip()

    @vf.reward(weight=1.0)
    async def both_done(self, trace: vf.Trace, runtime: Runtime) -> float:
        """Each sub-agent's second read left its nonce in its done file."""
        done = [
            await self._read(runtime, f"cat {name}.done") == trace.info[name]
            for name in NAMES
        ]
        return float(all(done))


class SubagentRestoreTaskset(vf.Taskset[SubagentRestoreTask, vf.TasksetConfig]):
    def load(self) -> list[SubagentRestoreTask]:
        briefs = "\n\n".join(
            f"Brief for sub-agent {i + 1}:\n{brief(name)}"
            for i, name in enumerate(NAMES)
        )
        return [
            SubagentRestoreTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        f"{ROOT_MARK}. Do not run the commands below yourself. Start "
                        "exactly two sub-agents so that both run at the same time, "
                        "and give each one its brief word for word. Wait for both, "
                        "then reply with the single word: done.\n\n"
                        f"{briefs}"
                    ),
                )
            )
        ]


__all__ = ["SubagentRestoreTaskset"]
