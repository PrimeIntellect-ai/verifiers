"""tick-restore: a task whose every command takes the next number from a counter that a
process holds only in its memory, and appends it to a file. The file ends as 1..N only
if every command ran exactly once in a box that kept the process: a command that ran
twice, was skipped, or met a new process leaves another sequence.

Each command prints the file, so the tool results of two runs are equal only if the
boxes went through the same states. The command first sleeps, so a save requested when
a turn lands on the trace stops the box while that turn's command runs.
"""

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime

TURNS = 4
FIFO = "tick.fifo"
LOG = "ticks.txt"
HOLD_SECONDS = 3
SERVER = (
    f"i=0; rm -f {FIFO}; mkfifo {FIFO}; "
    f'while :; do i=$((i+1)); printf "%s\\n" "$i" > {FIFO}; sleep 0.2; done'
)
COMMAND = f"sleep {HOLD_SECONDS}; timeout 20 cat {FIFO} >> {LOG}; cat {LOG}"
EXPECTED = "\n".join(str(i) for i in range(1, TURNS + 1))


class TickRestoreTask(vf.Task):
    async def setup(self, trace: vf.Trace, runtime: Runtime) -> None:
        await runtime.run_background(["sh", "-c", SERVER], {}, "server.log")

    @vf.reward(weight=1.0)
    async def counted(self, trace: vf.Trace, runtime: Runtime) -> float:
        """The file holds 1..TURNS, in order."""
        try:
            result = await runtime.run(["sh", "-c", f"cat {LOG}"], {})
        except (SandboxError, OSError):
            return 0.0
        return float(result.stdout.strip() == EXPECTED)


class TickRestoreTaskset(vf.Taskset[TickRestoreTask, vf.TasksetConfig]):
    def load(self) -> list[TickRestoreTask]:
        return [
            TickRestoreTask(
                vf.TaskData(
                    idx=0,
                    prompt=f"Run `{COMMAND}` {TURNS} times, one command per turn.",
                )
            )
        ]


__all__ = ["TickRestoreTaskset"]
