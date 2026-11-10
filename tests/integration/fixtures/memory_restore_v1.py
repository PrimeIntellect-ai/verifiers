"""memory-restore: a task whose first attempt dies mid-rollout, so the retry restores the
box's newest memory snapshot and continues the harness program the box still runs.

`setup` starts a process in the box that holds a nonce only in its memory and hands it
to whoever reads `nonce.fifo`. The nonce is on no disk, so only a box that continues
the process from a memory snapshot can answer.
The reward reads the nonce from the process. The task asks a harness with a shell for
two commands that build on each other, and a metric checks the file they leave.

A response `@intercept` raises a `ProviderError` once, at the model turn `fail_turn`:
the harness gets the error as its reply and dies, after the snapshot of that turn
exists. The restored box still waits for that turn's reply, gets it from the retry, and
finishes: no new harness process, no notice to the model.
"""

import uuid

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime

FIFO = "nonce.fifo"
STEP_ONE = "step1.txt"
DONE = "done.txt"
READ = f"timeout 10 cat {FIFO}"
# The nonce is the script's first argument: it lives in the process, not in a file. The
# sleep lets a reader see the end of one copy before the next write opens the FIFO.
SERVER = (
    f'rm -f {FIFO}; mkfifo {FIFO}; while :; do printf %s "$1" > {FIFO}; sleep 0.5; done'
)


class MemoryRestoreTaskConfig(vf.TaskConfig):
    fail_turn: int = 2
    """The model turn whose reply fails once (from 1)."""


class MemoryRestoreConfig(vf.TasksetConfig):
    task: MemoryRestoreTaskConfig = MemoryRestoreTaskConfig()


class MemoryRestoreTask(vf.Task[vf.TaskData, vf.State, MemoryRestoreTaskConfig]):
    failed = False
    """Whether the injected failure fired: once per task, so a retry from the start
    (checkpoints off) also gets through."""

    async def setup(self, trace: vf.Trace, runtime: Runtime) -> None:
        nonce = uuid.uuid4().hex
        trace.info["nonce"] = nonce
        await runtime.run_background(
            ["sh", "-c", SERVER, "sh", nonce], {}, "server.log"
        )

    @vf.intercept
    def fail_once(self, response: vf.Response, trace: vf.Trace) -> None:
        if trace.num_turns + 1 >= self.config.fail_turn and not self.failed:
            self.failed = True
            # A 4xx: harness SDKs end the turn at once instead of retrying it until the
            # rollout's deadline.
            raise vf.ProviderError("injected mid-rollout failure", status_code=400)

    async def _read(self, runtime: Runtime, command: str) -> str:
        try:
            result = await runtime.run(["sh", "-c", command], {})
        except (SandboxError, OSError):
            return ""
        return result.stdout.strip()

    @vf.reward(weight=1.0)
    async def process_alive(self, trace: vf.Trace, runtime: Runtime) -> float:
        """The process `setup` started still runs and still holds the nonce."""
        return float(await self._read(runtime, READ) == trace.info["nonce"])

    @vf.metric
    async def copied(self, trace: vf.Trace, runtime: Runtime) -> float:
        """The agent's two commands left the nonce in the done file."""
        return float(await self._read(runtime, f"cat {DONE}") == trace.info["nonce"])


class MemoryRestoreTaskset(vf.Taskset[MemoryRestoreTask, MemoryRestoreConfig]):
    def load(self) -> list[MemoryRestoreTask]:
        return [
            MemoryRestoreTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        "Run these shell commands in the working directory, one "
                        "command per turn, and wait for each result before the "
                        "next. If you cannot run commands, reply with the single "
                        "word: done.\n"
                        f"1. `{READ} > {STEP_ONE}`\n"
                        f"2. `cp {STEP_ONE} {DONE}`\n"
                        "3. Reply with the single word: done."
                    ),
                ),
                self.config.task,
            )
        ]


__all__ = ["MemoryRestoreTaskset"]
