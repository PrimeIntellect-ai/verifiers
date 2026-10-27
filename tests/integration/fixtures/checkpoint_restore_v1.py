"""checkpoint-restore: an agentic task whose first attempt dies mid-rollout, so the retry
restores the box's newest filesystem checkpoint instead of starting over.

`setup` stamps a nonce into the box that the prompt never reveals; the task asks the model
for two tool calls that build on each other (the second needs the first's token, so they
take two turns), and the reward reads the second one's file back out of the box. The
tools are colocated: they run in the box, so their files are in its snapshots, and any
harness with MCP tools can drive them. A request `@intercept` raises a `ProviderError`
once the first attempt has made two turns — after the checkpoint of turn 2 exists. The
restored attempt continues from that checkpoint in a new harness process: it never sees
the injected failure again (it fires once per task), finds the earlier calls' files, and
finishes. Needs a harness that can restore a checkpoint (`SUPPORTS_CHECKPOINT_RESTORE`)
on a runtime with checkpoints (prime).
"""

import shutil
import uuid
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime

MARKER = "marker.txt"
STEP_ONE = "step1.txt"
TOKEN = "token.txt"
DONE = "done.txt"
FAIL_AFTER_TURNS = 2


class RestoreState(vf.State):
    workdir: str = ""
    """The box's working directory, where `setup` put the marker."""


class StepsToolset(vf.Toolset[vf.ToolsetConfig, RestoreState]):
    TOOL_PREFIX = "steps"

    @vf.tool
    def copy_marker(self) -> str:
        """Copy the marker file to the step-one file. Returns the token `write_done`
        needs."""
        workdir = Path(self.state.workdir)
        shutil.copyfile(workdir / MARKER, workdir / STEP_ONE)
        token = uuid.uuid4().hex[:8]
        (workdir / TOKEN).write_text(token)
        return f"copied; token={token}"

    @vf.tool
    def write_done(self, token: str) -> str:
        """Write the step-one file's content to the done file. Needs the token that
        `copy_marker` returned."""
        workdir = Path(self.state.workdir)
        if not (workdir / TOKEN).is_file() or token != (workdir / TOKEN).read_text():
            return "wrong token: call copy_marker first and pass the token it returns"
        shutil.copyfile(workdir / STEP_ONE, workdir / DONE)
        return "written"


class CheckpointRestoreTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(colocated=True)


class CheckpointRestoreConfig(vf.TasksetConfig):
    task: CheckpointRestoreTaskConfig = CheckpointRestoreTaskConfig()


class CheckpointRestoreTask(
    vf.Task[vf.TaskData, RestoreState, CheckpointRestoreTaskConfig]
):
    failed = False
    """Whether the injected failure fired: once per task, so a retry from scratch
    (checkpoints off) also gets through."""

    @classmethod
    def toolsets(cls, config: CheckpointRestoreTaskConfig) -> list[vf.Toolset]:
        return [StepsToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: Runtime) -> None:
        nonce = uuid.uuid4().hex
        trace.info["nonce"] = nonce
        await runtime.write(MARKER, nonce.encode())
        result = await runtime.run(["pwd"], {})
        trace.state.workdir = result.stdout.strip()

    @vf.intercept
    def fail_once(self, request: vf.Request, trace: vf.Trace) -> None:
        # Request hooks also judge each proposed tool call after a turn commits, over
        # the branch plus an empty probe result; fail only as a real turn goes upstream
        # (every tool returns text, so a real result is never empty).
        last = request.messages[-1]
        probe = isinstance(last, vf.ToolMessage) and not last.content
        if trace.num_turns >= FAIL_AFTER_TURNS and not probe and not self.failed:
            self.failed = True
            # A 4xx: harness SDKs end the turn at once instead of retrying it until the
            # rollout's deadline.
            raise vf.ProviderError("injected mid-rollout failure", status_code=400)

    @vf.reward(weight=1.0)
    async def copied_marker(self, trace: vf.Trace, runtime: Runtime) -> float:
        try:
            content = (await runtime.read(DONE)).decode(errors="replace")
        except (SandboxError, OSError, ValueError):
            return 0.0
        return float(content.strip() == trace.info["nonce"])


class CheckpointRestoreTaskset(
    vf.Taskset[CheckpointRestoreTask, CheckpointRestoreConfig]
):
    def load(self) -> list[CheckpointRestoreTask]:
        return [
            CheckpointRestoreTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        "Use the tools of the `steps` MCP server. Make one tool call "
                        "per turn and wait for each result before the next.\n"
                        "1. Call `copy_marker`.\n"
                        "2. Call `write_done` with the token `copy_marker` returned.\n"
                        "3. Reply with the single word: done."
                    ),
                ),
                self.config.task,
            )
        ]


__all__ = ["CheckpointRestoreTaskset"]


if __name__ == "__main__":
    StepsToolset.run()
