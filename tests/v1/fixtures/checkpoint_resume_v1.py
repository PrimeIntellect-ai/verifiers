"""checkpoint-resume: an agentic task whose first attempt dies mid-rollout, so the retry
resumes from the box's newest filesystem checkpoint instead of starting over.

`setup` stamps a nonce into the box that the prompt never reveals; the task asks the model
for two file steps that build on each other, and the reward reads the second file back
out of the box. A request `@intercept` raises a `ProviderError` once the first attempt has
made two turns — after the checkpoint of turn 2 exists, whose box holds step one's file.
The resumed attempt continues from that checkpoint: it never sees the injected failure
again (the trace already carries the error), inherits step one, and finishes. Needs an
agentic (shell) harness that can continue a conversation, e.g. `bash`, on a runtime with
checkpoints (docker, prime).
"""

import uuid

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime

SYSTEM = "You complete the task by running shell commands with the bash tool."
MARKER = "marker.txt"
STEP_ONE = "step1.txt"
DONE = "done.txt"
FAIL_AFTER_TURNS = 2


class CheckpointResumeTask(vf.Task[vf.TaskData]):
    async def setup(self, trace: vf.Trace, runtime: Runtime) -> None:
        nonce = uuid.uuid4().hex
        trace.info["nonce"] = nonce
        await runtime.write(MARKER, nonce.encode())

    @vf.intercept
    def fail_once(self, request: vf.Request, trace: vf.Trace) -> None:
        # Request hooks also judge each proposed tool call after a turn commits, over
        # the branch plus an empty probe result; fail only as a real turn goes upstream
        # (every step's command echoes, so a real result is never empty).
        last = request.messages[-1]
        probe = isinstance(last, vf.ToolMessage) and not last.content
        if trace.num_turns >= FAIL_AFTER_TURNS and not probe and not trace.errors:
            raise vf.ProviderError("injected mid-rollout failure", status_code=503)

    @vf.reward(weight=1.0)
    async def copied_marker(self, trace: vf.Trace, runtime: Runtime) -> float:
        try:
            content = (await runtime.read(DONE)).decode(errors="replace")
        except (SandboxError, OSError, ValueError):
            return 0.0
        return float(content.strip() == trace.info["nonce"])


class CheckpointResumeTaskset(vf.Taskset[CheckpointResumeTask, vf.TasksetConfig]):
    def load(self) -> list[CheckpointResumeTask]:
        return [
            CheckpointResumeTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        "Complete these steps in order, one bash command per turn, "
                        "and wait for each result before the next.\n"
                        f"1. Run `cp {MARKER} {STEP_ONE} && echo copied`.\n"
                        f"2. Run `cat {STEP_ONE} > {DONE} && echo written`.\n"
                        "3. Reply with the single word: done."
                    ),
                    system_prompt=SYSTEM,
                ),
                self.config.task,
            )
        ]


__all__ = ["CheckpointResumeTaskset"]
