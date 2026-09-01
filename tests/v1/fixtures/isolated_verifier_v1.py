"""A shell task whose reward proves it ran after an isolated artifact transfer."""

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError

TARGET = "answer.txt"
SOLVER_MARKER = "/tmp/vf-solver-runtime"


class IsolatedVerifierData(vf.TaskData):
    answer: str


class IsolatedVerifierTask(vf.Task[IsolatedVerifierData]):
    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.info["solver_runtime"] = runtime.name
        await runtime.write(SOLVER_MARKER, b"solver")

    @vf.reward
    async def verified(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        trace.info["verifier_runtime"] = runtime.name
        try:
            await runtime.read(SOLVER_MARKER)
        except (SandboxError, OSError):
            pass
        else:
            return 0.0
        try:
            answer = (await runtime.read(TARGET)).decode().strip()
        except (SandboxError, OSError, UnicodeDecodeError):
            return 0.0
        return float(answer == self.data.answer)


class IsolatedVerifierTaskset(vf.Taskset[IsolatedVerifierTask, vf.TasksetConfig]):
    def load(self) -> list[IsolatedVerifierTask]:
        answer = "isolated verification works"
        return [
            IsolatedVerifierTask(
                IsolatedVerifierData(
                    idx=0,
                    prompt=f"Write exactly '{answer}' to {TARGET}, then finish.",
                    system_prompt="Use the bash tool to complete the task.",
                    answer=answer,
                    artifacts=[vf.Artifact(source=TARGET)],
                ),
                self.config.task,
            )
        ]


__all__ = ["IsolatedVerifierTaskset"]
