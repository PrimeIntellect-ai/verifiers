"""agentic-echo: write a phrase to a file with the bash tool, verified in the runtime.

Like harbor's hello-world but with no Dockerfile to build — it runs on the runtime's default
image, so the whole agentic loop works end to end: the model issues bash commands and the
reward reads the file back out of the runtime (`runtime.read`, runtime-opaque like harbor's
verifier). The minimal reward-1 agentic task for the e2e suite. Needs a harness with the bash
tool (`--harness.enable-bash true`).
"""

import verifiers.v1 as vf
from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes import Runtime

SYSTEM = "You complete the task by running shell commands with the bash tool."
TARGET = "answer.txt"  # workdir-relative, so it's isolated per rollout on every runtime


def lenient_match(answer: str, text: str) -> bool:
    """True if `answer` appears in `text` ignoring case, spacing, and punctuation."""
    key = lambda s: "".join(c for c in s.casefold() if c.isalnum())
    return key(answer) in key(text)


class AgenticEchoTask(vf.Task):
    answer: str
    """The phrase the model should write into the file."""


class AgenticEchoConfig(vf.TasksetConfig):
    phrase: str = "hello world"


class AgenticEchoTaskset(vf.Taskset[AgenticEchoTask, AgenticEchoConfig]):
    def load_tasks(self) -> list[AgenticEchoTask]:
        phrase = self.config.phrase
        return [
            AgenticEchoTask(
                idx=0,
                instruction=(
                    f"Use the bash tool to write exactly the text '{phrase}' to a file named "
                    f"{TARGET} in the current directory, then finish."
                ),
                system_prompt=SYSTEM,
                answer=phrase,
            )
        ]

    @vf.reward(weight=1.0)
    async def wrote_phrase(
        self, task: AgenticEchoTask, trace: vf.Trace, runtime: Runtime
    ) -> float:
        try:
            content = (await runtime.read(TARGET)).decode(errors="replace")
        except (ProgramError, OSError, ValueError):
            return 0.0  # the model never wrote the file
        return float(lenient_match(task.answer, content))


def load_taskset(config: AgenticEchoConfig) -> AgenticEchoTaskset:
    return AgenticEchoTaskset(config)
