"""echo-agentic: write a phrase to a file with the bash tool, verified in the runtime.

Like harbor's hello-world but with no Dockerfile to build — it runs on the runtime's default
image, so the whole agentic loop works end to end: the model issues bash commands and the
reward reads the file back out of the runtime (`runtime.read`, runtime-opaque like harbor's
verifier). The minimal reward-1 agentic task for the e2e suite. Needs an agentic (shell) harness,
e.g. `bash`.
"""

import verifiers.v1 as vf
from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes import Runtime

SYSTEM = "You complete the task by running shell commands with the bash tool."
TARGET = "answer.txt"  # workdir-relative, so it's isolated per rollout on every runtime


def _key(text: str) -> str:
    """Lenient comparison key: lowercase, alphanumerics only."""
    return "".join(c for c in text.casefold() if c.isalnum())


def lenient_match(answer: str, text: str) -> bool:
    """True if `answer` appears in `text` ignoring case, spacing, and punctuation."""
    return _key(answer) in _key(text)


class EchoAgenticTask(vf.Task):
    answer: str
    """The phrase the model should write into the file."""


class EchoAgenticConfig(vf.TasksetConfig):
    phrase: str = "hello world"


class EchoAgenticTaskset(vf.Taskset[EchoAgenticTask, EchoAgenticConfig]):
    def load_tasks(self) -> list[EchoAgenticTask]:
        phrase = self.config.phrase
        return [
            EchoAgenticTask(
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
        self, task: EchoAgenticTask, trace: vf.Trace, runtime: Runtime
    ) -> float:
        try:
            content = (await runtime.read(TARGET)).decode(errors="replace")
        except (ProgramError, OSError, ValueError):
            return 0.0  # the model never wrote the file
        return float(lenient_match(task.answer, content))


__all__ = ["EchoAgenticTaskset"]
