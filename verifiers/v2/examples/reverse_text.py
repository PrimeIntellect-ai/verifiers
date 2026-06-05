"""reverse-text: a single-turn environment.

The model is asked to reverse a string and wrap the result in <reversed_text>
tags; the reward is the character-level similarity to the true reversal. The
instruction is baked into the single user prompt at load time.
"""

import re
from difflib import SequenceMatcher

import verifiers.v2 as vf

TAG = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)
INSTRUCTION = (
    "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
)
WORDS = ["hello world", "reverse me", "prime intellect"]


class ReverseTextTask(vf.Task):
    answer: str
    """The correct reversal."""


class ReverseTextTaskset(vf.Taskset[ReverseTextTask, vf.TasksetConfig]):
    def load_tasks(self) -> list[ReverseTextTask]:
        return [
            ReverseTextTask(
                id=str(i),
                instruction=f"{INSTRUCTION}\n\n{word}",
                answer=word[::-1],
            )
            for i, word in enumerate(WORDS)
        ]

    @vf.reward(weight=1.0)
    async def similarity(
        self, task: ReverseTextTask, transcript: vf.Transcript
    ) -> float:
        last = next(
            (m.content for m in reversed(transcript.messages) if m.role == "assistant"),
            "",
        )
        match = TAG.search(last or "")
        predicted = match.group(1).strip() if match else ""
        return SequenceMatcher(None, predicted, task.answer).ratio()


def load_taskset(config: vf.TasksetConfig | None = None) -> ReverseTextTaskset:
    return ReverseTextTaskset(config or vf.TasksetConfig())


def load_harness(config: vf.HarnessConfig | None = None) -> vf.Harness:
    return vf.Harness(config or vf.HarnessConfig(max_turns=1))


def load_environment(config: vf.EnvConfig | None = None) -> vf.Environment:
    config = config or vf.EnvConfig()
    return vf.Environment(
        taskset=load_taskset(config.taskset),
        harness=load_harness(config.harness),
    )
