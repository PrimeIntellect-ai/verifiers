"""reverse-text: reverse a string character-by-character (single-turn).

The model is asked to reverse the prompt and wrap the answer in `<reversed_text>`
tags; the reward is the LCS ratio between the parsed answer and the ground-truth
reversal. A pure-trace reward (no runtime needed) — the canonical tiny RL env, and
the starter for prime-rl training against a v1 env server.
"""

import re
from difflib import SequenceMatcher

import verifiers.v1 as vf

SYSTEM = (
    "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
)
_TAG = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)


class ReverseTextTask(vf.Task):
    answer: str
    """The ground-truth reversal of the prompt text."""


class ReverseTextConfig(vf.TasksetConfig):
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL"
    dataset_split: str = "train"


class ReverseTextTaskset(vf.Taskset[ReverseTextTask, ReverseTextConfig]):
    def load_tasks(self) -> list[ReverseTextTask]:
        from datasets import load_dataset

        rows = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        return [
            ReverseTextTask(
                idx=i,
                instruction=row["prompt"],
                system_prompt=SYSTEM,
                answer=row["prompt"][::-1],
            )
            for i, row in enumerate(rows)
        ]

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        # Reverse-text is single-turn: refuse a second turn so the model answers once.
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def lcs(self, task: ReverseTextTask, trace: vf.Trace) -> float:
        completion = (
            trace.assistant_messages[-1].content if trace.assistant_messages else ""
        )
        match = _TAG.search(completion or "")
        response = match.group(1).strip() if match else ""
        return SequenceMatcher(None, response, task.answer).ratio()


Config = ReverseTextConfig
Taskset = ReverseTextTaskset
