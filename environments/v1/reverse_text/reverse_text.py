"""Pure-v1 ``reverse-text`` environment.

The whole env surface is a Taskset + a ``load_taskset`` factory. There is no
``EnvConfig`` subclass, no ``load_environment``, and no ``load_harness``;
``vf-eval`` auto-resolves the taskset config and pairs the taskset with the
base ``verifiers.v1.Harness`` (or any harness selected via the second
positional / ``--harness.name``).

Loadable through the v1 CLI (``vf-eval-v1 reverse-text``) or
``vf.load_environment("reverse-text")``. The legacy ``vf-eval`` expects a
``load_environment`` function and will not find one here.
"""

import re
from difflib import SequenceMatcher

from datasets import load_dataset

import verifiers as vf


class TagExtractor:
    def __init__(self, tag: str):
        self.pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)

    def __call__(self, completion: list[vf.ConfigData]) -> str:
        messages = vf.get_messages(completion, role="assistant")
        if not messages:
            return ""
        message = messages[-1]
        match = self.pattern.search(str(message.content or ""))
        return match.group(1).strip() if match else ""


REVERSED_TEXT_EXTRACTOR = TagExtractor("reversed_text")


class ReverseTextTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL"
    dataset_split: str = "train"
    instruction: str = (
        "Reverse the text character-by-character. Put your answer in "
        "<reversed_text> tags."
    )


class ReverseTextTaskset(vf.Taskset[ReverseTextTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        def map_row(row):
            return {
                "question": row["prompt"],
                "answer": row["prompt"][::-1],
                "info": {},
            }

        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
        ).map(map_row)
        dataset = dataset.remove_columns(["prompt"])
        instruction = self.config.instruction
        for index, row in enumerate(dataset):
            question = row["question"]
            content = f"{instruction}\n\n{question}" if instruction else question
            yield {
                "example_id": index,
                "prompt": [{"role": "user", "content": content}],
                "question": question,
                "answer": row["answer"],
                "info": row.get("info") or {},
            }

    @vf.reward(weight=1.0)
    async def lcs_reward(self, task, state) -> float:
        response = REVERSED_TEXT_EXTRACTOR(state.get("completion") or [])
        answer = str(task["answer"])
        return SequenceMatcher(None, response, answer).ratio()


def load_taskset(config: ReverseTextTasksetConfig) -> ReverseTextTaskset:
    return ReverseTextTaskset(config=config)
