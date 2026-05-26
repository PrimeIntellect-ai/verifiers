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


class ReverseTextTaskset(vf.Taskset[ReverseTextTasksetConfig]):
    def load_tasks(self) -> vf.Tasks:
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
        for index, row in enumerate(dataset):
            yield {
                "example_id": index,
                "prompt": [{"role": "user", "content": row["question"]}],
                "question": row["question"],
                "answer": row["answer"],
                "info": row.get("info") or {},
            }

    def load_system_prompt(self) -> vf.SystemPrompt:
        return (
            "Reverse the text character-by-character. Put your answer in "
            "<reversed_text> tags."
        )

    @vf.reward(weight=1.0)
    async def lcs_reward(self, task, state) -> float:
        response = REVERSED_TEXT_EXTRACTOR(state.get("completion") or [])
        answer = str(task["answer"])
        return SequenceMatcher(None, response, answer).ratio()


def load_taskset(config: ReverseTextTasksetConfig) -> ReverseTextTaskset:
    return ReverseTextTaskset(config=config)
