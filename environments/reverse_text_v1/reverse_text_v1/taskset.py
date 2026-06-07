import re
from difflib import SequenceMatcher

from datasets import load_dataset

import verifiers.v1 as vf


class TagExtractor:
    def __init__(self, tag: str):
        self.pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)

    def __call__(self, completion: vf.Messages) -> str:
        messages = [message for message in completion if message.role == "assistant"]
        if not messages:
            return ""
        message = messages[-1]
        match = self.pattern.search(str(message.content or ""))
        return match.group(1).strip() if match else ""


REVERSED_TEXT_EXTRACTOR = TagExtractor("reversed_text")


class ReverseTextTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL"
    dataset_split: str = "train"
    system_prompt: vf.SystemPrompt = (
        "Reverse the text character-by-character. Put your answer in "
        "<reversed_text> tags."
    )


class ReverseTextTask(vf.Task):
    question: str
    answer: str


class ReverseTextTaskset(vf.Taskset[ReverseTextTasksetConfig]):
    task_type = ReverseTextTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        def map_row(row):
            return {
                "question": row["prompt"],
                "answer": row["prompt"][::-1],
            }

        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
        ).map(map_row)
        dataset = dataset.remove_columns(["prompt"])
        for index, row in enumerate(dataset):
            yield {
                "row_id": index,
                "prompt": [{"role": "user", "content": row["question"]}],
                "question": row["question"],
                "answer": row["answer"],
            }

    @vf.reward(weight=1.0)
    async def lcs_reward(self, task: ReverseTextTask, state: vf.State) -> float:
        response = REVERSED_TEXT_EXTRACTOR(state.completion or [])
        return SequenceMatcher(None, response, task.answer).ratio()


def load_taskset(config: ReverseTextTasksetConfig) -> ReverseTextTaskset:
    return ReverseTextTaskset(config=config)
