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


extract_reversed_text = TagExtractor("reversed_text")


@vf.reward(weight=1.0)
async def lcs_reward_func(task, state) -> float:
    response = extract_reversed_text(state.get("completion") or [])
    answer = str(task["answer"])
    return SequenceMatcher(None, response, answer).ratio()


def build_source(
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL",
    dataset_split: str = "train",
):
    def source():
        def map_row(row):
            return {
                "question": row["prompt"],
                "answer": row["prompt"][::-1],
                "info": {},
            }

        dataset = load_dataset(dataset_name, split=dataset_split).map(map_row)
        dataset = dataset.remove_columns(["prompt"])
        for index, row in enumerate(dataset):
            yield {
                "example_id": index,
                "prompt": [{"role": "user", "content": row["question"]}],
                "question": row["question"],
                "answer": row["answer"],
                "info": row.get("info") or {},
            }

    return source


def load_taskset(
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL",
    dataset_split: str = "train",
    system_prompt: str | None = (
        "Reverse the text character-by-character. Put your answer in "
        "<reversed_text> tags."
    ),
    config=None,
):
    return vf.Taskset(
        source=build_source(dataset_name, dataset_split),
        system_prompt=system_prompt,
        rewards=[lcs_reward_func],
        config=config,
    )


def load_v1_environment(
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL",
    dataset_split: str = "train",
    system_prompt: str | None = (
        "Reverse the text character-by-character. Put your answer in "
        "<reversed_text> tags."
    ),
) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            system_prompt=system_prompt,
        )
    )
