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


@vf.reward(weight=1.0)
async def lcs_reward_func(task, state) -> float:
    response = REVERSED_TEXT_EXTRACTOR(state.get("completion") or [])
    answer = str(task["answer"])
    return SequenceMatcher(None, response, answer).ratio()


def source(
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL",
    dataset_split: str = "train",
):
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


class ReverseTextTasksetConfig(vf.TasksetConfig):
    system_prompt: str | None = (
        "Reverse the text character-by-character. Put your answer in "
        "<reversed_text> tags."
    )
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL"
    dataset_split: str = "train"


class ReverseTextTaskset(vf.Taskset[ReverseTextTasksetConfig]):
    _default_source = source
    _default_rewards = (lcs_reward_func,)


class ReverseTextEnvConfig(vf.EnvConfig):
    taskset: ReverseTextTasksetConfig = ReverseTextTasksetConfig()
    harness: vf.HarnessConfig = vf.HarnessConfig()


def load_environment(config: ReverseTextEnvConfig | None = None) -> vf.Env:
    config = config or ReverseTextEnvConfig()
    return vf.Env(taskset=ReverseTextTaskset(config=config.taskset))
