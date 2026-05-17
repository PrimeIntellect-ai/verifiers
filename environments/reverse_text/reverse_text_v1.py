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


@vf.reward(weight=1.0)
async def lcs_reward_func(task, state, extract_reversed_text) -> float:
    response = extract_reversed_text(state.get("completion") or [])
    answer = str(task["answer"])
    return SequenceMatcher(None, response, answer).ratio()


def extract_reversed_text() -> TagExtractor:
    return TagExtractor("reversed_text")


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
    source: str = f"{__name__}:source"
    system_prompt: str | None = (
        "Reverse the text character-by-character. Put your answer in "
        "<reversed_text> tags."
    )
    rewards: list[vf.CallableConfig] = [
        vf.CallableConfig(fn=f"{__name__}:lcs_reward_func")
    ]
    objects: dict[str, str] = {
        "extract_reversed_text": f"{__name__}:extract_reversed_text"
    }
    bindings: dict[str, str] = {
        "lcs_reward_func.extract_reversed_text": "objects.extract_reversed_text"
    }
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL"
    dataset_split: str = "train"


class ReverseTextEnvConfig(vf.EnvConfig):
    taskset: ReverseTextTasksetConfig = ReverseTextTasksetConfig()
    harness: vf.HarnessConfig = vf.HarnessConfig()


def load_taskset(
    config: ReverseTextTasksetConfig = ReverseTextTasksetConfig(),
) -> vf.Taskset:
    return vf.Taskset(config=config)


def load_v1_environment(
    config: ReverseTextEnvConfig = ReverseTextEnvConfig(),
) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )


load_environment = load_v1_environment
