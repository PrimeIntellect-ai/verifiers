from __future__ import annotations

from difflib import SequenceMatcher

from datasets import load_dataset

import verifiers.v1 as vf
from verifiers.parsers.xml_parser import XMLParser

DEFAULT_DATASET_NAME = "PrimeIntellect/Reverse-Text-RL"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_SYSTEM_PROMPT = (
    "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
)

parser = XMLParser(["reversed_text"], answer_field="reversed_text")


@vf.reward(weight=1.0)
async def lcs_reward_func(task, state) -> float:
    response = parser.parse_answer(state.get("completion") or []) or ""
    answer = str(task["answer"])
    return SequenceMatcher(None, response, answer).ratio()


class ReverseTextTasksetConfig(vf.TasksetConfig):
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_split: str = DEFAULT_DATASET_SPLIT
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT


def _build_source(dataset_name: str, dataset_split: str):
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
    config: ReverseTextTasksetConfig | None = None,
) -> vf.Taskset:
    config = config or ReverseTextTasksetConfig()
    return vf.Taskset(
        source=_build_source(config.dataset_name, config.dataset_split),
        system_prompt=config.system_prompt,
        rewards=[lcs_reward_func],
        config=config,
    )


def load_harness(config: vf.HarnessConfig | None = None) -> vf.Harness:
    return vf.Harness(config=config)


def load_v1_environment(
    config: ReverseTextTasksetConfig | None = None,
) -> vf.Env:
    return vf.Env(taskset=load_taskset(config), harness=load_harness())
