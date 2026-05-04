from __future__ import annotations

from difflib import SequenceMatcher

from datasets import load_dataset

import verifiers.v1 as vf
from verifiers.parsers.xml_parser import XMLParser

parser = XMLParser(["reversed_text"], answer_field="reversed_text")


@vf.reward(weight=1.0)
async def lcs_reward_func(task, state) -> float:
    response = parser.parse_answer(state.get("completion") or []) or ""
    answer = str(task["answer"])
    return SequenceMatcher(None, response, answer).ratio()


def build_source(
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL",
    dataset_split: str = "train",
    system_prompt: str | None = (
        "Reverse the text character-by-character. Put your answer in "
        "<reversed_text> tags."
    ),
):
    def source():
        dataset = load_dataset(dataset_name, split=dataset_split).map(
            lambda row: {
                "question": row["prompt"],
                "answer": row["prompt"][::-1],
                "info": {},
            }
        )
        dataset = dataset.remove_columns(["prompt"])
        for index, row in enumerate(dataset):
            prompt = []
            if system_prompt:
                prompt.append({"role": "system", "content": system_prompt})
            prompt.append({"role": "user", "content": row["question"]})
            yield {
                "example_id": index,
                "prompt": prompt,
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
        source=build_source(dataset_name, dataset_split, system_prompt),
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
