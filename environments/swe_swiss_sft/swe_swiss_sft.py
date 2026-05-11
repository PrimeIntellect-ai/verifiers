import json
from difflib import SequenceMatcher
from typing import Any

from datasets import Dataset, load_dataset

import verifiers as vf

DEFAULT_DATASET_NAME = "SWE-Swiss/SWESwiss-SFT-Repair-4K"
DEFAULT_SPLIT = "train"

SYSTEM_PROMPT = (
    "You are a code repair agent. Respond with the requested SEARCH/REPLACE "
    "edits or solution text only."
)


def parse_messages(raw_messages: Any) -> list[dict[str, str]]:
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)
    if not isinstance(raw_messages, list):
        raise ValueError("SWE-Swiss rows must contain a messages list or JSON string")

    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")
        if role and content:
            messages.append({"role": role, "content": content})
    return messages


def row_to_example(row: dict[str, Any]) -> dict[str, Any]:
    messages = parse_messages(row["messages"])
    assistant_messages = [
        message for message in messages if message.get("role") == "assistant"
    ]
    if not assistant_messages:
        raise ValueError("SWE-Swiss row does not contain an assistant target")

    prompt_messages = []
    for message in messages:
        if message.get("role") == "assistant":
            break
        prompt_messages.append(message)

    if not prompt_messages:
        raise ValueError("SWE-Swiss row does not contain prompt messages")

    answer = assistant_messages[0]["content"]
    return {
        "prompt": prompt_messages,
        "question": prompt_messages[-1]["content"],
        "answer": answer,
        "info": {
            "benchmark": "swe-swiss",
            "target_role": "assistant",
            "source_index": row.get("__index_level_0__"),
        },
    }


def load_swe_swiss_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    num_examples: int = -1,
    shuffle_seed: int | None = None,
) -> Dataset:
    ds = load_dataset(dataset_name, split=split)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    if num_examples != -1:
        ds = ds.select(range(min(num_examples, len(ds))))
    return ds.map(row_to_example, remove_columns=ds.column_names)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(
            str(message.get("content", ""))
            for message in completion
            if isinstance(message, dict) and message.get("role") == "assistant"
        )
    return ""


def normalized_similarity_reward(completion, answer, **kwargs) -> float:
    response = " ".join(_completion_text(completion).split())
    target = " ".join(str(answer).split())
    if not response or not target:
        return 0.0
    return SequenceMatcher(None, response, target).ratio()


def load_environment(
    dataset_name: str = DEFAULT_DATASET_NAME,
    eval_dataset_name: str | None = None,
    split: str = DEFAULT_SPLIT,
    eval_split: str | None = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    shuffle_seed: int | None = 777,
    system_prompt: str = SYSTEM_PROMPT,
) -> vf.Environment:
    eval_dataset_name = eval_dataset_name or dataset_name
    eval_split = eval_split or split

    rubric = vf.Rubric(funcs=[normalized_similarity_reward], weights=[1.0])
    return vf.SingleTurnEnv(
        dataset=lambda: load_swe_swiss_dataset(
            dataset_name=dataset_name,
            split=split,
            num_examples=num_train_examples,
            shuffle_seed=shuffle_seed,
        ),
        eval_dataset=lambda: load_swe_swiss_dataset(
            dataset_name=eval_dataset_name,
            split=eval_split,
            num_examples=num_eval_examples,
            shuffle_seed=shuffle_seed,
        ),
        system_prompt=system_prompt,
        parser=vf.Parser(),
        rubric=rubric,
    )
