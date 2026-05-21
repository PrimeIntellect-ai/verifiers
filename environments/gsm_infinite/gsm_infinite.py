import re
from collections.abc import Iterable
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

import verifiers as vf


DATASET_PREFIXES = {
    "symbolic": "InfiniAILab/gsm_infinite_symbolic",
    "medium": "InfiniAILab/gsm_infinite_medium",
    "hard": "InfiniAILab/gsm_infinite_hard",
}


def _dataset_name(subset: str, context_length: str) -> str:
    if subset not in DATASET_PREFIXES:
        raise ValueError(f"Unknown GSM-Infinite subset: {subset}")
    return f"{DATASET_PREFIXES[subset]}_{context_length}"


def _normalize_op_splits(op_splits: str | Iterable[str]) -> list[str]:
    if isinstance(op_splits, str):
        return [split.strip() for split in op_splits.split(",") if split.strip()]
    return [str(split) for split in op_splits]


def _select_examples(dataset: Dataset, count: int) -> Dataset:
    if count == -1:
        return dataset
    return dataset.select(range(min(count, len(dataset))))


def _solution_integer(solution: str) -> str:
    match = re.search(r"answer:\s*(-?\d+)", solution, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Could not extract integer answer from GSM-Infinite solution")
    return match.group(1)


def _extract_integer_answer(text: str) -> str | None:
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    patterns = [
        r"\\boxed\{\s*(-?\d+)\s*\}",
        r"answer:\s*(-?\d+)",
        r"solution:\s*(-?\d+)",
        r"final answer:\s*(-?\d+)",
        r"\\text\{answer:\s*\}\s*(-?\d+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1]
    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else None


def _extract_symbolic_answer(text: str) -> list[str]:
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if re.search(r"\bnone\b", text, flags=re.IGNORECASE):
        return []
    return sorted(set(re.findall(r"\bV\d+\b", text)))


def _format_row(example: dict[str, Any], subset: str) -> dict[str, Any]:
    answer = (
        sorted(example["answer_list"])
        if subset == "symbolic"
        else _solution_integer(example["solution"])
    )
    info = {
        "subset": subset,
        "op": example.get("op"),
        "length": example.get("length"),
        "id": example.get("id"),
        "answer": answer,
    }
    if subset != "symbolic":
        info.update(
            {
                "template": example.get("template"),
                "mode": example.get("mode"),
            }
        )
    return {
        "prompt": example["messages"],
        "answer": answer,
        "info": info,
    }


def _load_gsm_infinite_dataset(
    subset: str,
    context_length: str,
    op_splits: str | Iterable[str],
    num_examples: int,
) -> Dataset:
    dataset_name = _dataset_name(subset, context_length)
    datasets = []
    for split in _normalize_op_splits(op_splits):
        dataset = load_dataset(dataset_name, split=split)
        remove_columns = [
            column
            for column in dataset.column_names
            if column not in {"messages", "solution", "answer_list", "op", "length", "id"}
        ]
        if subset != "symbolic":
            remove_columns = [
                column
                for column in remove_columns
                if column not in {"template", "mode"}
            ]
        dataset = dataset.map(
            lambda example, subset=subset: _format_row(example, subset),
            remove_columns=remove_columns,
        )
        datasets.append(_select_examples(dataset, num_examples))
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


class GSMInfiniteRubric(vf.Rubric):
    def __init__(self):
        super().__init__(funcs=[self.gsm_infinite_reward])

    def gsm_infinite_reward(self, parser, completion, info, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        if info["subset"] == "symbolic":
            predicted = _extract_symbolic_answer(response)
            return float(predicted == sorted(info["answer"]))
        predicted_integer = _extract_integer_answer(response)
        return float(predicted_integer == str(info["answer"]))


def load_environment(
    subset: str = "medium",
    context_length: str = "0",
    train_op_splits: str | Iterable[str] = "ops_2",
    eval_op_splits: str | Iterable[str] = "ops_3",
    num_train_examples: int = 100,
    num_eval_examples: int = 100,
    system_prompt: str | None = None,
) -> vf.Environment:
    def build_dataset() -> Dataset:
        return _load_gsm_infinite_dataset(
            subset=subset,
            context_length=context_length,
            op_splits=train_op_splits,
            num_examples=num_train_examples,
        )

    def build_eval_dataset() -> Dataset:
        return _load_gsm_infinite_dataset(
            subset=subset,
            context_length=context_length,
            op_splits=eval_op_splits,
            num_examples=num_eval_examples,
        )

    parser = vf.Parser()
    rubric = GSMInfiniteRubric()
    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
