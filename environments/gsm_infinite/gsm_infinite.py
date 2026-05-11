import re
from typing import Any

from datasets import Dataset, load_dataset

import verifiers as vf

SYSTEM_PROMPT = (
    "Solve the GSM-Infinite problem step by step. End with the final result in "
    "the format `Answer: ...`."
)

DATASET_TEMPLATE = "InfiniAILab/gsm_infinite_{subset}_{context_length}"
SYMBOLIC_DATASET_TEMPLATE = "InfiniAILab/gsm_infinite_symbolic_example_{context_length}"


def dataset_name(subset: str = "medium", context_length: str = "0") -> str:
    if subset not in {"symbolic", "medium", "hard"}:
        raise ValueError("subset must be one of: symbolic, medium, hard")
    if subset == "symbolic":
        return SYMBOLIC_DATASET_TEMPLATE.format(context_length=context_length)
    return DATASET_TEMPLATE.format(subset=subset, context_length=context_length)


def split_name(operations: int = 2) -> str:
    return f"ops_{operations}"


def extract_answer(text: Any) -> str:
    if isinstance(text, list):
        return ", ".join(str(item) for item in text)
    text = str(text)
    match = re.search(r"ANSWER:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"Answer:\s*([^.\n]+)", text)
    if match is None:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return numbers[-1] if numbers else text.strip()
    return match.group(1).strip().rstrip(".")


def normalize_answer(text: Any) -> str:
    return re.sub(r"\s+", " ", extract_answer(text)).strip().lower()


def row_to_example(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    question = ""
    if messages:
        question = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        )
    if not question:
        question = (
            f"Problem: {row.get('problem', '')}\nQuestion: {row.get('question', '')}"
        )

    answer_source = row.get("answer_list", row.get("solution", ""))
    return {
        "question": question,
        "answer": extract_answer(answer_source),
        "info": {
            "benchmark": "gsm-infinite",
            "subset": row.get("d"),
            "operations": row.get("op"),
            "length": row.get("length"),
            "source_id": row.get("id"),
        },
    }


def load_gsm_infinite_dataset(
    subset: str = "medium",
    context_length: str = "0",
    operations: int = 2,
    num_examples: int = -1,
) -> Dataset:
    ds = load_dataset(
        dataset_name(subset=subset, context_length=context_length),
        split=split_name(operations),
    )
    ds = ds.map(row_to_example, remove_columns=ds.column_names)
    if num_examples != -1:
        ds = ds.select(range(min(num_examples, len(ds))))
    return ds


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


def exact_answer_reward(completion, answer, **kwargs) -> float:
    return float(
        normalize_answer(_completion_text(completion)) == normalize_answer(answer)
    )


def load_environment(
    subset: str = "medium",
    eval_subset: str | None = None,
    context_length: str = "0",
    eval_context_length: str | None = None,
    operations: int = 2,
    eval_operations: int | None = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    system_prompt: str = SYSTEM_PROMPT,
) -> vf.Environment:
    eval_subset = eval_subset or subset
    eval_context_length = eval_context_length or context_length
    eval_operations = eval_operations or operations

    rubric = vf.Rubric(funcs=[exact_answer_reward], weights=[1.0])
    return vf.SingleTurnEnv(
        dataset=lambda: load_gsm_infinite_dataset(
            subset=subset,
            context_length=context_length,
            operations=operations,
            num_examples=num_train_examples,
        ),
        eval_dataset=lambda: load_gsm_infinite_dataset(
            subset=eval_subset,
            context_length=eval_context_length,
            operations=eval_operations,
            num_examples=num_eval_examples,
        ),
        system_prompt=system_prompt,
        parser=vf.Parser(),
        rubric=rubric,
    )
