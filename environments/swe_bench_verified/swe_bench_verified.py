from __future__ import annotations

import re
from collections.abc import Mapping
from difflib import SequenceMatcher
from typing import Any

from datasets import Dataset, load_dataset

import verifiers.v1 as vf

DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
TASKSET_ID = "swe-bench/verified"

SYSTEM_PROMPT = """\
You are repairing a real GitHub repository issue from SWE-bench Verified.
Return only a unified diff that applies to the repository at the specified base
commit. Wrap the diff in <patch>...</patch> tags."""


def format_prompt(row: Mapping[str, Any]) -> str:
    hints = str(row.get("hints_text") or "").strip()
    hints_block = f"\n\nHints:\n{hints}" if hints else ""
    return (
        f"Repository: {row['repo']}\n"
        f"Instance: {row['instance_id']}\n"
        f"Base commit: {row['base_commit']}\n"
        f"Difficulty: {row.get('difficulty') or 'unknown'}\n\n"
        "Problem statement:\n"
        f"{str(row['problem_statement']).strip()}"
        f"{hints_block}\n\n"
        "Return the minimal source-code patch as a unified diff."
    )


def build_record(row: Mapping[str, Any]) -> dict[str, Any]:
    info_keys = (
        "repo",
        "instance_id",
        "base_commit",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "environment_setup_commit",
        "difficulty",
        "created_at",
        "version",
    )
    info = {key: row.get(key) for key in info_keys if key in row}
    return {
        "task_id": row["instance_id"],
        "prompt": [{"role": "user", "content": format_prompt(row)}],
        "question": format_prompt(row),
        "answer": row["patch"],
        "info": info,
    }


def load_rows(
    dataset_name: str,
    split: str,
    limit: int | None,
    repos: list[str] | None,
    difficulties: list[str] | None,
    keep_in_memory: bool,
) -> Dataset:
    dataset = load_dataset(dataset_name, split=split, keep_in_memory=keep_in_memory)
    if repos:
        allowed_repos = frozenset(repos)
        dataset = dataset.filter(lambda row: row["repo"] in allowed_repos)
    if difficulties:
        allowed_difficulties = frozenset(difficulties)
        dataset = dataset.filter(
            lambda row: str(row.get("difficulty") or "") in allowed_difficulties
        )
    if limit is not None and limit >= 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset.map(build_record, remove_columns=dataset.column_names)


def extract_patch(completion: object) -> str:
    text = completion_to_text(completion)
    match = re.search(r"<patch>\s*(.*?)\s*</patch>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def completion_to_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, Mapping):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    if completion is None:
        return ""
    return str(completion)


def normalize_patch(patch: str) -> str:
    lines = []
    for raw_line in patch.replace("\r\n", "\n").splitlines():
        line = raw_line.rstrip()
        if line.startswith("index "):
            continue
        lines.append(line)
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


async def exact_patch(task: vf.Task, state: vf.State) -> float:
    expected = normalize_patch(str(task["answer"]))
    actual = normalize_patch(extract_patch(state.get("completion")))
    return 1.0 if actual == expected else 0.0


async def patch_similarity(task: vf.Task, state: vf.State) -> float:
    expected = normalize_patch(str(task["answer"]))
    actual = normalize_patch(extract_patch(state.get("completion")))
    if not expected or not actual:
        return 0.0
    return SequenceMatcher(None, actual, expected).ratio()


async def patch_line_count(task: vf.Task, state: vf.State) -> float:
    patch = normalize_patch(extract_patch(state.get("completion")))
    return float(len([line for line in patch.splitlines() if line]))


async def gold_patch_line_count(task: vf.Task, state: vf.State) -> float:
    patch = normalize_patch(str(task["answer"]))
    return float(len([line for line in patch.splitlines() if line]))


def load_taskset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    eval_split: str | None = None,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    repos: list[str] | None = None,
    difficulties: list[str] | None = None,
    keep_in_memory: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
    exact_weight: float = 1.0,
    similarity_weight: float = 0.0,
    config: vf.TasksetConfig | None = None,
) -> vf.Taskset:
    def build_train() -> Dataset:
        return load_rows(
            dataset_name,
            split,
            train_limit,
            repos,
            difficulties,
            keep_in_memory,
        )

    def build_eval() -> Dataset:
        return load_rows(
            dataset_name,
            eval_split or split,
            eval_limit,
            repos,
            difficulties,
            keep_in_memory,
        )

    rewards = []
    metrics = [patch_similarity, patch_line_count, gold_patch_line_count]
    if exact_weight > 0:
        rewards.append(vf.reward(weight=exact_weight)(exact_patch))
    else:
        metrics.insert(0, exact_patch)
    if similarity_weight > 0:
        rewards.append(vf.reward(weight=similarity_weight)(patch_similarity))

    return vf.Taskset(
        source=build_train,
        eval_source=build_eval,
        taskset_id=TASKSET_ID,
        system_prompt=system_prompt,
        rewards=rewards,
        metrics=metrics,
        config=config,
    )


def load_environment(
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    eval_split: str | None = None,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    repos: list[str] | None = None,
    difficulties: list[str] | None = None,
    keep_in_memory: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
    exact_weight: float = 1.0,
    similarity_weight: float = 0.0,
    config: vf.EnvConfig | None = None,
) -> vf.Env:
    config = config or vf.EnvConfig()
    return vf.Env(
        taskset=load_taskset(
            dataset_name=dataset_name,
            split=split,
            eval_split=eval_split,
            train_limit=train_limit,
            eval_limit=eval_limit,
            repos=repos,
            difficulties=difficulties,
            keep_in_memory=keep_in_memory,
            system_prompt=system_prompt,
            exact_weight=exact_weight,
            similarity_weight=similarity_weight,
            config=config.taskset,
        ),
        harness=vf.Harness(config=config.harness),
    )
