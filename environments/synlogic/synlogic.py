"""Verifiers environment for the SynLogic reasoning benchmark.

SynLogic examples are single-turn logic/puzzle tasks.  The Hugging Face dataset
contains chat prompts plus ``extra_info.game_data_str`` payloads used by the
upstream SynLogic verifier classes.  This wrapper keeps the upstream verifier
logic local to the environment package and exposes the benchmark through a
standard ``vf.SingleTurnEnv``.
"""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset
import verifiers as vf

DATASET_NAME = "MiniMaxAI/SynLogic"
DEFAULT_CONFIG = "easy"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "validation"
SYNLOGIC_SRC = Path(__file__).with_name("synlogic_src")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

SYSTEM_PROMPT = (
    "Solve each SynLogic puzzle step by step. Put any reasoning inside "
    "<think>...</think> and put only the final answer inside "
    "<answer>...</answer>."
)


def _ensure_synlogic_import_path() -> None:
    src = str(SYNLOGIC_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


@lru_cache(maxsize=1)
def _synlogic_runtime() -> tuple[type[Any], dict[str, type[Any]]]:
    """Import the vendored SynLogic Data class and verifier registry lazily."""
    _ensure_synlogic_import_path()
    from base.data import Data  # type: ignore[import-not-found]
    from task2verifier import verifier_classes  # type: ignore[import-not-found]

    return Data, verifier_classes


def _completion_text(completion: vf.Messages | list[dict[str, Any]] | str) -> str:
    if isinstance(completion, str):
        return completion
    if not completion:
        return ""
    message = completion[-1]
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def extract_answer(response: str) -> str:
    """Extract the final answer, following SynLogic's post-</think> convention."""
    model_output = response.split("</think>", 1)[1] if "</think>" in response else response
    match = ANSWER_RE.search(model_output)
    return match.group(1).strip() if match else ""


def response_has_required_format(response: str) -> bool:
    return (
        response.startswith("<think>")
        and response.endswith("</answer>")
        and response.count("<think>") == 1
        and response.count("</think>") == 1
        and response.count("<answer>") == 1
        and response.count("</answer>") == 1
    )


def normalize_data_source(data_source: str) -> str:
    """Normalize validation sources such as ``val/campsite`` to ``campsite``."""
    return data_source.split("/", 1)[1] if data_source.startswith("val/") else data_source


def _info_from_row(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = row.get("extra_info") or {}
    data_source = normalize_data_source(str(row.get("data_source", "")))
    return {
        "data_source": data_source,
        "ability": row.get("ability"),
        "game_data_str": extra_info.get("game_data_str", ""),
        "source_file": extra_info.get("source_file"),
        "index": extra_info.get("index"),
        "metadata": extra_info.get("metadata"),
        "original_answer": extra_info.get("original_answer"),
        "original_question": extra_info.get("original_question"),
    }


def _row_to_example(row: dict[str, Any]) -> dict[str, Any]:
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        prompt = [{"role": "user", "content": str(prompt or row.get("question", ""))}]
    return {
        "prompt": prompt,
        "answer": str((row.get("extra_info") or {}).get("original_answer", "")),
        "info": json.dumps(_info_from_row(row), ensure_ascii=False),
    }


def _load_synlogic_dataset(
    *,
    config: str,
    split: str,
    max_examples: int | None,
    shuffle: bool,
    seed: int,
) -> Dataset:
    ds = load_dataset(DATASET_NAME, config, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if max_examples is not None and max_examples >= 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    rows = [_row_to_example(cast(dict[str, Any], row)) for row in ds]
    return Dataset.from_list(rows)


def make_dataset_builder(
    *,
    config: str,
    split: str,
    max_examples: int | None,
    shuffle: bool,
    seed: int,
) -> vf.DatasetBuilder:
    def build() -> Dataset:
        return _load_synlogic_dataset(
            config=config,
            split=split,
            max_examples=max_examples,
            shuffle=shuffle,
            seed=seed,
        )

    return build


def synlogic_accuracy(response: str, info: dict[str, Any]) -> float:
    """Run the upstream verifier for one SynLogic response."""
    Data, verifier_classes = _synlogic_runtime()
    data_source = normalize_data_source(str(info.get("data_source", "")))
    game_data_str = str(info.get("game_data_str", ""))
    verifier_cls = verifier_classes.get(data_source)
    if verifier_cls is None or not game_data_str:
        return 0.0
    try:
        game_data = Data.from_json_str(game_data_str)
        verifier = verifier_cls()
        score = verifier.verify(game_data, extract_answer(response))
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return 0.0


async def synlogic_reward(completion: vf.Messages, info: dict[str, Any]) -> float:
    response = _completion_text(completion)
    if not response_has_required_format(response):
        return 0.0
    return synlogic_accuracy(response, info)


async def format_reward(completion: vf.Messages) -> float:
    return 1.0 if response_has_required_format(_completion_text(completion)) else 0.0


async def accuracy_reward(completion: vf.Messages, info: dict[str, Any]) -> float:
    return synlogic_accuracy(_completion_text(completion), info)


def load_environment(
    config: str = DEFAULT_CONFIG,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    eval_split: str = DEFAULT_EVAL_SPLIT,
    max_examples: int | None = 512,
    max_eval_examples: int | None = 128,
    shuffle: bool = True,
    seed: int = 0,
    system_prompt: str = SYSTEM_PROMPT,
) -> vf.Environment:
    """Load the SynLogic environment.

    Args:
        config: Hugging Face dataset config, usually ``easy`` or ``hard``.
        train_split: Dataset split used for training.
        eval_split: Dataset split used for evaluation.
        max_examples: Optional cap for training rows; ``None`` loads all rows.
        max_eval_examples: Optional cap for eval rows; ``None`` loads all rows.
        shuffle: Whether to shuffle before applying caps.
        seed: Shuffle seed.
        system_prompt: System prompt prepended by ``SingleTurnEnv`` when needed.
    """
    if config not in {"easy", "hard"}:
        raise ValueError("config must be 'easy' or 'hard'")

    rubric = vf.Rubric(
        funcs=[synlogic_reward, format_reward, accuracy_reward],
        weights=[1.0, 0.0, 0.0],
    )
    return vf.SingleTurnEnv(
        dataset=make_dataset_builder(
            config=config,
            split=train_split,
            max_examples=max_examples,
            shuffle=shuffle,
            seed=seed,
        ),
        eval_dataset=make_dataset_builder(
            config=config,
            split=eval_split,
            max_examples=max_eval_examples,
            shuffle=shuffle,
            seed=seed + 1,
        ),
        system_prompt=system_prompt,
        rubric=rubric,
    )
