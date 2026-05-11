import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset

import verifiers as vf

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "eq_bench3_train_sample.jsonl"

SYSTEM_PROMPT = (
    "Predict character emotions carefully from the dialogue. Return only JSON "
    "mapping each requested emotion to a numeric intensity from 0 to 10."
)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> Dataset:
    return Dataset.from_list(_load_jsonl(path))


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


def parse_scores(text: str) -> dict[str, float]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    if not isinstance(parsed, dict):
        return {}
    scores: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            scores[str(key).strip().lower()] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


def emotion_score_reward(completion, answer, **kwargs) -> float:
    expected = parse_scores(answer)
    predicted = parse_scores(_completion_text(completion))
    if not expected or set(expected) - set(predicted):
        return 0.0

    total = 0.0
    for emotion, expected_score in expected.items():
        predicted_score = max(0.0, min(predicted[emotion], 10.0))
        error = abs(predicted_score - expected_score)
        total += max(0.0, 1.0 - (error / 10.0))
    return total / len(expected)


def load_environment(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    system_prompt: str = SYSTEM_PROMPT,
) -> vf.Environment:
    rubric = vf.Rubric(funcs=[emotion_score_reward], weights=[1.0])
    return vf.SingleTurnEnv(
        dataset=lambda: build_dataset(dataset_path),
        system_prompt=system_prompt,
        parser=vf.Parser(),
        rubric=rubric,
    )
