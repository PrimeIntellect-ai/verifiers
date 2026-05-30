import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset

import verifiers as vf

DATA_DIR = Path(__file__).with_name("data")
SYSTEM_PROMPT = (
    "You are evaluating thematic-generalization candidates. "
    "Score every candidate from 0 to 10 and use exactly the requested XML tags."
)
_SCORE_RE = re.compile(
    r"<number>\s*([1-8])\s*</number>\s*<score>\s*(10|[0-9])\s*</score>",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_dataset(split: str, max_examples: int | None = None) -> Dataset:
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    rows = _read_jsonl(DATA_DIR / f"{split}.jsonl")
    if max_examples is not None:
        rows = rows[:max_examples]
    return Dataset.from_list(rows)


def _completion_text(completion: list[dict[str, Any]] | str) -> str:
    if isinstance(completion, str):
        return completion
    if not completion:
        return ""
    content = completion[-1].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _parse_scores(text: str) -> dict[int, int]:
    scores: dict[int, int] = {}
    for candidate, score in _SCORE_RE.findall(text):
        scores[int(candidate)] = int(score)
    return scores


def _average_rank(scores: dict[int, int], answer: str | int) -> float | None:
    answer_idx = int(answer)
    if answer_idx not in scores:
        return None
    answer_score = scores[answer_idx]
    # Higher score is better. Use average rank for ties, matching ranking-eval practice.
    better = sum(1 for score in scores.values() if score > answer_score)
    tied = sum(1 for score in scores.values() if score == answer_score)
    return better + (tied + 1) / 2


def reciprocal_rank_reward(completion, answer, **kwargs) -> float:
    scores = _parse_scores(_completion_text(completion))
    rank = _average_rank(scores, answer)
    if rank is None:
        return 0.0
    return 1.0 / rank


def top_1_reward(completion, answer, **kwargs) -> float:
    scores = _parse_scores(_completion_text(completion))
    rank = _average_rank(scores, answer)
    if rank is None:
        return 0.0
    return 1.0 if rank == 1.0 else 0.0


def format_reward(completion, **kwargs) -> float:
    scores = _parse_scores(_completion_text(completion))
    return len(scores) / 8.0


def load_environment(
    split: str = "train",
    eval_split: str = "eval",
    max_examples: int | None = None,
    max_eval_examples: int | None = None,
    system_prompt: str | None = SYSTEM_PROMPT,
) -> vf.Environment:
    """Load the LLM Thematic Generalization benchmark environment.

    The dataset is derived from lechmazur/generalization's V1 pick prompts. Each
    prompt contains three positive examples, three anti-examples, and eight
    candidates. The hidden true candidate is used only as the environment answer;
    the marker is removed from the model-visible prompt.
    """

    def dataset_builder() -> Dataset:
        return _build_dataset(split=split, max_examples=max_examples)

    def eval_dataset_builder() -> Dataset:
        return _build_dataset(split=eval_split, max_examples=max_eval_examples)

    rubric = vf.Rubric(
        funcs=[reciprocal_rank_reward, top_1_reward, format_reward],
        weights=[1.0, 0.0, 0.0],
    )

    return vf.SingleTurnEnv(
        dataset=dataset_builder,
        eval_dataset=eval_dataset_builder,
        system_prompt=system_prompt,
        rubric=rubric,
    )
