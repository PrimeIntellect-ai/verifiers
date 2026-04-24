from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from datasets import Dataset

from verifiers.envs.experimental.taskset import Taskset
from verifiers.rubrics.rubric import Rubric


class DatasetTaskset(Taskset):
    """Thin named wrapper around a Hugging Face Dataset or row iterable."""

    def __init__(
        self,
        dataset: Dataset | Iterable[Mapping[str, Any]],
        eval_dataset: Dataset | Iterable[Mapping[str, Any]] | None = None,
        rubric: Rubric | None = None,
        name: str | None = None,
    ):
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            name=name,
        )
