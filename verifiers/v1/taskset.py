from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import Any, cast

from datasets import Dataset

from .config import TasksetConfig, merge_config_value
from .state import State
from .task import Task
from .toolset import Toolset


class Taskset:
    def __init__(
        self,
        source: Iterable[Mapping[str, Any]]
        | Callable[[], Iterable[Mapping[str, Any]]]
        | None = None,
        taskset_id: str | None = None,
        metrics: Iterable[Callable[..., object]] = (),
        rewards: Iterable[Callable[..., object]] = (),
        toolsets: Iterable[Toolset] = (),
        cleanup: Iterable[Callable[..., object]] = (),
        config: TasksetConfig | Mapping[str, object] | None = None,
    ):
        self.config = TasksetConfig.model_validate(config or {})
        source_value = merge_config_value(source, self.config.source)
        self.source = cast(
            Iterable[Mapping[str, Any]]
            | Callable[[], Iterable[Mapping[str, Any]]]
            | None,
            source_value,
        )
        self.taskset_id = taskset_id or type(self).__name__
        self.metrics = list(metrics)
        self.rewards = list(rewards)
        self.toolsets = list(toolsets)
        self.cleanup = list(cleanup)
        self._rows: list[dict[str, Any]] | None = None
        self._dataset: Dataset | None = None

    def add_metric(self, fn: Callable[..., object]) -> None:
        self.metrics.append(fn)

    def add_reward(self, fn: Callable[..., object]) -> None:
        self.rewards.append(fn)

    def add_toolset(self, toolset: Toolset) -> None:
        self.toolsets.append(toolset)

    def rows(self) -> list[dict[str, Any]]:
        if self._rows is None:
            if self.source is None:
                self._rows = []
            elif callable(self.source):
                self._rows = [dict(row) for row in self.source()]
            else:
                self._rows = [dict(row) for row in self.source]
        return self._rows

    def task(self, row: Mapping[str, Any]) -> Task:
        task = Task(row)
        task["taskset_id"] = self.taskset_id
        task["task_id"] = str(task.get("task_id") or task.get("id") or uuid.uuid4().hex)
        return task.freeze()

    def to_task(self, value: Mapping[str, Any] | Task) -> Task:
        if isinstance(value, Task):
            return value
        return self.task(value)

    async def init_group(
        self, task: Task, num_rollouts: int
    ) -> tuple[list[Task], list[State]]:
        tasks = [task for _ in range(num_rollouts)]
        return tasks, [State.for_task(task) for task in tasks]

    def get_dataset(self) -> Dataset:
        if self._dataset is None:
            self._dataset = Dataset.from_list(
                [self._dataset_row(row, index) for index, row in enumerate(self.rows())]
            )
        return self._dataset

    def __iter__(self):
        for row in self.rows():
            yield self.task(row)

    def __len__(self) -> int:
        return len(self.rows())

    def _dataset_row(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        normalized = deepcopy(dict(row))
        normalized.setdefault("example_id", index)
        if "prompt" not in normalized:
            question = normalized.get("question")
            normalized["prompt"] = (
                [{"role": "user", "content": str(question)}]
                if question is not None
                else []
            )
        return normalized
