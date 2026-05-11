from __future__ import annotations

import json
import uuid
import weakref
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import Any, ClassVar, cast

from datasets import Dataset
from verifiers.types import task_payload_from_info

from .config import (
    CallableKind,
    TasksetConfig,
    merge_config_callables,
    merge_config_value,
    resolve_config_object,
)
from .state import State
from .task import Task
from .toolset import merge_toolsets, normalize_toolset_collection
from .user import normalize_user
from .utils.prompt_utils import normalize_system_prompt

TasksetSource = Iterable[Mapping[str, Any]] | Callable[[], Iterable[Mapping[str, Any]]]
TASKSET_HANDLER_FIELDS: tuple[tuple[str, CallableKind], ...] = (
    ("stops", "stop"),
    ("setups", "setup"),
    ("updates", "update"),
    ("metrics", "metric"),
    ("rewards", "reward"),
    ("advantages", "advantage"),
    ("cleanups", "cleanup"),
)


class Taskset:
    config_type: ClassVar[type[TasksetConfig]] = TasksetConfig

    def __init__(
        self,
        # Singleton fields.
        source: TasksetSource | None = None,
        eval_source: TasksetSource | None = None,
        taskset_id: str | None = None,
        system_prompt: object | None = None,
        user: object | None = None,
        # Collection fields.
        toolsets: Iterable[object] = (),
        stops: Iterable[Callable[..., object]] = (),
        setups: Iterable[Callable[..., object]] = (),
        updates: Iterable[Callable[..., object]] = (),
        metrics: Iterable[Callable[..., object]] = (),
        rewards: Iterable[Callable[..., object]] = (),
        advantages: Iterable[Callable[..., object]] = (),
        cleanups: Iterable[Callable[..., object]] = (),
        # Config.
        config: TasksetConfig | Mapping[str, object] | None = None,
    ):
        self.config = type(self).config_type.from_config(config)
        config = self.config
        source = source if source is not None else config.source
        eval_source = eval_source if eval_source is not None else config.eval_source
        self.source = cast(
            TasksetSource | None,
            resolve_config_object(source),
        )
        self.eval_source = cast(
            TasksetSource | None,
            resolve_config_object(eval_source),
        )
        resolved_taskset_id = (
            taskset_id if taskset_id is not None else config.taskset_id
        )
        if resolved_taskset_id is not None and not isinstance(resolved_taskset_id, str):
            raise TypeError("taskset_id must be a string.")
        self.taskset_id = resolved_taskset_id or type(self).__name__
        self.system_prompt = normalize_system_prompt(
            system_prompt if system_prompt is not None else config.system_prompt,
            field_name="taskset.system_prompt",
        )
        self.user = normalize_user(merge_config_value(user, config.user))
        self.toolsets, self.named_toolsets = merge_toolsets(toolsets, config.toolsets)
        handler_values = {
            "stops": stops,
            "setups": setups,
            "updates": updates,
            "metrics": metrics,
            "rewards": rewards,
            "advantages": advantages,
            "cleanups": cleanups,
        }
        for field, kind in TASKSET_HANDLER_FIELDS:
            setattr(
                self,
                field,
                cast(
                    list[Callable[..., object]],
                    merge_config_callables(
                        handler_values[field], getattr(config, field), kind
                    ),
                ),
            )
        self._rows: list[dict[str, Any]] | None = None
        self._eval_rows: list[dict[str, Any]] | None = None
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None
        self._attached_harnesses: weakref.WeakSet[object] = weakref.WeakSet()

    @classmethod
    def config_schema(cls) -> str:
        return cls.config_type.schema_text()

    def _add_handler(
        self, handlers: list[Callable[..., object]], fn: Callable[..., object]
    ) -> None:
        handlers.append(fn)
        self._refresh_attached_harnesses()

    def add_metric(self, fn: Callable[..., object]) -> None:
        self._add_handler(self.metrics, fn)

    def add_reward(self, fn: Callable[..., object]) -> None:
        self._add_handler(self.rewards, fn)

    def add_advantage(self, fn: Callable[..., object]) -> None:
        self._add_handler(self.advantages, fn)

    def add_toolset(self, toolset: object) -> None:
        toolsets, named_toolsets = normalize_toolset_collection(toolset)
        duplicate = set(self.named_toolsets) & set(named_toolsets)
        if duplicate:
            raise ValueError(f"Toolsets are defined twice: {sorted(duplicate)}.")
        self.toolsets.extend(toolsets)
        self.named_toolsets.update(named_toolsets)
        self._refresh_attached_harnesses()

    def add_stop(self, fn: Callable[..., object]) -> None:
        self._add_handler(self.stops, fn)

    def add_setup(self, fn: Callable[..., object]) -> None:
        self._add_handler(self.setups, fn)

    def add_update(self, fn: Callable[..., object]) -> None:
        self._add_handler(self.updates, fn)

    def add_cleanup(self, fn: Callable[..., object]) -> None:
        self._add_handler(self.cleanups, fn)

    def attach_harness(self, harness: object) -> None:
        self._attached_harnesses.add(harness)

    def _refresh_attached_harnesses(self) -> None:
        for harness in list(self._attached_harnesses):
            resolve_runtime = getattr(harness, "resolve_runtime", None)
            if callable(resolve_runtime):
                setattr(harness, "runtime", resolve_runtime())

    def rows(self) -> list[dict[str, Any]]:
        if self._rows is None:
            self._rows = rows_from_source(self.source)
        return self._rows

    def eval_rows(self) -> list[dict[str, Any]]:
        if self.eval_source is None:
            return self.rows()
        if self._eval_rows is None:
            self._eval_rows = rows_from_source(self.eval_source)
        return self._eval_rows

    def task(self, row: Mapping[str, Any]) -> Task:
        task = Task(row)
        task["taskset_id"] = self.taskset_id
        task_id = task.get("task_id")
        if task_id is None:
            task_id = task.get("id")
        if task_id is None:
            task_id = task.get("example_id")
        task["task_id"] = str(task_id if task_id is not None else uuid.uuid4().hex)
        return task.freeze()

    def to_task(self, value: Mapping[str, Any] | Task | str) -> Task:
        if isinstance(value, Task):
            return value
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, Mapping):
            raise TypeError("Taskset.to_task expects a mapping, Task, or JSON string.")
        serialized_task = task_payload_from_info(value.get("info"))
        if serialized_task is not None:
            return self.task(serialized_task)
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

    def get_eval_dataset(self) -> Dataset:
        if self.eval_source is None:
            return self.get_dataset()
        if self._eval_dataset is None:
            self._eval_dataset = Dataset.from_list(
                [
                    self._dataset_row(row, index)
                    for index, row in enumerate(self.eval_rows())
                ]
            )
        return self._eval_dataset

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
        task_payload = dict(self.task(normalized))
        dataset_row: dict[str, Any] = {
            "prompt": task_payload["prompt"],
            "example_id": normalized["example_id"],
            "info": dataset_info_with_task(task_payload),
        }
        if "answer" in normalized:
            dataset_row["answer"] = normalized["answer"]
        return dataset_row


def dataset_info_with_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return {"task": json.dumps(task)}


def rows_from_source(
    source: TasksetSource | None,
) -> list[dict[str, Any]]:
    if source is None:
        return []
    if callable(source):
        source_loader = cast(Callable[[], Iterable[Mapping[str, Any]]], source)
        return [dict(row) for row in source_loader()]
    return [dict(row) for row in source]
