from __future__ import annotations

import json
import uuid
import weakref
from importlib.abc import Traversable
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import ClassVar, cast

from datasets import Dataset
from verifiers.types import task_payload_from_info
from typing_extensions import NotRequired, TypedDict

from .config import (
    TasksetConfig,
    merge_config_handler_map,
    merge_config_value,
    resolve_config_object,
)
from .types import ConfigMap, Handler, PromptInput, TaskRow, TaskRowsSource
from .state import State
from .task import Task
from .toolset import merge_toolsets, normalize_toolset_collection
from .user import normalize_user
from .utils.prompt_utils import normalize_system_prompt
from .utils.taskset_utils import dataset_info_with_task, discover_sibling_dir
from .utils.taskset_utils import rows_from_source


TaskSourceValue = TaskRowsSource | None


class TasksetKwargs(TypedDict):
    eval_source: NotRequired[TaskSourceValue]
    taskset_id: NotRequired[str | None]
    system_prompt: NotRequired[PromptInput | None]
    user: NotRequired[Handler | str | ConfigMap | None]
    toolsets: NotRequired[Iterable[object]]
    stops: NotRequired[Iterable[Handler]]
    setups: NotRequired[Iterable[Handler]]
    updates: NotRequired[Iterable[Handler]]
    metrics: NotRequired[Iterable[Handler]]
    rewards: NotRequired[Iterable[Handler]]
    advantages: NotRequired[Iterable[Handler]]
    cleanups: NotRequired[Iterable[Handler]]


class Taskset:
    config_type: ClassVar[type[TasksetConfig]] = TasksetConfig

    def __init__(
        self,
        # Singleton fields.
        source: TaskSourceValue = None,
        eval_source: TaskSourceValue = None,
        taskset_id: str | None = None,
        system_prompt: PromptInput | None = None,
        user: Handler | str | ConfigMap | None = None,
        # Collection fields.
        toolsets: Iterable[object] = (),
        stops: Iterable[Handler] = (),
        setups: Iterable[Handler] = (),
        updates: Iterable[Handler] = (),
        metrics: Iterable[Handler] = (),
        rewards: Iterable[Handler] = (),
        advantages: Iterable[Handler] = (),
        cleanups: Iterable[Handler] = (),
        # Config.
        config: TasksetConfig | ConfigMap | None = None,
    ):
        self.config = type(self).config_type.from_config(config)
        source_value = resolve_config_object(
            merge_config_value(source, self.config.source)
        )
        self.source = cast(
            TaskSourceValue,
            source_value,
        )
        eval_source_value = resolve_config_object(
            merge_config_value(eval_source, self.config.eval_source)
        )
        self.eval_source = cast(
            TaskSourceValue,
            eval_source_value,
        )
        resolved_taskset_id = merge_config_value(taskset_id, self.config.taskset_id)
        if resolved_taskset_id is not None and not isinstance(resolved_taskset_id, str):
            raise TypeError("taskset_id must be a string.")
        self.taskset_id = resolved_taskset_id or type(self).__name__
        system_prompt_value = cast(
            PromptInput | None,
            merge_config_value(system_prompt, self.config.system_prompt),
        )
        self.system_prompt = normalize_system_prompt(
            system_prompt_value, field_name="taskset.system_prompt"
        )
        self.user = normalize_user(merge_config_value(user, self.config.user))
        self.toolsets, self.named_toolsets = merge_toolsets(
            toolsets, self.config.toolsets
        )
        handlers = merge_config_handler_map(
            {
                "stop": stops,
                "setup": setups,
                "update": updates,
                "metric": metrics,
                "reward": rewards,
                "advantage": advantages,
                "cleanup": cleanups,
            },
            self.config,
        )
        self.stops = handlers["stop"]
        self.setups = handlers["setup"]
        self.updates = handlers["update"]
        self.metrics = handlers["metric"]
        self.rewards = handlers["reward"]
        self.advantages = handlers["advantage"]
        self.cleanups = handlers["cleanup"]
        self._rows: list[dict[str, object]] | None = None
        self._eval_rows: list[dict[str, object]] | None = None
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None
        self._attached_harnesses: weakref.WeakSet[object] = weakref.WeakSet()

    @classmethod
    def config_schema(cls) -> str:
        return cls.config_type.schema_text()

    def _add_handler(self, handlers: list[Handler], fn: Handler) -> None:
        handlers.append(fn)
        self._refresh_attached_harnesses()

    def add_metric(self, fn: Handler) -> None:
        self._add_handler(self.metrics, fn)

    def add_reward(self, fn: Handler) -> None:
        self._add_handler(self.rewards, fn)

    def add_advantage(self, fn: Handler) -> None:
        self._add_handler(self.advantages, fn)

    def add_toolset(self, toolset: object) -> None:
        toolsets, named_toolsets = normalize_toolset_collection(toolset)
        duplicate = set(self.named_toolsets) & set(named_toolsets)
        if duplicate:
            raise ValueError(f"Toolsets are defined twice: {sorted(duplicate)}.")
        self.toolsets.extend(toolsets)
        self.named_toolsets.update(named_toolsets)
        self._refresh_attached_harnesses()

    def add_stop(self, fn: Handler) -> None:
        self._add_handler(self.stops, fn)

    def add_setup(self, fn: Handler) -> None:
        self._add_handler(self.setups, fn)

    def add_update(self, fn: Handler) -> None:
        self._add_handler(self.updates, fn)

    def add_cleanup(self, fn: Handler) -> None:
        self._add_handler(self.cleanups, fn)

    def attach_harness(self, harness: object) -> None:
        self._attached_harnesses.add(harness)

    def get_skills_dir(self) -> Traversable | Path | None:
        return discover_sibling_dir(type(self), "skills")

    def get_upload_dirs(self) -> dict[str, Traversable | Path]:
        skills = self.get_skills_dir()
        return {} if skills is None else {"skills": skills}

    def _refresh_attached_harnesses(self) -> None:
        for harness in list(self._attached_harnesses):
            resolve_runtime = getattr(harness, "resolve_runtime", None)
            if callable(resolve_runtime):
                setattr(harness, "runtime", resolve_runtime())

    def rows(self) -> list[dict[str, object]]:
        if self._rows is None:
            self._rows = rows_from_source(self.source)
        return self._rows

    def eval_rows(self) -> list[dict[str, object]]:
        if self.eval_source is None:
            return self.rows()
        if self._eval_rows is None:
            self._eval_rows = rows_from_source(self.eval_source)
        return self._eval_rows

    def task(self, row: Mapping[str, object]) -> Task:
        task = Task(row)
        task["taskset_id"] = self.taskset_id
        task_id = task.get("task_id")
        if task_id is None:
            task_id = task.get("id")
        if task_id is None:
            task_id = task.get("example_id")
        task["task_id"] = str(task_id if task_id is not None else uuid.uuid4().hex)
        return task.freeze()

    def to_task(self, value: Mapping[str, object] | Task | str) -> Task:
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

    def _dataset_row(self, row: TaskRow, index: int) -> dict[str, object]:
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
        dataset_row: dict[str, object] = {
            "prompt": task_payload["prompt"],
            "example_id": normalized["example_id"],
            "info": dataset_info_with_task(task_payload),
        }
        if "answer" in normalized:
            dataset_row["answer"] = normalized["answer"]
        return dataset_row
