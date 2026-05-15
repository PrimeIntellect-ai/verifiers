import json
import uuid
import weakref
from importlib.abc import Traversable
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from datasets import Dataset
from verifiers.types import task_payload_from_info

from .config import (
    TasksetConfig,
    merge_config_handler_map,
    resolve_config_object,
)
from .state import State
from .task import Task
from .toolset import merge_toolsets, normalize_toolset_collection
from .user import normalize_user
from .utils.prompt_utils import normalize_system_prompt
from .utils.taskset_utils import dataset_info_with_task, discover_sibling_dir
from .utils.taskset_utils import rows_from_source
from .types import (
    ConfigData,
    ConfigMap,
    Handler,
    TaskRow,
    TaskRowsSource,
)

if TYPE_CHECKING:
    from .harness import Harness


TaskSourceValue = TaskRowsSource | None


class Taskset:
    config_type: ClassVar[type[TasksetConfig]] = TasksetConfig

    def __init__(self, config: TasksetConfig = TasksetConfig()):
        self.config = type(self).config_type.from_config(config)
        source_value = resolve_config_object(self.config.source)
        self.source = cast(
            TaskSourceValue,
            source_value,
        )
        eval_source_value = resolve_config_object(self.config.eval_source)
        self.eval_source = cast(
            TaskSourceValue,
            eval_source_value,
        )
        resolved_taskset_id = self.config.taskset_id
        if resolved_taskset_id is not None and not isinstance(resolved_taskset_id, str):
            raise TypeError("taskset_id must be a string.")
        self.taskset_id = resolved_taskset_id or type(self).__name__
        self.system_prompt = normalize_system_prompt(
            self.config.system_prompt, field_name="taskset.system_prompt"
        )
        self.user = normalize_user(self.config.user)
        self.bindings = dict(self.config.bindings)
        self.objects = {
            **{
                str(key): resolve_config_object(item)
                for key, item in self.config.objects.items()
            }
        }
        self.toolsets, self.named_toolsets = merge_toolsets((), self.config.toolsets)
        handlers = merge_config_handler_map(
            {
                "stop": (),
                "setup": (),
                "update": (),
                "metric": (),
                "reward": (),
                "advantage": (),
                "cleanup": (),
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
        self._rows: list[ConfigData] | None = None
        self._eval_rows: list[ConfigData] | None = None
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None
        self._attached_harnesses: weakref.WeakSet["Harness"] = weakref.WeakSet()

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

    def attach_harness(self, harness: "Harness") -> None:
        self._attached_harnesses.add(harness)

    def get_skills_dir(self) -> Traversable | Path | None:
        return discover_sibling_dir(type(self), "skills")

    def get_upload_dirs(self) -> dict[str, Traversable | Path]:
        skills = self.get_skills_dir()
        return {} if skills is None else {"skills": skills}

    def _refresh_attached_harnesses(self) -> None:
        for harness in list(self._attached_harnesses):
            harness.runtime = harness.resolve_runtime()

    def rows(self) -> list[ConfigData]:
        if self._rows is None:
            self._rows = rows_from_source(self.source, self.config)
        return self._rows

    def eval_rows(self) -> list[ConfigData]:
        if self.eval_source is None:
            return self.rows()
        if self._eval_rows is None:
            self._eval_rows = rows_from_source(self.eval_source, self.config)
        return self._eval_rows

    def task(self, row: ConfigMap) -> Task:
        task = Task(row)
        task["taskset_id"] = self.taskset_id
        task_id = task.get("task_id")
        if task_id is None:
            task_id = task.get("id")
        if task_id is None:
            task_id = task.get("example_id")
        task["task_id"] = str(task_id if task_id is not None else uuid.uuid4().hex)
        return task.freeze()

    def to_task(self, value: ConfigMap | Task | str) -> Task:
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

    def _dataset_row(self, row: TaskRow, index: int) -> ConfigData:
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
        dataset_row: ConfigData = {
            "prompt": task_payload["prompt"],
            "example_id": normalized["example_id"],
            "info": dataset_info_with_task(task_payload),
        }
        if "answer" in normalized:
            dataset_row["answer"] = normalized["answer"]
        return dataset_row
