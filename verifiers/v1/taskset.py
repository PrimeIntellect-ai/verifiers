import json
import uuid
import weakref
from importlib.abc import Traversable
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset
from verifiers.types import task_payload_from_info

from .config import (
    ConfigSource,
    TasksetConfig,
    resolve_config_object,
)
from .state import State
from .task import Task
from .utils.prompt_utils import normalize_system_prompt
from .utils.config_utils import coerce_config, config_ref_context
from .utils.runtime_owner_utils import RuntimeOwnerMixin
from .utils.taskset_utils import (
    dataset_info_with_task,
    discover_sibling_dir,
    resolve_task_loader,
    task_data_from_loader,
)
from .types import (
    ConfigData,
    ConfigMap,
    TaskRow,
)

if TYPE_CHECKING:
    from .harness import Harness


class Taskset(RuntimeOwnerMixin):
    config: TasksetConfig

    def __init__(self, config: ConfigSource = None):
        self.config = coerce_config(TasksetConfig, config)
        with config_ref_context(self.config):
            resolved_taskset_id = self.config.taskset_id
            if resolved_taskset_id is not None and not isinstance(
                resolved_taskset_id, str
            ):
                raise TypeError("taskset_id must be a string.")
            self.taskset_id = resolved_taskset_id or type(self).__name__
            self.system_prompt = normalize_system_prompt(
                self.config.system_prompt, field_name="taskset.system_prompt"
            )
            self._init_runtime_user()
            self.bindings = dict(self.config.bindings)
            self.objects = {
                **{
                    str(key): resolve_config_object(item)
                    for key, item in self.config.objects.items()
                }
            }
            self._init_runtime_toolsets()
            self._init_runtime_handlers()
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None
        self._attached_harnesses: weakref.WeakSet["Harness"] = weakref.WeakSet()
        self._configure_runtime_defaults()

    @classmethod
    def config_schema(cls) -> str:
        return TasksetConfig.schema_text()

    def attach_harness(self, harness: "Harness") -> None:
        self._attached_harnesses.add(harness)

    def get_skills_dir(self) -> Traversable | Path | None:
        return discover_sibling_dir(type(self), "skills")

    def get_upload_dirs(self) -> dict[str, Traversable | Path]:
        skills = self.get_skills_dir()
        return {} if skills is None else {"skills": skills}

    def _runtime_owner_changed(self) -> None:
        for harness in list(self._attached_harnesses):
            harness.runtime = harness.resolve_runtime()

    def to_task(self, value: ConfigMap | Task | str) -> Task:
        if isinstance(value, Task):
            return value
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, Mapping):
            raise TypeError("Taskset.to_task expects a mapping, Task, or JSON string.")
        serialized_task = task_payload_from_info(value.get("info"))
        if serialized_task is not None:
            value = serialized_task
        task = Task(value)
        task["taskset_id"] = self.taskset_id
        task_id = task.get("task_id")
        if task_id is None:
            task_id = task.get("id")
        if task_id is None:
            task_id = task.get("example_id")
        task["task_id"] = str(task_id if task_id is not None else uuid.uuid4().hex)
        return task.freeze()

    async def init_group(
        self, task: Task, num_rollouts: int
    ) -> tuple[list[Task], list[State]]:
        tasks = [task for _ in range(num_rollouts)]
        return tasks, [State.for_task(task) for task in tasks]

    def get_dataset(self) -> Dataset:
        if self._dataset is None:
            with config_ref_context(self.config):
                load_tasks = resolve_task_loader("tasks", self.config.tasks)
                tasks = task_data_from_loader(load_tasks, self.config)
            self._dataset = Dataset.from_list(
                [self._dataset_row(row, index) for index, row in enumerate(tasks)]
            )
        return self._dataset

    def get_eval_dataset(self) -> Dataset:
        if self.config.eval_tasks is None:
            return self.get_dataset()
        if self._eval_dataset is None:
            with config_ref_context(self.config):
                load_tasks = resolve_task_loader("eval_tasks", self.config.eval_tasks)
                tasks = task_data_from_loader(load_tasks, self.config)
            self._eval_dataset = Dataset.from_list(
                [self._dataset_row(row, index) for index, row in enumerate(tasks)]
            )
        return self._eval_dataset

    def __iter__(self):
        for row in self.get_dataset():
            yield self.to_task(row)

    def __len__(self) -> int:
        return len(self.get_dataset())

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
        task_payload = dict(self.to_task(normalized))
        dataset_row: ConfigData = {
            "prompt": task_payload["prompt"],
            "example_id": normalized["example_id"],
            "info": dataset_info_with_task(task_payload),
        }
        if "answer" in normalized:
            dataset_row["answer"] = normalized["answer"]
        return dataset_row
