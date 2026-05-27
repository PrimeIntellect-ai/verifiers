import json
import uuid
import weakref
from importlib.abc import Traversable
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar, cast

from datasets import Dataset
from pydantic import field_validator, model_validator
from verifiers.types import task_payload_from_info

from .config import (
    ConfigSource,
    LifecycleConfig,
    resolve_config_object,
)
from .state import State
from .task import Task
from .user import UserConfig
from .utils.binding_utils import BindingMap, normalize_binding_map
from .utils.prompt_utils import normalize_system_prompt
from .utils.config_utils import (
    coerce_config,
    config_ref_context,
    config_type_from_class,
    registered_config_type,
    register_config_type,
)
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
    PromptInput,
    TaskLoader,
    TaskRow,
)

if TYPE_CHECKING:
    from .harness import Harness


TaskLoaderRef: TypeAlias = str


class TasksetConfig(LifecycleConfig):
    # Core fields configure taskset-owned loaders and runtime behavior.
    tasks: TaskLoaderRef | None = None
    eval_tasks: TaskLoaderRef | None = None
    taskset_id: str | None = None
    system_prompt: PromptInput | None = None
    user: UserConfig | str | None = None
    bindings: BindingMap = {}
    objects: dict[str, str] = {}

    @model_validator(mode="before")
    @classmethod
    def normalize_task_fields(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "train_tasks" not in data:
            return data
        if "tasks" in data:
            raise ValueError(
                "TasksetConfig received multiple values for 'tasks': "
                "'tasks', 'train_tasks'; provide only one."
            )
        train_tasks = data.pop("train_tasks")
        if train_tasks is not None and not isinstance(train_tasks, str):
            raise ValueError(
                "TasksetConfig.train_tasks must be an import ref string. "
                "Inline task rows are not supported; define load_train_tasks() "
                "or load_tasks() and set TasksetConfig.train_tasks or "
                "TasksetConfig.tasks to its import ref."
            )
        data["tasks"] = train_tasks
        return data

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> BindingMap:
        return normalize_binding_map(value, "taskset.bindings")


ConfigT = TypeVar("ConfigT", bound=TasksetConfig)


class Taskset(RuntimeOwnerMixin, Generic[ConfigT]):
    config: ConfigT

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        config_type = config_type_from_class(
            cls,
            inherited=False,
            owner_base=Taskset,
            config_base=TasksetConfig,
        )
        if config_type is not None:
            register_config_type(cls, config_type)

    def __init__(self, config: ConfigSource = None):
        config_type = registered_config_type(type(self), TasksetConfig)
        self.config = cast(ConfigT, coerce_config(config_type, config))
        with config_ref_context(self.config):
            resolved_taskset_id = self.config.taskset_id
            if resolved_taskset_id is not None and not isinstance(
                resolved_taskset_id, str
            ):
                raise TypeError("taskset_id must be a string.")
            self.taskset_id = resolved_taskset_id or type(self).__name__
            self.system_prompt = normalize_system_prompt(
                self._system_prompt_input(),
                field_name="taskset.system_prompt",
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

    @classmethod
    def config_schema(cls) -> str:
        return registered_config_type(cls, TasksetConfig).schema_text()

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
                load_tasks = self._task_loader(
                    "tasks", ("load_tasks", "load_train_tasks")
                )
                tasks = task_data_from_loader(load_tasks)
            self._dataset = Dataset.from_list(
                [self._dataset_row(row, index) for index, row in enumerate(tasks)]
            )
        return self._dataset

    def get_eval_dataset(self) -> Dataset:
        with config_ref_context(self.config):
            load_tasks = self._task_loader("eval_tasks", ("load_eval_tasks",))
        if load_tasks is None:
            return self.get_dataset()
        if self._eval_dataset is None:
            with config_ref_context(self.config):
                tasks = task_data_from_loader(load_tasks)
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

    def _task_loader(
        self, field: str, method_names: tuple[str, ...]
    ) -> TaskLoader | None:
        if field not in self.config.model_fields_set:
            methods = [
                getattr(self, method_name)
                for method_name in method_names
                if callable(getattr(self, method_name, None))
            ]
            if len(methods) > 1:
                raise ValueError(
                    f"Taskset defines multiple class loaders for {field}: "
                    f"{', '.join(method_names)}."
                )
            if methods:
                return cast(TaskLoader, methods[0])
        value = getattr(self.config, field)
        return resolve_task_loader(field, cast(str | None, value))

    def _system_prompt_input(self) -> PromptInput | None:
        if "system_prompt" in self.config.model_fields_set:
            return cast(PromptInput | None, self.config.system_prompt)
        method = getattr(self, "load_system_prompt", None)
        if callable(method):
            return cast(PromptInput | None, method())
        return cast(PromptInput | None, self.config.system_prompt)
