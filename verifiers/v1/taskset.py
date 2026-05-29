from importlib.abc import Traversable
from pathlib import Path
from typing import Generic, TypeAlias, TypeVar, cast, final

from datasets import Dataset

from .config import (
    ConfigSource,
    LifecycleConfig,
    resolve_config_object,
)
from .state import State
from .task import Task
from .user import UserConfig
from .utils.binding_utils import (
    BindingMap,
    normalize_binding_map,
    normalize_object_map,
)
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
    dataset_rows_from_tasks,
    discover_sibling_dir,
    task_data_from_result,
    task_from_row,
)
from .types import (
    PromptInput,
    TaskLoader,
    TaskSplit,
    TaskRow,
    Tasks,
    Objects,
    SystemPrompt,
)

TaskLoaderRef: TypeAlias = str


class TasksetConfig(LifecycleConfig):
    # Core fields configure taskset-owned loaders and runtime behavior.
    tasks: TaskLoaderRef | None = None
    taskset_id: str | None = None
    system_prompt: PromptInput | None = None
    user: UserConfig | None = None
    bindings: BindingMap = {}
    objects: dict[str, str] = {}


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

    @final
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
            system_prompt_value = (
                self.config.system_prompt
                if "system_prompt" in self.config.model_fields_set
                else self.load_system_prompt()
            )
            self.system_prompt = normalize_system_prompt(
                system_prompt_value,
                field_name="taskset.system_prompt",
            )
            self.initialize_runtime_user(
                self.config.user,
                explicitly_configured="user" in self.config.model_fields_set,
            )
            self.bindings = normalize_binding_map(
                self.config.bindings, "taskset.bindings"
            )
            self.objects = cast(
                Objects,
                {
                    str(key): resolve_config_object(item)
                    for key, item in normalize_object_map(
                        self.config.objects, "taskset.objects"
                    ).items()
                },
            )
            self.initialize_runtime_toolsets(self.config.toolsets)
            self.initialize_runtime_handlers()
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None

    def get_skills_dir(self) -> Traversable | Path | None:
        return discover_sibling_dir(type(self), "skills")

    def get_upload_dirs(self) -> dict[str, Traversable | Path]:
        skills = self.get_skills_dir()
        return {} if skills is None else {"skills": skills}

    def load_user(self) -> UserConfig | None:
        return self.config.user

    def to_task(self, row: TaskRow | Task) -> Task:
        return task_from_row(row, self.taskset_id)

    def load_tasks(self, split: TaskSplit = "train") -> Tasks:
        if self.config.tasks is None:
            return []
        loader = resolve_config_object(self.config.tasks)
        if not callable(loader):
            raise TypeError("TasksetConfig.tasks must resolve to a callable.")
        return cast(TaskLoader, loader)(split=split)

    async def init_group(
        self, task: Task, num_rollouts: int
    ) -> tuple[list[Task], list[State]]:
        tasks = [task for _ in range(num_rollouts)]
        return tasks, [State.for_task(task) for task in tasks]

    def get_dataset(self) -> Dataset:
        if self._dataset is None:
            with config_ref_context(self.config):
                tasks = task_data_from_result(self.load_tasks(split="train"))
            self._dataset = Dataset.from_list(
                dataset_rows_from_tasks(tasks, self.taskset_id)
            )
        return self._dataset

    def get_eval_dataset(self) -> Dataset:
        if self._eval_dataset is None:
            with config_ref_context(self.config):
                tasks = task_data_from_result(self.load_tasks(split="eval"))
            self._eval_dataset = Dataset.from_list(
                dataset_rows_from_tasks(tasks, self.taskset_id)
            )
        return self._eval_dataset

    def __iter__(self):
        for row in self.get_dataset():
            yield self.to_task(row)

    def __len__(self) -> int:
        return len(self.get_dataset())

    def load_system_prompt(self) -> SystemPrompt | None:
        return self.config.system_prompt
