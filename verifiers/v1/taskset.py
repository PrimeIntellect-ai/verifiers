from importlib.abc import Traversable
from pathlib import Path
from typing import Generic, TypeVar, cast, final

from datasets import Dataset
from pydantic import AliasChoices, Field

from .config import (
    ConfigSource,
    LifecycleConfig,
)
from .artifact import ArtifactsConfig
from .state import State
from .task import Task
from .user import UserConfig
from .utils.binding_utils import (
    BindingSources,
    BindingsConfig,
    ObjectsConfig,
)
from .utils.prompt_utils import SystemPrompt, normalize_system_prompt
from .utils.config_utils import (
    coerce_config,
    config_ref_context,
    config_type_from_class,
    registered_config_type,
    register_config_type,
)
from .utils.runtime_owner_utils import RuntimeOwnerMixin
from .utils.taskset_utils import (
    dataset_from_result,
    discover_sibling_dir,
    prepare_task,
    task_from_dataset_record,
)
from .types import (
    JsonData,
    Objects,
    TaskSplit,
    Tasks,
)


class TasksetConfig(LifecycleConfig):
    # Core fields configure taskset-owned loaders and runtime behavior.
    taskset_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("taskset_id", "id"),
    )
    system_prompt: SystemPrompt = None
    user: UserConfig | None = None
    bindings: BindingsConfig = BindingsConfig()
    objects: ObjectsConfig = ObjectsConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        field = cls.model_fields.get("taskset_id")
        if field is not None:
            field.validation_alias = AliasChoices("taskset_id", "id")
            cls.model_rebuild(force=True)


ConfigT = TypeVar("ConfigT", bound=TasksetConfig)


class Taskset(RuntimeOwnerMixin[ConfigT], Generic[ConfigT]):
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
            self.initialize_runtime_refresh()
            resolved_taskset_id = self.config.taskset_id
            if resolved_taskset_id is not None and not isinstance(
                resolved_taskset_id, str
            ):
                raise TypeError("taskset_id must be a string.")
            self.taskset_id = resolved_taskset_id or type(self).__name__
            system_prompt_value = self.load_system_prompt(self.config)
            self.system_prompt = normalize_system_prompt(
                system_prompt_value,
                field_name="taskset.system_prompt",
            )
            self.initialize_runtime_user(self.config.user)
            self.bindings: BindingSources = self.config.bindings.entries(
                "taskset.bindings"
            )
            self.objects: Objects = self.load_objects(self.config.objects)
            self.artifacts = self.load_artifacts(self.config.artifacts)
            self.initialize_runtime_toolsets(self.config, self.config.toolsets)
            self.initialize_runtime_handlers()
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None

    def get_skills_dir(self) -> Traversable | Path | None:
        return discover_sibling_dir(type(self), "skills")

    def get_upload_dirs(self) -> dict[str, Traversable | Path]:
        skills = self.get_skills_dir()
        return {} if skills is None else {"skills": skills}

    def to_task(self, task: Task | JsonData) -> Task:
        if isinstance(task, Task):
            return prepare_task(task, self.taskset_id)
        return task_from_dataset_record(task, self.taskset_id)

    def load_tasks(self, split: TaskSplit = "train") -> Tasks:
        return []

    async def init_group(
        self, task: Task, num_rollouts: int
    ) -> tuple[list[Task], list[State]]:
        tasks = [task for _ in range(num_rollouts)]
        return tasks, [State.for_task(task) for task in tasks]

    def get_dataset(self) -> Dataset:
        if self._dataset is None:
            with config_ref_context(self.config):
                self._dataset = dataset_from_result(
                    self.load_tasks(split="train"), self.taskset_id
                )
        return self._dataset

    def get_eval_dataset(self) -> Dataset:
        if self._eval_dataset is None:
            with config_ref_context(self.config):
                self._eval_dataset = dataset_from_result(
                    self.load_tasks(split="eval"), self.taskset_id
                )
        return self._eval_dataset

    def __iter__(self):
        for record in self.get_dataset():
            yield task_from_dataset_record(dict(record), self.taskset_id)

    def __len__(self) -> int:
        return len(self.get_dataset())

    def load_system_prompt(self, config: ConfigT) -> SystemPrompt:
        return config.system_prompt
