from __future__ import annotations

from collections.abc import Mapping
from importlib.abc import Traversable
from pathlib import Path
from typing import Generic, TypeVar, cast, final

from datasets import Dataset
from pydantic import Field, field_serializer, model_validator

from .config import Config, ConfigSource
from .decorators import discover_decorated
from .state import Extras, State
from .task import Task
from .toolset import ServerConfig, ToolsetConfig, ToolsetConfigs
from .runtime import RuntimeConfig
from .types import Handler, JsonData, TaskSplit, Tasks
from .user import UserConfig
from .utils.config_utils import (
    coerce_config,
    config_ref_context,
    config_type_from_class,
    registered_config_type,
    register_config_type,
)
from .utils.prompt_utils import SystemPrompt, normalize_system_prompt
from .utils.scoring_utils import build_signals
from .utils.taskset_utils import (
    dataset_from_result_typed,
    discover_sibling_dir,
    prepare_task,
    task_from_dataset_record,
)

LifecycleKind = str


class TasksetConfig(Config):
    id: str | None = None
    system_prompt: SystemPrompt = None
    user: UserConfig | None = None
    toolsets: ToolsetConfigs = Field(default_factory=dict)
    runtime: RuntimeConfig | None = None
    extras: Extras | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_server_sources(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "toolsets" in data:
            data["toolsets"] = cls.resolve_toolsets_config(
                data["toolsets"],
                cls.default_toolsets_config(),
            )
        if "user" in data:
            data["user"] = cls.resolve_user_config(
                data["user"],
                cls.default_user_config(),
            )
        return data

    @field_serializer("user")
    def serialize_user(self, value: UserConfig | None) -> dict[str, object] | None:
        if value is None:
            return None
        return value.model_dump(mode="json", exclude_none=True)

    @field_serializer("toolsets")
    def serialize_toolsets(self, value: ToolsetConfigs) -> dict[str, dict[str, object]]:
        return {
            name: config.model_dump(mode="json", exclude_none=True)
            for name, config in value.items()
        }

    @classmethod
    def default_toolsets_config(cls) -> ToolsetConfigs:
        value = cls.model_fields["toolsets"].get_default(call_default_factory=True)
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.__name__}.toolsets must be a mapping.")
        toolsets: ToolsetConfigs = {}
        for name, item in value.items():
            if not isinstance(name, str) or not name:
                raise TypeError(f"{cls.__name__}.toolsets keys must be strings.")
            toolsets[name] = ServerConfig.resolve_config(
                name,
                item,
                default=None,
                base_type=ToolsetConfig,
            )
        return toolsets

    @staticmethod
    def resolve_toolsets_config(
        value: object, defaults: ToolsetConfigs
    ) -> ToolsetConfigs:
        if not isinstance(value, Mapping):
            raise TypeError("TasksetConfig.toolsets must be a mapping.")
        toolsets: ToolsetConfigs = dict(defaults)
        for name, item in value.items():
            if not isinstance(name, str) or not name:
                raise TypeError(
                    "TasksetConfig.toolsets keys must be non-empty strings."
                )
            toolsets[name] = ServerConfig.resolve_config(
                name,
                item,
                default=defaults.get(name),
                base_type=ToolsetConfig,
            )
        return toolsets

    @staticmethod
    def enabled_toolsets(toolsets: ToolsetConfigs) -> ToolsetConfigs:
        return {
            name: toolset for name, toolset in toolsets.items() if bool(toolset.enabled)
        }

    @classmethod
    def default_user_config(cls) -> UserConfig | None:
        value = cls.model_fields["user"].get_default(call_default_factory=True)
        if value is None:
            return None
        return ServerConfig.resolve_config(
            "user",
            value,
            default=None,
            base_type=UserConfig,
        )

    @staticmethod
    def resolve_user_config(
        value: object, default: UserConfig | None
    ) -> UserConfig | None:
        if value is None:
            return None
        return ServerConfig.resolve_config(
            "user",
            value,
            default=default,
            base_type=UserConfig,
        )


ConfigT = TypeVar("ConfigT", bound=TasksetConfig)


class Taskset(Generic[ConfigT]):
    config: ConfigT
    task_type: type[Task] = Task

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
            resolved_id = self.config.id
            if resolved_id is not None and not isinstance(resolved_id, str):
                raise TypeError("taskset id must be a string.")
            self.id = resolved_id or type(self).__name__
            self.system_prompt = normalize_system_prompt(
                self.load_system_prompt(self.config),
                field_name="taskset.system_prompt",
            )
            self.user = self.config.user
            self.toolsets = TasksetConfig.enabled_toolsets(self.config.toolsets)
            self.handlers = self.load_handlers()
            self.signals = build_signals(self)
            for signal in self.signals:
                if signal["kind"] == "advantage":
                    raise ValueError(
                        "Taskset signals must be metrics or rewards; configure "
                        "env advantages with Env(advantage=...)."
                    )
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None

    def get_skills_dir(self) -> Traversable | Path | None:
        return discover_sibling_dir(type(self), "skills", require_non_empty=True)

    def get_upload_dirs(self) -> dict[str, Traversable | Path]:
        skills = self.get_skills_dir()
        return {} if skills is None else {"skills": skills}

    def load_system_prompt(self, config: ConfigT) -> SystemPrompt:
        return config.system_prompt

    def load_handlers(self) -> dict[LifecycleKind, list[Handler]]:
        handlers: dict[LifecycleKind, list[Handler]] = {
            "stop": [],
            "setup": [],
            "update": [],
            "cleanup": [],
            "teardown": [],
        }
        for kind in ("stop", "setup", "update", "cleanup", "teardown"):
            handlers[kind].extend(discover_decorated(self, kind))
        return handlers

    @property
    def has_group_signals(self) -> bool:
        return any(signal["stage"] == "group" for signal in self.signals)

    def to_task(self, task: Task | JsonData) -> Task:
        if isinstance(task, Task):
            return prepare_task(task)
        return task_from_dataset_record(task, self.task_type)

    def load_tasks(self, split: TaskSplit = "train") -> Tasks:
        if split not in ("train", "eval"):
            raise ValueError(f"Unknown task split: {split}")
        return []

    async def init_group(
        self, task: Task, num_rollouts: int
    ) -> tuple[list[Task], list[State]]:
        tasks = [task for _ in range(num_rollouts)]
        return tasks, [State(task_id=task.task_id) for task in tasks]

    def get_dataset(self) -> Dataset:
        if self._dataset is None:
            with config_ref_context(self.config):
                self._dataset = dataset_from_result_typed(
                    self.load_tasks(split="train"), self.task_type
                )
        return self._dataset

    def get_eval_dataset(self) -> Dataset:
        if self._eval_dataset is None:
            with config_ref_context(self.config):
                self._eval_dataset = dataset_from_result_typed(
                    self.load_tasks(split="eval"), self.task_type
                )
        return self._eval_dataset

    def __iter__(self):
        for record in self.get_dataset():
            yield self.to_task(dict(record))

    def __len__(self) -> int:
        return len(self.get_dataset())
