from __future__ import annotations

from importlib.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar, cast, final

from datasets import Dataset
from pydantic import Field

from .config import Config, ConfigSource
from .decorators import discover_decorated
from .state import State
from .task import Task
from .toolset import Toolset
from .runtime import RuntimeConfig
from .types import Handler, JsonData, TaskSplit, Tasks
from .user import User
from .utils.config_utils import (
    coerce_config,
    config_ref_context,
    config_type_from_class,
    registered_config_type,
    register_config_type,
)
from .utils.prompt_utils import SystemPrompt, normalize_system_prompt
from .utils.scoring_utils import SignalRecord, build_signals
from .utils.taskset_utils import (
    dataset_from_result_typed,
    discover_sibling_dir,
    prepare_task,
    task_from_dataset_record,
)

if TYPE_CHECKING:
    from .harness import Harness

LifecycleKind = str


class TasksetConfig(Config):
    id: str | None = None
    system_prompt: SystemPrompt = None
    user: User | None = None
    toolsets: list[Toolset] = Field(default_factory=list)
    runtime: RuntimeConfig | None = None


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
            self.user = self.load_user(self.config.user)
            self.toolsets = [
                normalize_toolset(item)
                for item in [
                    *(self.load_toolsets(self.config) or []),
                    *self.config.toolsets,
                ]
            ]
            self.handlers = self.load_handlers()
            self.signals = build_signals(self)
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None

    def get_skills_dir(self) -> Traversable | Path | None:
        return discover_sibling_dir(type(self), "skills", require_non_empty=True)

    def get_upload_dirs(self) -> dict[str, Traversable | Path]:
        skills = self.get_skills_dir()
        return {} if skills is None else {"skills": skills}

    def load_system_prompt(self, config: ConfigT) -> SystemPrompt:
        return config.system_prompt

    def load_user(self, config: User | None) -> User | None:
        return config

    def load_toolsets(self, config: ConfigT) -> list[Toolset]:
        return []

    def load_handlers(self) -> dict[LifecycleKind, list[Handler]]:
        handlers: dict[LifecycleKind, list[Handler]] = {
            "stop": [],
            "setup": [],
            "update": [],
            "cleanup": [],
            "teardown": [],
        }
        for kind in ("stop", "setup", "update", "cleanup", "teardown"):
            handlers[kind].extend(cast(list[Handler], discover_decorated(self, kind)))
        return handlers

    @property
    def has_group_signals(self) -> bool:
        return any(signal["stage"] == "group" for signal in self.signals)

    @property
    def has_advantages(self) -> bool:
        return any(signal["kind"] == "advantage" for signal in self.signals)

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
            yield task_from_dataset_record(dict(record), self.task_type)

    def __len__(self) -> int:
        return len(self.get_dataset())


def collect_owner_signals(taskset: Taskset, harness: "Harness") -> list[SignalRecord]:
    signals = list(taskset.signals)
    harness_signals = getattr(harness, "signals", [])
    seen = {signal["name"] for signal in signals}
    for signal in harness_signals:
        if signal["name"] in seen:
            raise ValueError(f"Signal {signal['name']!r} is defined twice.")
        signals.append(signal)
    return sorted(signals, key=lambda signal: (-signal["priority"], signal["name"]))


def normalize_toolset(value: Toolset | dict[str, object]) -> Toolset:
    if isinstance(value, Toolset):
        return value
    if isinstance(value, dict):
        return Toolset.model_validate(value)
    raise TypeError("Taskset toolsets must be Toolset objects or mappings.")
