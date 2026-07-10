"""The taskset: a thin loader that yields typed tasks.

A `Taskset` is the data half of an environment: config in, tasks out. `load()` — the one
subclass hook, and the one method consumers call — builds each row's `TaskData` and wraps
it in one of the taskset's declared task types with its task-facing config:

    def load(self) -> list[MyTask]:
        return [MyTask(MyData(idx=i, ...), self.config.task) for i in ...]

Load-time knobs (dataset, split, seed) live on the taskset config. A single-type taskset
usually narrows `TasksetConfig.task`; a multi-type taskset declares one explicit config
field per task type. An eval-wide shared tool server is declared on `tools` with its knobs
at the taskset level (`--taskset.tools.*`). All per-task behavior — runtime prep, tools,
user simulation, `@reward`/`@metric` scoring — lives on the `Task` (see
`verifiers.v1.task`).

The class stays generic over its task and config types (`Taskset[TaskT, TasksetConfigT]`)
so the loaders can read them: `TaskT` may be a union (`ProposerTask | SolvedTask`),
`taskset_config_type` narrows `--taskset.*` CLI/toml flags to the real config, and
`task_types` supplies the closed set replay may resolve from the wire.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic, get_args, get_origin

from pydantic import SerializeAsAny
from pydantic_config import BaseConfig
from typing_extensions import TypeVar

from verifiers.v1.errors import TaskError
from verifiers.v1.task import (
    Task,
    TaskConfig,
    TaskT,
    resolve_server_config,
    task_config_cls,
)
from verifiers.v1.types import ID
from verifiers.v1.utils.install import env_name

if TYPE_CHECKING:
    from verifiers.v1.mcp import Toolset


class TasksetConfig(BaseConfig):
    """Base taskset config: load-time knobs plus task-facing configs. A single-type
    taskset normally narrows `task`; a multi-type taskset adds one explicitly typed field
    per task type (for example `proposer` and `solved`)."""

    id: ID = ""
    """The taskset id, which selects this taskset: a local package, or an
    `org/name[@version]` package installed on demand from the Environments Hub (see
    `ID`). Set via `--taskset.id`."""
    task: SerializeAsAny[TaskConfig] = TaskConfig()
    """The conventional task-facing config for a single-type taskset: server placement,
    judges, scoring knobs. Multi-type tasksets use explicit sibling fields instead."""

    @property
    def name(self) -> str:
        """The taskset's package name (the id with any org / version stripped)."""
        return env_name(self.id)


TasksetConfigT = TypeVar("TasksetConfigT", bound=TasksetConfig, default=TasksetConfig)


class Taskset(Generic[TaskT, TasksetConfigT]):
    """Generic over its task types and config, so `self.config` and `load` are fully
    typed. `TaskT` may be a concrete task or a union of every task this factory owns.
    Subclass: implement `load`; additional taskset-specific factory methods may construct
    later task types from finished traces."""

    tools: ClassVar[tuple[type[Toolset], ...]] = ()
    """TASKSET-scoped (shared) tool server classes: each is launched ONCE per eval by the
    Environment and reached by every rollout — for an expensive, task-agnostic resource (a
    corpus, an index) built once. Declarative like `Task.tools`, but the scope is the
    registration site: taskset = eval-wide, task = per-rollout. A shared toolset declares a
    `SharedToolsetConfig` (no `colocated` — there's no single harness runtime), resolved off
    the TASKSET config's fields by `server_config` (so its knobs live at `--taskset.*`, not
    `--taskset.task.*`); it never receives a task (`setup_task` is not called)."""

    def __init__(self, config: TasksetConfigT) -> None:
        self.config = config

    def load(self) -> list[TaskT]:
        raise NotImplementedError

    @classmethod
    def task_types(cls) -> tuple[type[Task], ...]:
        """Concrete task types declared by `Taskset[TaskT, ConfigT]`, including unions."""
        for klass in cls.__mro__:
            for orig in getattr(klass, "__orig_bases__", ()):
                if get_origin(orig) is Taskset:
                    task_arg = get_args(orig)[0]
                    declared = tuple(
                        task
                        for task in get_args(task_arg) or (task_arg,)
                        if isinstance(task, type) and issubclass(task, Task)
                    )
                    if declared:
                        return declared
        return (Task,)

    def task_config(self, task_cls: type[Task]) -> TaskConfig:
        """Resolve a task type's config from an explicitly typed taskset-config field.

        Exact type wins, then a unique isinstance match. Multi-type tasksets use a
        distinct `TaskConfig` subclass and field for each task type.
        """
        if task_cls not in self.task_types():
            raise TaskError(
                f"{task_cls.__name__} is not declared by {type(self).__name__}"
            )
        config_cls = task_config_cls(task_cls)
        values = {
            name: getattr(self.config, name) for name in type(self.config).model_fields
        }
        matched = [name for name, value in values.items() if type(value) is config_cls]
        if not matched:
            matched = [
                name for name, value in values.items() if isinstance(value, config_cls)
            ]
        if len(matched) == 1:
            return values[matched[0]]
        detail = f"matching fields {matched}" if matched else "no matching field"
        raise TaskError(
            f"{type(self).__name__}: {detail} for {task_cls.__name__}'s "
            f"{config_cls.__name__}; declare exactly one explicit field of that type and "
            "use a distinct TaskConfig subclass per task type"
        )

    def server_config(self, server_cls: type) -> BaseConfig:
        """The config a `tools` entry is built with, resolved off `self.config` (the
        taskset config; see `resolve_server_config`). Override to pair explicitly."""
        return resolve_server_config(type(self).__name__, self.config, server_cls)

    def tool_servers(self) -> list[Toolset]:
        """Build this taskset's shared tool servers: one instance per class in `tools`,
        each constructed with `server_config(cls)`. Called once per eval by
        `Environment.shared_tools`; a Toolset instance is a launcher spec — the server
        itself runs as its own process (see `verifiers.v1.mcp`)."""
        return [cls(self.server_config(cls)) for cls in type(self).tools]

    @classmethod
    def task_type(cls) -> type[Task]:
        """The sole declared task type; fail loudly for a multi-type taskset."""
        declared = cls.task_types()
        if len(declared) != 1:
            raise TypeError(
                f"{cls.__name__} declares multiple task types "
                f"{[task.__name__ for task in declared]}; use task_types()"
            )
        return declared[0]
