"""The taskset: a thin loader that yields typed tasks.

A `Taskset` is the data half of an environment: config in, tasks out. `load()` — the one
subclass hook, and the one method consumers call — builds each row's `TaskData` and wraps
it in the task type with the config's task-facing subtree:

    def load(self) -> list[MyTask]:
        return [MyTask(MyData(idx=i, ...), self.config.task) for i in ...]

Load-time knobs (dataset, split, seed) live on the taskset config; the task-facing knobs
under its `task` subtree (`TasksetConfig.task`, a `TaskConfig`, everything under
`--taskset.task.*`); an eval-wide shared tool server is declared on `tools` with its knobs
at the taskset level (`--taskset.tools.*`). All per-task behavior — runtime prep, tools,
user simulation, `@reward`/`@metric` scoring — lives on the `Task` (see
`verifiers.v1.task`).

The class stays generic over its task and config types (`Taskset[TaskT, TasksetConfigT]`)
so the loaders can read them: `taskset_config_type` narrows `--taskset.*` CLI/toml flags
to the real config, and `task_type` types the wire trace — one task type per taskset, so
replay can rebuild every saved row as the declared type's data and re-wrap it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic, get_args, get_origin

from pydantic import SerializeAsAny
from pydantic_config import BaseConfig
from typing_extensions import TypeVar

from verifiers.v1.task import Task, TaskConfig, TaskT, resolve_server_config
from verifiers.v1.types import ID
from verifiers.v1.utils.install import env_name

if TYPE_CHECKING:
    from verifiers.v1.mcp import Toolset


class TasksetConfig(BaseConfig):
    """Base taskset config: load-time knobs (dataset, split, seed, ...) plus the task's own
    config under `task`. Subclass to add task-generation knobs; narrow `task` to your
    `TaskConfig` subclass when the task reads knobs (`task: MyTaskConfig = MyTaskConfig()`),
    so `--taskset.task.*` flags validate against it. `load()` passes `config.task` to every
    task it constructs."""

    id: ID = ""
    """The taskset id, which selects this taskset: a local package, or an
    `org/name[@version]` package installed on demand from the Environments Hub (see
    `ID`). Set via `--taskset.id`."""
    task: SerializeAsAny[TaskConfig] = TaskConfig()
    """The task-facing config, passed to every constructed task (`Task.config`): server
    placement, judges, scoring knobs. Everything under `--taskset.task.*`."""

    @property
    def name(self) -> str:
        """The taskset's package name (the id with any org / version stripped)."""
        return env_name(self.id)


TasksetConfigT = TypeVar("TasksetConfigT", bound=TasksetConfig, default=TasksetConfig)


class Taskset(Generic[TaskT, TasksetConfigT]):
    """Generic over its task and config types, so `self.config` and `load` are fully
    typed (and the loaders can narrow CLI flags / type the wire trace off the generics).
    Subclass: implement `load`, constructing each task from its row's data and the
    config's task subtree (`MyTask(data, self.config.task)`)."""

    tools: ClassVar[tuple[type[Toolset], ...]] = ()
    """TASKSET-scoped (shared) tool server classes: each is launched ONCE per eval by the
    Environment and reached by every rollout — for an expensive, task-agnostic resource (a
    corpus, an index) built once. Declarative like `Task.tools`, but the scope is the
    registration site: taskset = eval-wide, task = per-rollout. A shared toolset declares a
    `SharedToolsetConfig` (no `colocated` — there's no single harness runtime), resolved off
    the TASKSET config's fields by `server_config` (so its knobs live at `--taskset.*`, not
    `--taskset.task.*`); it never receives a task (`setup_task` is not called)."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Mirror of Task's check: taskset scope means shared, so the config must be a
        # SharedToolsetConfig (a ToolsetConfig's `colocated` flag would be silently ignored).
        from verifiers.v1.mcp.toolset import SharedToolsetConfig

        for toolset in cls.tools:
            if not issubclass(toolset._config_cls(), SharedToolsetConfig):
                raise TypeError(
                    f"{cls.__name__}.tools declares {toolset.__name__}, whose config is not a "
                    "SharedToolsetConfig — Taskset.tools servers are shared (launched once per "
                    "eval); a per-rollout server belongs on Task.tools"
                )

    def __init__(self, config: TasksetConfigT) -> None:
        self.config = config

    def load(self) -> list[TaskT]:
        raise NotImplementedError

    def server_config(self, server_cls: type) -> BaseConfig:
        """The config a `tools` entry is built with, resolved off `self.config` (the
        taskset config; see `resolve_server_config`). Override to pair explicitly."""
        return resolve_server_config(
            type(self).__name__,
            self.config,
            server_cls,
            sole=len(set(type(self).tools)) == 1,
        )

    def tool_servers(self) -> list[Toolset]:
        """Build this taskset's shared tool servers: one instance per class in `tools`,
        each constructed with `server_config(cls)`. Called once per eval by
        `Environment.shared_tools`; a Toolset instance is a launcher spec — the server
        itself runs as its own process (see `verifiers.v1.mcp`)."""
        return [cls(self.server_config(cls)) for cls in type(self).tools]

    @classmethod
    def task_type(cls) -> type[Task]:
        """The declared `TaskT`, read off the `Taskset[TaskT, TasksetConfigT]` generic
        across the MRO (most-derived specialization wins, so a thin wrapper re-binding only
        the config inherits its parent's task type). Falls back to the base `Task` when no
        subclass is given."""
        for klass in cls.__mro__:
            for orig in getattr(klass, "__orig_bases__", ()):
                if get_origin(orig) is Taskset:
                    for arg in get_args(orig):
                        if isinstance(arg, type) and issubclass(arg, Task):
                            return arg
        return Task
