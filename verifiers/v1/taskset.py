"""The taskset: a thin loader that yields typed tasks.

A `Taskset` is the data half of an environment: config in, tasks out. `load()` — the one
subclass hook, and the one method consumers call — builds each row's `TaskData` and wraps
it in the task type with the config's task-facing subtree:

    def load(self) -> list[MyTask]:
        return [MyTask(MyData(idx=i, ...), self.config.task) for i in ...]

Load-time knobs (dataset, split, seed) live on the taskset config; the task-facing knobs
under its `task` subtree (`TasksetConfig.task`, a `TaskConfig`, everything under
`--taskset.task.*`); a worker-scoped shared tool server is declared on `tools` with its knobs
at the taskset level (`--taskset.tools.*`). All per-task behavior — runtime prep, tools,
user simulation, `@reward`/`@metric` scoring — lives on the `Task` (see
`verifiers.v1.task`).

The class stays generic over its task and config types (`Taskset[TaskT, TasksetConfigT]`)
so the loaders can read them: `taskset_config_type` narrows `--taskset.*` CLI/toml flags
to the real config, and `task_type` types the wire trace — one task type per taskset, so
replay can rebuild every saved row as the declared type's data and re-wrap it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic

from pydantic import SerializeAsAny
from pydantic_config import BaseConfig
from typing_extensions import TypeVar

from verifiers.v1.task import TaskConfig, TaskT, resolve_server_config
from verifiers.v1.types import ID
from verifiers.v1.utils.install import env_name

if TYPE_CHECKING:
    from verifiers.v1.mcp import Toolset


class TasksetConfig(BaseConfig):
    id: ID = ""
    """Local package or Hub `org/name[@version]`, set with `--taskset.id`."""
    task: SerializeAsAny[TaskConfig] = TaskConfig()
    """Config passed to each task, under `--taskset.task.*`."""

    @property
    def name(self) -> str:
        return env_name(self.id)


TasksetConfigT = TypeVar("TasksetConfigT", bound=TasksetConfig, default=TasksetConfig)


class Taskset(Generic[TaskT, TasksetConfigT]):
    tools: ClassVar[tuple[type[Toolset], ...]] = ()
    """Tool server classes shared by one environment worker's rollouts."""

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
        return [cls(self.server_config(cls)) for cls in type(self).tools]
