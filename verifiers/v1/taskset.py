"""The taskset: a thin loader that yields typed tasks.

A `Taskset` is the data half of an environment: config in, tasks out. `load()` — the
one subclass hook — builds each row's `TaskData` and wraps it in the task type with
the config's task-facing subtree:

    def load(self) -> Iterable[MyTask]:
        return [MyTask(MyData(idx=i, ...), self.config.task) for i in ...]

`load` may also be a generator, possibly infinite (declare `INFINITE = True`); runs
materialize what they need through `select`, the env server pulls task by task.

Load-time knobs live on the taskset config, task-facing knobs under its `task`
subtree, shared tool servers on `tools`. All per-task behavior lives on the `Task`.

The class stays generic (`Taskset[TaskT, TasksetConfigT]`) so the loaders can read
the types: `taskset_config_type` narrows `--env.taskset.*` flags, `task_type` types
the wire trace — one task type per taskset, so replay can rebuild saved rows.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar, Generic

from pydantic import SerializeAsAny
from pydantic_config import BaseConfig
from typing_extensions import TypeVar

from verifiers.v1.task import Task, TaskConfig, TaskT, resolve_server_config
from verifiers.v1.types import ID
from verifiers.v1.utils.generic import generic_type
from verifiers.v1.utils.install import env_name
from verifiers.v1.utils.sampling import sample

if TYPE_CHECKING:
    from verifiers.v1.mcp import Toolset

logger = logging.getLogger(__name__)


class TasksetConfig(BaseConfig):
    id: ID = ""
    """Local package or Hub `org/name[@version]`, set with `--env.taskset.id` (or the
    positional `eval <taskset-id>`)."""
    task: SerializeAsAny[TaskConfig] = TaskConfig()
    """Config passed to each task, under `--env.taskset.task.*`."""

    @property
    def name(self) -> str:
        return env_name(self.id)


TasksetConfigT = TypeVar("TasksetConfigT", bound=TasksetConfig, default=TasksetConfig)


class Taskset(Generic[TaskT, TasksetConfigT]):
    INFINITE: ClassVar[bool] = False
    """Whether `load` yields tasks forever. Inherent to the taskset, not a config
    knob: runs bound themselves with `select(num_tasks)`, and shuffle is impossible."""

    tools: ClassVar[tuple[type[Toolset], ...]] = ()
    """Tool server classes shared by one environment worker's rollouts."""

    def __init__(self, config: TasksetConfigT) -> None:
        self.config = config

    @classmethod
    def task_type(cls) -> type[Task]:
        """The taskset's declared `Task` subclass, read off the `Taskset[TaskT, ...]`
        generic — no data is loaded, so consumers (env server, replay) can cheaply
        rebuild wire rows as the declared type."""
        return generic_type(cls, Task, origin=Taskset) or Task

    def load(self) -> Iterable[TaskT]:
        raise NotImplementedError

    def select(
        self, num_tasks: int | None = None, shuffle: bool = False
    ) -> list[TaskT]:
        """Materialize the first `num_tasks` off `load` (all when `None`), pulled
        lazily. `shuffle` samples from the whole taskset instead (fixed-seed), which
        materializes everything first; on an `INFINITE` taskset it's a warned no-op —
        the first `num_tasks` generated are already an arbitrary sample."""
        if type(self).INFINITE:
            if num_tasks is None:
                raise ValueError(
                    f"{type(self).__name__} is infinite - select a bounded subset "
                    "with num_tasks (-n on the CLI)"
                )
            if shuffle:
                logger.warning(
                    "shuffle is a no-op on an infinite taskset - "
                    "taking the first %d generated tasks",
                    num_tasks,
                )
            return list(itertools.islice(self.load(), num_tasks))
        if shuffle:
            return sample(self.load(), shuffle=True, limit=num_tasks)
        return list(itertools.islice(self.load(), num_tasks))

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
