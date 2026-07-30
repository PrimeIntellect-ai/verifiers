"""The taskset: a thin loader that yields typed tasks.

A `Taskset` is the data half of an environment: config in, tasks out. `load()` is
the main hook that builds each task:

    def load(self) -> Iterable[MyTask]:
        for i in ...:
            yield MyTask(MyData(idx=i, ...), self.config.task)

`load` may also be a generator for infinite tasksets. There is a one-to-one
mapping between taskset and task type, i.e. a taskset may only yield one task
type.
"""

from __future__ import annotations

import copy
import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, ClassVar, Generic, Self

from pydantic_config import BaseConfig
from typing_extensions import TypeVar

from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.task import Task, TaskT, resolve_server_config
from verifiers.v1.utils.generic import concrete_type
from verifiers.v1.utils.sampling import sample

if TYPE_CHECKING:
    from verifiers.v1.mcp import Toolset

TasksetConfigT = TypeVar("TasksetConfigT", bound=TasksetConfig, default=TasksetConfig)


class Taskset(ABC, Generic[TaskT, TasksetConfigT]):
    INFINITE: bool = False
    """Whether the taskset is infinite (yields tasks forever). Class-declared;
    a `head(n)` view shadows it per instance (bounded by construction)."""

    tools: ClassVar[tuple[type[Toolset], ...]] = ()
    """Tool servers shared by all tasks in the taskset. The environment will
    spawn a single, global instance, reused across tasks."""

    def __init__(self, config: TasksetConfigT) -> None:
        self.config = config
        override = config.system_prompt
        self.system_prompt = override.read_text() if override is not None else None
        self.transform: Callable[[Iterator[TaskT]], Iterator[TaskT]] | None = None
        """Iteration transform carried by `head`/`shuffle` views (see `view`)."""

    @classmethod
    def task_type(cls) -> type[Task]:
        return concrete_type(cls, Task, origin=Taskset) or Task

    @abstractmethod
    def load(self) -> Iterable[TaskT]:
        """Build and yield the taskset's tasks; may be a generator (see module doc)."""

    def __iter__(self) -> Iterator[TaskT]:
        """Lazily iterate `load()` with the config-layer system prompt applied and
        any view transform on top — the read path; `load` is the subclass hook."""
        prompt = self.system_prompt
        tasks = (
            task.with_system_prompt(prompt) if prompt is not None else task
            for task in self.load()
        )
        yield from self.transform(tasks) if self.transform is not None else tasks

    def view(self, transform: Callable[[Iterator[TaskT]], Iterator[TaskT]]) -> Self:
        """A shallow copy of this taskset iterating through `transform`, composed
        onto any transform this taskset already carries — the general combinator
        behind `head`/`shuffle`; use it for custom lazy filters/maps."""
        clone = copy.copy(self)
        prev = self.transform
        clone.transform = (
            transform if prev is None else lambda tasks: transform(prev(tasks))
        )
        return clone

    def head(self, num_tasks: int) -> Self:
        """A lazy, always-finite view of the first `num_tasks` tasks."""
        view = self.view(lambda tasks: itertools.islice(tasks, num_tasks))
        view.INFINITE = False
        return view

    def shuffle(self) -> Self:
        """A fixed-seed shuffled view (materializes the receiver on iteration);
        raises on an infinite taskset — bound it first (`head(n).shuffle()`)."""
        if self.INFINITE:
            raise ValueError(
                f"{type(self).__name__} is infinite - cannot shuffle; "
                "bound it first with head(num_tasks)"
            )
        return self.view(lambda tasks: iter(sample(tasks, shuffle=True)))

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
