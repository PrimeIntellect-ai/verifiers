"""The taskset: a thin loader that yields typed tasks.

A `Taskset` is the data half of an environment: config in, tasks out. `load()` — the one
subclass hook — builds each row's `TaskData` and wraps it in the task type with the
config's task-facing subtree:

    def load(self) -> Iterable[MyTask]:
        return [MyTask(MyData(idx=i, ...), self.config.task) for i in ...]

`load` may also be a generator — yielding each task as it's built, possibly forever (a
procedural taskset; declare `INFINITE = True`). Runs materialize the tasks they need
through `select`, which pulls only that many off a generator; the env server instead
pulls `load` on demand, task by task.

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

import itertools
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Generic

from pydantic import SerializeAsAny, model_validator
from pydantic_config import BaseConfig
from typing_extensions import TypeVar

from verifiers.v1.task import TaskConfig, TaskT, resolve_server_config
from verifiers.v1.types import ID
from verifiers.v1.utils.install import env_name
from verifiers.v1.utils.sampling import sample

if TYPE_CHECKING:
    from verifiers.v1.mcp import Toolset

logger = logging.getLogger(__name__)


class TasksetConfig(BaseConfig):
    id: ID = ""
    """Local package or Hub `org/name[@version]`, set with `--taskset.id`."""
    task: SerializeAsAny[TaskConfig] = TaskConfig()
    """Config passed to each task, under `--taskset.task.*`."""
    system_prompt: str | None = None
    """Override `TaskData.system_prompt` for tasksets that honor it in `load()`."""
    system_prompt_file: Path | None = None
    """File override for `system_prompt`, mutually exclusive with `system_prompt`."""

    @model_validator(mode="after")
    def check_system_prompt_source(self) -> TasksetConfig:
        if self.system_prompt is not None and self.system_prompt_file is not None:
            raise ValueError("set `system_prompt` or `system_prompt_file`, not both")
        return self

    @property
    def name(self) -> str:
        return env_name(self.id)


TasksetConfigT = TypeVar("TasksetConfigT", bound=TasksetConfig, default=TasksetConfig)


def resolve_system_prompt(config: TasksetConfig) -> str | None:
    """Inline `system_prompt`, else contents of `system_prompt_file`, else None."""
    if config.system_prompt_file is not None:
        return config.system_prompt_file.read_text(encoding="utf-8")
    return config.system_prompt


class Taskset(Generic[TaskT, TasksetConfigT]):
    INFINITE: ClassVar[bool] = False
    """Whether `load` yields tasks forever (a procedural generator). Infinity is inherent
    to the taskset, not a config knob: consumers bound a run with `select(num_tasks)`
    (`-n` on the CLI), and shuffle is impossible."""

    tools: ClassVar[tuple[type[Toolset], ...]] = ()
    """Tool server classes shared by one environment worker's rollouts."""

    def __init__(self, config: TasksetConfigT) -> None:
        self.config = config

    def load(self) -> Iterable[TaskT]:
        raise NotImplementedError

    def select(
        self, num_tasks: int | None = None, shuffle: bool = False
    ) -> list[TaskT]:
        """Materialize the tasks a run needs: the first `num_tasks` off `load` (all of
        them when `None`), pulled lazily — a generator `load` only builds what the run
        takes. `shuffle` samples the subset from the whole taskset instead (the shared
        fixed-seed shuffle, `verifiers.v1.utils.sampling`), which materializes everything
        first; on an `INFINITE` taskset it's a no-op (warned) — the first `num_tasks`
        generated tasks are already an arbitrary sample."""
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
