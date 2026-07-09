"""The taskset: a thin loader that yields data rows, constructed into typed tasks.

A `Taskset` is the data half of an environment: config in, tasks out. `load()` — the
one subclass hook — returns the rows as `TaskData` (per-row data: prompt, image,
ground truths); `tasks()` wraps each row in the declared `Task` type (behavior),
constructed with the config's task-facing subtree (`TasksetConfig.task`, a
`TaskConfig`, everything under `--taskset.task.*`). Load-time knobs (dataset, split,
seed) stay on the taskset config; the task reads its knobs off `self.config` and its
row off `self.data` (see `verifiers.v1.task`).

The class stays generic over its task and config types (`Taskset[TaskT, ConfigT]`)
so the loaders can read them: `taskset_config_type` narrows `--taskset.*` CLI/toml
flags to the real config, and `task_type` types the wire trace. One task type per
taskset — `tasks()` constructs it, so replay can rebuild every saved row as the
declared type's data and re-wrap it.
"""

from typing import Generic, get_args, get_origin

from verifiers.v1.errors import TaskError
from verifiers.v1.judge import check_judges
from verifiers.v1.task import (
    ConfigT,
    Task,
    TaskConfig,
    TaskData,
    TaskT,
    TasksetConfig,
    task_config_cls,
    task_data_cls,
)

__all__ = ["ConfigT", "TaskConfig", "Taskset", "TasksetConfig"]


class Taskset(Generic[TaskT, ConfigT]):
    """Generic over its task and config types, so `self.config` and `load` are fully
    typed (and the loaders can narrow CLI flags / type the wire trace off the generics).
    Subclass: implement `load`, returning the rows as the task's declared `TaskData` —
    `tasks()` constructs the declared `Task` around each row with `config.task`."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    def load(self) -> "list[TaskData]":
        raise NotImplementedError

    @classmethod
    def task_type(cls) -> type[Task]:
        """The declared `TaskT`, read off the `Taskset[TaskT, ConfigT]` generic across the
        MRO (most-derived specialization wins, so a thin wrapper re-binding only the config
        inherits its parent's task type). Falls back to the base `Task` when no subclass
        is given."""
        for klass in cls.__mro__:
            for orig in getattr(klass, "__orig_bases__", ()):
                if get_origin(orig) is Taskset:
                    for arg in get_args(orig):
                        if isinstance(arg, type) and issubclass(arg, Task):
                            return arg
        return Task

    def tasks(self) -> list[TaskT]:
        """`load`, then construct: every loaded row must be the declared task type's
        `TaskData` (loud, because the wire is typed on the declaration — `replay` rebuilds
        every saved row as it), each wrapped in the declared `Task` with the eval's task
        config (`config.task`) and the config-plugged judges baked onto its `judges`
        (after any the row already carries; a shared reward key raises) — so a
        constructed task is complete and `Task.score` reads only `data.judges`.
        Consumers load through this; `load` is the subclass hook."""
        rows = self.load()
        declared = self.task_type()
        declared_data = task_data_cls(declared)
        if wrong := sorted(
            {type(row).__name__ for row in rows} - {declared_data.__name__}
        ):
            raise TaskError(
                f"{type(self).__name__} must load {declared_data.__name__} rows "
                f"({declared.__name__}'s declared data type) but loaded {wrong} — "
                f"a taskset yields one task type"
            )
        # The config the tasks read (`Task.config`) is the taskset config's `task`
        # subtree; hold it to the task type's declared TaskConfig so a mismatched
        # narrowing fails at load, not as an AttributeError mid-rollout.
        task_config = self.config.task
        declared_config = task_config_cls(declared)
        if not isinstance(task_config, declared_config):
            raise TaskError(
                f"{type(self.config).__name__}.task is a {type(task_config).__name__}, "
                f"but {declared.__name__} declares {declared_config.__name__} — narrow "
                f"the config field (`task: {declared_config.__name__} = "
                f"{declared_config.__name__}()`)"
            )
        plugged = tuple(task_config.judges)
        tasks = []
        for row in rows:
            if plugged:
                merged = (*row.judges, *plugged)
                check_judges(merged)
                row = row.model_copy(update={"judges": merged})
            tasks.append(declared(row, config=task_config))
        return tasks
