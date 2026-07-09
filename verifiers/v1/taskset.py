"""The taskset: a thin loader that yields typed tasks.

A `Taskset` is the data half of an environment: config in, tasks out. Per-row data
(prompt, image, ground truths) rides on each task's fields; load-time knobs (dataset,
split, seed) live on the taskset config; the task-facing knobs live under its `task`
subtree (`TasksetConfig.task`, a `TaskConfig`), which `tasks()` attaches to every row
so hooks and rewards read them off `self.config`. All per-task behavior — runtime
prep, tools, user simulation, `@reward`/`@metric` scoring — lives on the `Task`
subclass it yields (see `verifiers.v1.task`).

The class stays generic over its task and config types (`Taskset[TaskT, ConfigT]`)
so the loaders can read them: `taskset_config_type` narrows `--taskset.*` CLI/toml
flags to the real config, and `task_type` types the wire trace. Subclass: implement
`load`; consumers call `tasks`, which checks the one-type contract: a taskset yields
one concrete task type (replay rebuilds every saved row as the declared `TaskT`, so
a row of any other class would silently lose its own behavior there).
"""

from typing import Generic, get_args, get_origin

from verifiers.v1.errors import TaskError
from verifiers.v1.judge import check_judges
from verifiers.v1.task import (
    ConfigT,
    Task,
    TaskConfig,
    TaskT,
    TasksetConfig,
    task_config_cls,
)

__all__ = ["ConfigT", "TaskConfig", "Taskset", "TasksetConfig"]


class Taskset(Generic[TaskT, ConfigT]):
    """Generic over its task and config types, so `self.config` and `load` are fully
    typed (and the loaders can narrow CLI flags / type the wire trace off the generics).
    Subclass: implement `load` — per-row data goes in task fields; load-time knobs on
    the taskset config; task-facing knobs under its `task` subtree, which `tasks()`
    attaches to every row (`Task.config`)."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    def load(self) -> list[TaskT]:
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
        """`load`, plus the one-type contract check: every row must be exactly the declared
        `TaskT`. Enforced loudly because the wire is typed on the declaration — `replay`
        rebuilds every saved row as `task_type()`, so a row of any other class would
        silently replay without its own behavior. Consumers load through this; `load` is
        the subclass hook."""
        tasks = self.load()
        declared = self.task_type()
        kinds = {type(task) for task in tasks}
        if declared is not Task:
            if wrong := sorted(cls.__name__ for cls in kinds - {declared}):
                raise TaskError(
                    f"{type(self).__name__} declares task type {declared.__name__} "
                    f"but loaded {wrong} — a taskset yields one task type"
                )
        elif len(kinds) > 1:
            raise TaskError(
                f"{type(self).__name__} loaded mixed task types "
                f"{sorted(cls.__name__ for cls in kinds)} — a taskset yields one task type"
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
        # Stamp every row with the eval's task config (hooks read `self.config`; excluded
        # from the wire — `replay` re-stamps) and bake the config-plugged judges onto it,
        # after any judges the task already carries (a shared reward key raises) — so a
        # loaded task is complete and `Task.score` reads only `task.judges`.
        plugged = tuple(task_config.judges)
        stamped = []
        for task in tasks:
            update: dict = {"config": task_config}
            if plugged:
                merged = (*task.judges, *plugged)
                check_judges(merged)
                update["judges"] = merged
            stamped.append(task.model_copy(update=update))
        return stamped
