"""The taskset: a thin loader that yields typed tasks.

A `Taskset` is the data half of an environment: config in, tasks out. It resolves its
config at load time and bakes the result into each task's fields, so tasks come out
self-contained — everything a rollout needs (prompt, image, ground truths, scoring
knobs) rides on the task. All per-task behavior — runtime prep, tools, user
simulation, `@reward`/`@metric` scoring — lives on the `Task` subclass it yields
(see `verifiers.v1.task`).

The class stays generic over its task and config types (`Taskset[TaskT, ConfigT]`)
so the loaders can read them: `taskset_config_type` narrows `--taskset.*` CLI/toml
flags to the real config, and `task_type` types the wire trace. Subclass: implement
`load`; consumers call `tasks`, which checks the one-type contract: a taskset yields
one concrete task type (replay rebuilds every saved row as the declared `TaskT`, so
a row of any other class would silently lose its own behavior there).
"""

from typing import Generic, TypeVar, get_args, get_origin

from pydantic import model_validator
from pydantic_config import BaseConfig

from verifiers.v1.errors import TasksetError
from verifiers.v1.judge import Judges, check_judges, resolve_judges
from verifiers.v1.task import Task, TaskT
from verifiers.v1.types import ID
from verifiers.v1.utils.install import env_name


class TasksetConfig(BaseConfig):
    """Base taskset config. Subclass to add task-generation knobs."""

    id: ID = ""
    """The taskset id, which selects this taskset: a local package, or an
    `org/name[@version]` package installed on demand from the Environments Hub (see
    `ID`). Set via `--taskset.id`."""
    judges: Judges = []
    """Config-plugged judges, each resolved by `id` — a built-in (`reference`, `rubric`), a local
    package, or a hub `org/name[@version]` package exporting a `Judge` subclass: grading plugged
    into any taskset/harness pair from the eval config alone, no taskset code. `tasks()` appends
    these to every row's own `Task.judges`, and `Task.score` runs them after the task's
    `@reward`s. Each entry records its verdict in `trace.rewards` under its `name` with its
    `weight` (see `JudgeConfig`)."""

    @property
    def name(self) -> str:
        """The taskset's package name (the id with any org / version stripped)."""
        return env_name(self.id)

    @model_validator(mode="before")
    @classmethod
    def _resolve_judges(cls, data):
        """Narrow each `judges` entry to the config type its `id` resolves to (see
        `judge.resolve_judges`), so judge-specific fields (e.g. rubric's `path`)
        validate against the real config instead of being rejected by the base type."""
        if isinstance(data, dict) and data.get("judges"):
            data["judges"] = resolve_judges(data["judges"])
        return data

    @model_validator(mode="after")
    def _check_judges(self) -> "TasksetConfig":
        """Validate the resolved `judges` — after the before-hook so class-level *defaults*
        (which never pass through it, e.g. a taskset config pre-plugging a judge) are held
        to the same rules (see `judge.check_judges`)."""
        check_judges(self.judges)
        return self


ConfigT = TypeVar("ConfigT", bound=TasksetConfig)


class Taskset(Generic[TaskT, ConfigT]):
    """Generic over its task and config types, so `self.config` and `load` are fully
    typed (and the loaders can narrow CLI flags / type the wire trace off the generics).
    Subclass: implement `load` — resolve config there and bake the results into task
    fields, so each task carries everything its hooks and rewards need."""

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
                raise TasksetError(
                    f"{type(self).__name__} declares task type {declared.__name__} "
                    f"but loaded {wrong} — a taskset yields one task type"
                )
        elif len(kinds) > 1:
            raise TasksetError(
                f"{type(self).__name__} loaded mixed task types "
                f"{sorted(cls.__name__ for cls in kinds)} — a taskset yields one task type"
            )
        if plugged := tuple(self.config.judges):
            # Bake the config-plugged judges onto every row, after any judges the task
            # already carries (a shared reward key raises) — so a loaded task is complete
            # and `Task.score` reads only `task.judges`.
            baked = []
            for task in tasks:
                merged = (*task.judges, *plugged)
                check_judges(merged)
                baked.append(task.model_copy(update={"judges": merged}))
            tasks = baked
        return tasks
