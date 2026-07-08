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
`load`.
"""

from functools import cached_property
from typing import Generic, TypeVar

from pydantic import model_validator
from pydantic_config import BaseConfig

from verifiers.v1.judge import Judge, Judges
from verifiers.v1.task import TaskT
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
    package, or a hub `org/name[@version]` package exporting a `Judge` subclass — and run by
    `Task.score` after the task's own `@reward`s: grading plugged into any taskset/harness pair
    from the eval config alone, no taskset code. Each entry records its verdict in
    `trace.rewards` under its `name` with its `weight` (see `JudgeConfig`)."""

    @property
    def name(self) -> str:
        """The taskset's package name (the id with any org / version stripped)."""
        return env_name(self.id)

    @model_validator(mode="before")
    @classmethod
    def _resolve_judges(cls, data):
        """Narrow each `judges` entry to the config type its `id` resolves to (mirrors
        `EnvConfig._resolve_plugins`), so judge-specific fields (e.g. rubric's `path`)
        validate against the real config instead of being rejected by the base type."""
        if not isinstance(data, dict) or not data.get("judges"):
            return data
        from verifiers.v1.loaders import judge_config_type

        entries = []
        for entry in data["judges"]:
            raw = entry.model_dump() if isinstance(entry, BaseConfig) else dict(entry)
            if not raw.get("id"):
                raise ValueError(
                    "each `judges` entry needs an `id` (a judge plugin: `reference`, "
                    "`rubric`, a local package, or a hub `org/name` package)"
                )
            entries.append(judge_config_type(raw["id"]).model_validate(raw))
        data["judges"] = entries
        return data

    @model_validator(mode="after")
    def _check_judges(self) -> "TasksetConfig":
        """Validate the resolved `judges` — after the before-hook so class-level *defaults*
        (which never pass through it, e.g. a taskset config pre-plugging a judge) are held
        to the same rules: an `id` on every entry, and no two entries sharing a reward key
        (the second would clobber the first's verdict)."""
        names = []
        for entry in self.judges:
            if not entry.id:
                raise ValueError(
                    "each `judges` entry needs an `id` (a judge plugin: `reference`, "
                    "`rubric`, a local package, or a hub `org/name` package)"
                )
            names.append(entry.name or env_name(entry.id))
        if duplicates := {name for name in names if names.count(name) > 1}:
            raise ValueError(
                f"`judges` entries share a reward key {sorted(duplicates)}; set a "
                "distinct `name` on each to keep both verdicts"
            )
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

    @cached_property
    def judges(self) -> list[Judge]:
        """The plugged judges, built once from `config.judges` (each entry resolved by its
        `id` — see `JudgeConfig` / `verifiers.v1.judges`). Attached to each task
        (`Task.judges`) by the scoring caller — the Environment at episode time, `replay`
        before re-scoring; `Task.score` runs them after the task's decorated rewards."""
        from verifiers.v1.loaders import load_judge

        return [load_judge(entry) for entry in self.config.judges]
