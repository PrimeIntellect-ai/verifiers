"""Run-time task knobs (`--env.taskset.task.*`), read by `Task` behavior."""

from __future__ import annotations

from pydantic import Field, FiniteFloat, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.judge import Judges, check_judges, resolve_judges


class DecoratedFunctionConfig(BaseConfig):
    """A plugged function standing in for a decorated task method."""

    fn: str | None = None
    """Import path to the function: `pkg.module.function`, `pkg.module:function`,
    or `path/to/file.py:function`. Unset keeps the task's decorated method with the
    entry's name and only overrides its metadata (priority, weight)."""

    priority: int | None = None
    """Execution order, like the decorator's — higher runs first, ties break by name.
    Unset keeps the replaced method's priority (0 for a fresh function)."""


class RewardFunctionConfig(DecoratedFunctionConfig):
    """A plugged function standing in for a `@vf.reward` task method."""

    weight: FiniteFloat | None = None
    """Unset keeps the replaced method's weight (1.0 for a fresh function)."""


class TaskConfig(BaseConfig):
    """Run-time knobs read by `Task` behavior.

    Subclass for server placement, judge, or scoring settings. Every field needs a
    default because constructing a task without a config builds the declared config type.
    Load-time dataset settings belong on `TasksetConfig` instead.
    """

    judges: Judges = Field(default_factory=list)
    """Judge plugins run after task rewards, set through `--env.taskset.task.judges`."""

    stops: dict[str, DecoratedFunctionConfig] = Field(default_factory=dict)
    """Stop conditions `(trace) -> bool` plugged by name, merged with the task's
    `@vf.stop` methods (a plugged function replaces a decorated one with the same
    name)."""
    metrics: dict[str, DecoratedFunctionConfig] = Field(default_factory=dict)
    """Metrics `(task, trace, runtime) -> float` plugged by name, merged with the
    task's `@vf.metric` methods."""
    rewards: dict[str, RewardFunctionConfig] = Field(default_factory=dict)
    """Weighted rewards `(task, trace, runtime) -> float` plugged by name, merged with
    the task's `@vf.reward` methods."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_judges(cls, data):
        if isinstance(data, dict) and data.get("judges"):
            data["judges"] = resolve_judges(data["judges"])
        return data

    @model_validator(mode="after")
    def _check_judges(self) -> TaskConfig:
        check_judges(self.judges)
        return self
