"""Run-time task knobs (`--env.taskset.task.*`), read by `Task` behavior."""

from __future__ import annotations

from pydantic import Field, FiniteFloat, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.judge import Judges, check_judges, resolve_judges


class HookConfig(BaseConfig):
    """A hook function plugged into a task by import path."""

    fn: str
    """Import path to the hook function: `pkg.module.function`, `pkg.module:function`,
    or `path/to/file.py:function`."""
    priority: int = 0
    """Execution order among the task's hooks — higher runs first, ties break by name."""


class RewardConfig(HookConfig):
    """A reward hook plugged into a task by import path."""

    weight: FiniteFloat = 1.0


class TaskConfig(BaseConfig):
    """Run-time knobs read by `Task` behavior.

    Subclass for server placement, judge, or scoring settings. Every field needs a
    default because constructing a task without a config builds the declared config type.
    Load-time dataset settings belong on `TasksetConfig` instead.
    """

    judges: Judges = Field(default_factory=list)
    """Judge plugins run after task rewards, set through `--env.taskset.task.judges`."""

    stops: dict[str, HookConfig] = Field(default_factory=dict)
    """Stop conditions `(trace) -> bool` plugged by name, merged with the task's
    `@vf.stop` methods (a plugged hook replaces a decorated one with the same name),
    set through `--env.taskset.task.stops`."""
    metrics: dict[str, HookConfig] = Field(default_factory=dict)
    """Metrics `(task, trace, runtime) -> float` plugged by name, merged with the
    task's `@vf.metric` methods, set through `--env.taskset.task.metrics`."""
    rewards: dict[str, RewardConfig] = Field(default_factory=dict)
    """Weighted rewards `(task, trace, runtime) -> float` plugged by name, merged with
    the task's `@vf.reward` methods, set through `--env.taskset.task.rewards`."""

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
