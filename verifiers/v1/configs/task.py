"""Run-time task knobs (`--env.taskset.task.*`), read by `Task` behavior."""

from __future__ import annotations

from pydantic import model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.judge import Judges, check_judges, resolve_judges


class TaskConfig(BaseConfig):
    """Run-time knobs read by `Task` behavior.

    Subclass for server placement, judge, or scoring settings. Every field needs a
    default because constructing a task without a config builds the declared config type.
    Load-time dataset settings belong on `TasksetConfig` instead.
    """

    judges: Judges = []
    """Judge plugins run after task rewards, set through `--env.taskset.task.judges`."""

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
