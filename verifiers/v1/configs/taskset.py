"""The taskset plugin's config: which rows load, under `--env.taskset.*`."""

from pathlib import Path

from pydantic import SerializeAsAny
from pydantic_config import BaseConfig

from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.types import ID


class SharedTasksetConfig(BaseConfig):
    """The knobs every taskset has, whatever its id — what several tasksets of one run
    can share."""

    task: SerializeAsAny[TaskConfig] = TaskConfig()
    """Config passed to each task, under `--env.taskset.task.*`."""
    system_prompt: Path | None = None
    """File whose text overrides each task's `TaskData.system_prompt` on
    iteration (e.g. a GEPA `best_system_prompt.txt`)."""
    network_allow: list[str] | None = None
    """Execution-time destinations every task of this taskset may reach, composed
    with each task's own `network_allow` on iteration the way a task's policy
    composes with the runtime's: restrictions intersect, so `[]` allows none and
    `["*"]` adds no restriction. None leaves each task's policy alone, which lets
    an eval entrypoint fall back to its own default (`restrict_network_by_default`)."""
    network_block: list[str] | None = None
    """Execution-time destinations denied to every task of this taskset, combined
    with each task's own `network_block` on iteration. None adds none."""


class TasksetConfig(SharedTasksetConfig):
    id: ID = ""
    """Installed taskset package, set with `--env.taskset.id` (or the
    positional `eval <taskset-id>`)."""

    @property
    def name(self) -> str:
        return self.id
