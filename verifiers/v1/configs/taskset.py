"""The taskset plugin's config: which rows load, under `--env.taskset.*`."""

from pathlib import Path

from pydantic import SerializeAsAny
from pydantic_config import BaseConfig

from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.types import ID


class TasksetConfig(BaseConfig):
    id: ID = ""
    """Installed taskset package, set with `--env.taskset.id` (or the
    positional `eval <taskset-id>`)."""
    task: SerializeAsAny[TaskConfig] = TaskConfig()
    """Config passed to each task, under `--env.taskset.task.*`."""
    system_prompt: Path | None = None
    """File whose text overrides each task's `TaskData.system_prompt` on
    iteration (e.g. a GEPA `best_system_prompt.txt`)."""

    @property
    def name(self) -> str:
        return self.id
