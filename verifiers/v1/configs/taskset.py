"""The taskset plugin's config: which rows load, under `--env.taskset.*`."""

from pathlib import Path

from pydantic import SerializeAsAny
from pydantic_config import BaseConfig

from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.types import ID
from verifiers.v1.utils.install import env_name


class TasksetConfig(BaseConfig):
    id: ID = ""
    """Local package or Hub `org/name[@version]`, set with `--env.taskset.id` (or the
    positional `eval <taskset-id>`)."""
    task: SerializeAsAny[TaskConfig] = TaskConfig()
    """Config passed to each task, under `--env.taskset.task.*`."""
    system_prompt: Path | None = None
    """File whose text overrides each task's `TaskData.system_prompt` in
    `Taskset.select` (e.g. a GEPA `best_system_prompt.txt`)."""

    @property
    def name(self) -> str:
        return env_name(self.id)
