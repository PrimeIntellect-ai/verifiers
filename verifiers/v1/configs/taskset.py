"""The taskset plugin's config: which rows load, under `--env.taskset.*`."""

import logging
from pathlib import Path

from pydantic import SerializeAsAny
from pydantic_config import BaseConfig

from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.types import ID
from verifiers.v1.utils.install import env_name

logger = logging.getLogger(__name__)


class TasksetConfig(BaseConfig):
    id: ID = ""
    """Local package or Hub `org/name[@version]`, set with `--env.taskset.id` (or the
    positional `eval <taskset-id>`)."""
    task: SerializeAsAny[TaskConfig] = TaskConfig()
    """Config passed to each task, under `--env.taskset.task.*`."""
    system_prompt: str | None = None
    """Config-layer system prompt: replaces each task's baked-in `TaskData.system_prompt`
    after `load()` (applied in `Taskset.select`). Lets a run set the system prompt without
    editing the taskset — e.g. hand a GEPA `best_system_prompt.txt` to eval/train via
    `system_prompt_file`. When both are set, `system_prompt_file` takes precedence."""
    system_prompt_file: Path | None = None
    """File form of `system_prompt` (read as UTF-8); takes precedence over `system_prompt`
    (including a taskset's default) when both are set."""

    def resolve_system_prompt(self) -> str | None:
        """The effective config-layer system prompt: the file's text if `system_prompt_file`
        is set, else `system_prompt` (None when neither is). The file wins over `system_prompt`
        — including a taskset's default — rather than erroring, so it survives a `model_dump` /
        revalidate round trip across the env-server boundary; a warning flags the shadowed value."""
        if self.system_prompt_file is not None:
            if self.system_prompt is not None:
                logger.warning(
                    "taskset config sets both system_prompt and system_prompt_file; using "
                    "system_prompt_file (%s) and ignoring system_prompt",
                    self.system_prompt_file,
                )
            return self.system_prompt_file.read_text(encoding="utf-8")
        return self.system_prompt

    @property
    def name(self) -> str:
        return env_name(self.id)
