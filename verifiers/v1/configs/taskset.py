"""The taskset plugin's config: which rows load, under `--env.taskset.*`."""

from pathlib import Path

from pydantic import SerializeAsAny, model_validator
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
    system_prompt: str | None = None
    """Config-layer system prompt: replaces each task's baked-in `TaskData.system_prompt`
    after `load()` (applied in `Taskset.select`). Lets a run set the system prompt without
    editing the taskset — e.g. hand a GEPA `best_system_prompt.txt` to eval/train via
    `system_prompt_file`. Mutually exclusive with `system_prompt_file`."""
    system_prompt_file: Path | None = None
    """File form of `system_prompt` (read as UTF-8), mutually exclusive with it."""

    @model_validator(mode="after")
    def check_system_prompt_source(self) -> "TasksetConfig":
        # Reject only when the run explicitly sets *both* — a taskset subclass that defaults
        # `system_prompt` to a non-None value (e.g. lean) must still accept `system_prompt_file`,
        # which overrides it (see `resolve_system_prompt`).
        set_fields = self.model_fields_set
        if "system_prompt" in set_fields and "system_prompt_file" in set_fields:
            raise ValueError("set `system_prompt` or `system_prompt_file`, not both")
        return self

    def resolve_system_prompt(self) -> str | None:
        """The effective config-layer system prompt: the file's text if `system_prompt_file`
        is set (it wins over a taskset's default `system_prompt`), else `system_prompt` (None
        when neither is)."""
        if self.system_prompt_file is not None:
            return self.system_prompt_file.read_text(encoding="utf-8")
        return self.system_prompt

    @property
    def name(self) -> str:
        return env_name(self.id)
