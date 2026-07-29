"""The taskset plugin's config: which rows load, under `--env.taskset.*`."""

from pathlib import Path
from typing import Any

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
    `system_prompt_file`. Mutually exclusive with `system_prompt_file`; the file form is
    collapsed into this string at validation so the runtime only sees `system_prompt`."""
    system_prompt_file: Path | None = None
    """File form of `system_prompt` (read as UTF-8 at validation). Mutually exclusive with
    `system_prompt`; after validation the text lives on `system_prompt` and this is `None`."""

    @model_validator(mode="before")
    @classmethod
    def collapse_system_prompt_file(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        prompt = data.get("system_prompt")
        prompt_file = data.get("system_prompt_file")
        if prompt is not None and prompt_file is not None:
            raise ValueError("set `system_prompt` or `system_prompt_file`, not both")
        if prompt_file is not None:
            path = prompt_file if isinstance(prompt_file, Path) else Path(prompt_file)
            data["system_prompt"] = path.read_text(encoding="utf-8")
            data["system_prompt_file"] = None
        return data

    @property
    def name(self) -> str:
        return env_name(self.id)
