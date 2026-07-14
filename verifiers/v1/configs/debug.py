"""The `DebugConfig`: setup a task runtime and run one command or host script."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.validate import CheckTimeoutConfig
from verifiers.v1.runtimes import DockerConfig, RuntimeConfig
from verifiers.v1.taskset import TasksetConfig


class DebugConfig(BaseConfig):
    uuid: str = Field(default_factory=lambda: str(uuid4()), exclude=True)
    """Auto-generated run id, used as the default output directory leaf."""
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    runtime: RuntimeConfig = DockerConfig()
    """Where each task's setup hook and debug action run."""
    command: str | None = None
    """Inline shell command executed as `sh -lc <command>` after setup."""
    script_path: Path | None = Field(
        None, validation_alias=AliasChoices("script_path", "script")
    )
    """Host script to upload and execute after setup."""
    timeout: CheckTimeoutConfig = CheckTimeoutConfig()
    """Per-task stage timeouts: `--timeout.setup` for the `setup` hook, `--timeout.total`
    for the debug command/script."""
    num_tasks: int | None = Field(
        None,
        ge=1,
        validation_alias=AliasChoices("num_tasks", "n", "num_examples", "batch_size"),
    )
    """How many tasks to debug (None = all)."""
    shuffle: bool = Field(False, validation_alias=AliasChoices("shuffle", "s"))
    """Shuffle tasks before taking the first `num_tasks`."""
    max_concurrent: int | None = Field(
        128, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Max tasks debugged in flight at once."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of the default info."""
    output_dir: Path | None = Field(
        None, validation_alias=AliasChoices("output_dir", "o")
    )
    """Where to write `config.toml` and `traces.jsonl`. None = a fresh per-run dir."""

    @property
    def name(self) -> str:
        return self.taskset.name

    @model_validator(mode="before")
    @classmethod
    def resolve_taskset(cls, data):
        from verifiers.v1.loaders import narrow_plugin_field, taskset_config_type

        narrow_plugin_field(data, "taskset", taskset_config_type)
        return data

    @model_validator(mode="after")
    def validate_action(self):
        if (self.command is None) == (self.script_path is None):
            raise ValueError("pass exactly one of `--command` or `--script-path`")
        if self.script_path is not None and not self.script_path.is_file():
            raise ValueError(
                f"script_path does not exist or is not a file: {self.script_path}"
            )
        return self
