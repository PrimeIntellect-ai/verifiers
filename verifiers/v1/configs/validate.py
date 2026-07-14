"""Configuration for model-free task validation."""

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.runtimes import DockerConfig, RuntimeConfig
from verifiers.v1.taskset import TasksetConfig


class CheckTimeoutConfig(BaseConfig):
    setup: float | None = None
    """Max wall-clock for the task's `setup` hook."""
    total: float | None = None
    """Max wall-clock for the check itself per task — the `validate` hook, or the debug
    command/script."""


class ValidateConfig(BaseConfig):
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    runtime: RuntimeConfig = DockerConfig()
    """Where each task's validation hooks run."""
    timeout: CheckTimeoutConfig = CheckTimeoutConfig()
    only_setup: bool = False
    """Run only `Task.setup`."""
    only_gold: bool = False
    """Run only `Task.setup` and `Task.validate`."""
    num_tasks: int | None = Field(
        None,
        ge=1,
        validation_alias=AliasChoices("num_tasks", "n", "num_examples", "batch_size"),
    )
    """How many tasks to validate (None = all)."""
    shuffle: bool = Field(False, validation_alias=AliasChoices("shuffle", "s"))
    """Shuffle tasks before taking the first `num_tasks`."""
    max_concurrent: int | None = Field(
        128, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Max tasks validated in flight at once (and, for a container runtime, live sandboxes)."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of the default info."""
    rich: bool = True
    """Show a live dashboard (one row per task) instead of per-task log lines."""

    @property
    def name(self) -> str:
        return self.taskset.name

    @model_validator(mode="after")
    def _validate_only(self):
        if self.only_setup and self.only_gold:
            raise ValueError("pass at most one of `--only-setup` or `--only-gold`")
        return self

    @model_validator(mode="before")
    @classmethod
    def _resolve_taskset(cls, data):
        from verifiers.v1.loaders import narrow_plugin_field, taskset_config_type

        narrow_plugin_field(data, "taskset", taskset_config_type)
        return data
