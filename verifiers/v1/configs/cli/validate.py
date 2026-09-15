"""Configuration for model-free task validation."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.cli.run import RunConfig
from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.runtimes import PrimeConfig, RuntimeConfig


class CheckTimeoutConfig(BaseConfig):
    setup: float | None = None
    """Max wall-clock for the task's `setup` hook."""
    total: float | None = None
    """Max wall-clock for the check itself per task — the `validate` hook, or the debug
    command/script."""


class ValidateConfig(BaseConfig):
    run: RunConfig = Field(default_factory=RunConfig)
    """Run identity: `run.name` auto-generates as `<taskset>--validate--<short-id>` and
    names the run directory under `output_dir`."""
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    runtime: RuntimeConfig = PrimeConfig()
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
    output_dir: Path = Field(
        Path("outputs"), validation_alias=AliasChoices("output_dir", "o")
    )
    """Directory that groups related runs. The run (`configs/validate.json`,
    `results.jsonl`, `summary.json`, `logs/validate.log`) writes to
    `output_dir / run.dir`."""
    resume: bool = Field(False, exclude=True)
    """Re-run the run's missing, errored, and timed-out tasks in place. The run dir comes
    from the resolved config (`output_dir / run.dir`), so resume with the run's own
    config — e.g. `uv run validate @ <run-dir>/configs/validate.json --resume`.
    Excluded from the saved config."""
    clean: bool = Field(False, exclude=True)
    """Delete the run directory (`output_dir / run.dir`) before running, overwriting a
    previous run's results. Excluded from the saved config."""

    @property
    def name(self) -> str:
        return self.taskset.name

    @model_validator(mode="after")
    def auto_setup_run_name(self):
        if self.run.name is None:
            self.run.name = f"{self.name}--validate--{uuid4().hex[:8]}".lower()
        if self.run.dir is None:
            self.run.dir = self.run.name
        return self

    @model_validator(mode="after")
    def _validate_only(self):
        if self.only_setup and self.only_gold:
            raise ValueError("pass at most one of `--only-setup` or `--only-gold`")
        return self

    @model_validator(mode="before")
    @classmethod
    def _resolve_taskset(cls, data):
        from verifiers.v1.utils.loaders import narrow_plugin_field, taskset_config_type

        narrow_plugin_field(data, "taskset", taskset_config_type)
        return data
