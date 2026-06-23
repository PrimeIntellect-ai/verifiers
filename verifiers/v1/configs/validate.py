"""The `ValidateConfig`: the single config object the validate CLI parses.

Validation is per-task and model-free — it runs the taskset's `validate` hook (apply a gold
solution, run a verifier) in a runtime, not a harness rollout. So the config carries just the
taskset, the `runtime` its hook runs in (docker by default — a gold check often needs the
task's declared container), the stage timeouts, and the run knobs. No harness, model, or
sampling — and it's fire-and-forget: nothing is written to disk, so there's no output dir /
resume / dry-run.
"""

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.runtimes import DockerConfig, RuntimeConfig
from verifiers.v1.taskset import TasksetConfig


class ValidateConfig(BaseConfig):
    """A taskset plus how to validate it. The taskset is selected by `--taskset.id` (with a
    bare positional shorthand, `validate gsm8k-v1`); its fields stay typed and overridable
    via dotted flags (`--taskset.split test`). The `validate` hook runs in `--runtime.*`
    (docker by default; SWE rows carry their own per-task image)."""

    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    runtime: RuntimeConfig = DockerConfig()
    """Where each task's `validate` hook runs. Docker by default — unlike the eval harness
    (subprocess), a gold check often needs the task's declared container; a SWE task overrides
    the image per task. Use `--runtime.type subprocess` for a check that needs no container —
    one that only reads task data or runs a uv-script verifier (e.g. gsm8k)."""
    setup_timeout: float | None = None
    """Max wall-clock for the taskset's `setup` hook per task (None = no limit)."""
    validate_timeout: float | None = None
    """Max wall-clock for the taskset's `validate` hook per task (None = no limit)."""
    num_tasks: int | None = Field(
        None,
        ge=0,
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

    @model_validator(mode="before")
    @classmethod
    def _resolve_taskset(cls, data):
        """Narrow the generic `taskset` to its specific config type by `id` — the taskset-only
        half of `EnvConfig._resolve_plugins` (validate has no harness)."""
        from verifiers.v1.loaders import narrow_plugin_field, taskset_config_type

        narrow_plugin_field(data, "taskset", taskset_config_type)
        return data
