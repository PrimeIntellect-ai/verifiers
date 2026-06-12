"""The `ValidateConfig`: the single config object the validate CLI parses.

Validation is per-task and model-free — it runs the taskset's `validate` hook (apply a gold
solution, run a verifier) in a runtime, not a harness rollout. So the config carries just the
taskset, the `runtime` its hook runs in (subprocess by default — many checks need no
container), the stage timeouts, and the run knobs. No harness, model, or sampling.
"""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.retries import RetryConfig
from verifiers.v1.runtimes import RuntimeConfig, SubprocessConfig
from verifiers.v1.taskset import TasksetConfig


class ValidateConfig(BaseConfig):
    """A taskset plus how to validate it. The taskset is selected by `--taskset.id` (with a
    bare positional shorthand, `validate gsm8k-v1`); its fields stay typed and overridable
    via dotted flags (`--taskset.split test`). The `validate` hook runs in `--runtime.*`
    (default subprocess; SWE rows carrying a per-task image need `--runtime.type docker`)."""

    uuid: str = Field(default_factory=lambda: str(uuid4()), exclude=True)
    """Auto-generated run id — the leaf of the output dir, so runs never overwrite."""
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    runtime: RuntimeConfig = SubprocessConfig()
    """Where each task's `validate` hook runs. Subprocess by default — a check that only needs
    to read task data or run a uv-script verifier wants no container; a task with a per-task
    image (SWE) needs `--runtime.type docker` (or prime)."""
    retries: RetryConfig = RetryConfig()
    """Reused only for `retries.runtime` — retry a task on a transient runtime fault (sandbox
    create/exec), each retry on a fresh runtime. Model/rollout retries don't apply here."""
    setup_timeout: float | None = None
    """Max wall-clock for the taskset's `setup` hook per task (None = no limit)."""
    validate_timeout: float | None = None
    """Max wall-clock for the taskset's `validate` hook per task (None = no limit)."""
    num_tasks: int | None = Field(
        None,
        validation_alias=AliasChoices("num_tasks", "n", "num_examples", "batch_size"),
    )
    """How many tasks to validate (None = all)."""
    shuffle: bool = Field(False, validation_alias=AliasChoices("shuffle", "s"))
    """Shuffle tasks before taking the first `num_tasks`."""
    max_concurrent: int | None = Field(
        16, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Max tasks validated in flight at once (and, for a container runtime, live sandboxes)."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of the default info."""
    dry_run: bool = False
    """Resolve + validate the config, write it to the output dir, then exit."""
    output_dir: Path | None = Field(
        None, validation_alias=AliasChoices("output_dir", "o")
    )
    """Where to write the run (config.toml + results.jsonl + summary.json). None = a fresh
    per-run dir under `outputs/<taskset>--validate/<uuid>` (so runs never overwrite)."""
    resume: bool = Field(False, validation_alias=AliasChoices("resume"))
    """Skip tasks already recorded in `output_dir/results.jsonl` and append the rest (needs an
    explicit `--output-dir`, since the default per-run dir is always fresh)."""

    @property
    def name(self) -> str:
        return self.taskset.name

    @model_validator(mode="before")
    @classmethod
    def _resolve_taskset(cls, data):
        """Narrow the generic `taskset` to its specific config type by `id`, so taskset-specific
        fields validate (mirrors `EnvConfig._resolve_plugins`, taskset-only)."""
        from verifiers.v1.loaders import taskset_config_type

        raw = data.get("taskset")
        if isinstance(raw, BaseConfig):
            raw = raw.model_dump()
        raw = dict(raw or {})
        ident = raw.get("id")
        if ident:
            data["taskset"] = taskset_config_type(ident).model_validate(
                {**raw, "id": ident}
            )
        return data
