"""Offline re-scoring config; source eval-only fields are ignored."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, ConfigDict, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.taskset import TasksetConfig


class ReplayConfig(BaseConfig):
    model_config = ConfigDict(extra="ignore")

    uuid: str = Field(default_factory=lambda: str(uuid4()), exclude=True)
    """Auto-generated run id — the leaf of the output dir, so replays never overwrite."""
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    """The taskset, selected by `--taskset.id` (narrowed to its config type). Its
    `taskset.task.judges` replace the source run's judges entirely (and new ones
    join); set them via `@ file.toml` / dotted flags."""
    num_traces: int | None = Field(
        None, validation_alias=AliasChoices("num_traces", "n")
    )
    """How many saved traces to re-score (None = all)."""
    num_rescores: int = Field(1, validation_alias=AliasChoices("num_rescores", "r"))
    """Re-score each selected trace this many times — e.g. to measure judge variance across
    repeated gradings of the same trace. Named distinctly from eval's `num_rollouts` so the
    replayed run's own `num_rollouts` (in the base `config.toml`) is ignored, not inherited."""
    max_concurrent: int | None = Field(
        128, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Max traces re-scored (judge calls) in flight at once."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of the default info."""
    dry_run: bool = False
    """Resolve + validate the config and write it to the output dir, then exit (no re-scoring)."""
    rich: bool = True
    """Show a live dashboard (one row per trace) instead of per-trace log lines."""
    output_dir: Path | None = Field(
        None, validation_alias=AliasChoices("output_dir", "o")
    )
    """Where to write the re-scored run (config.toml + traces.jsonl). None = a fresh per-run
    dir under `outputs/<taskset>--replay/<uuid>` (so a replay never overwrites the source run)."""

    @property
    def name(self) -> str:
        return self.taskset.name

    @model_validator(mode="before")
    @classmethod
    def _resolve_taskset(cls, data):
        from verifiers.v1.loaders import narrow_plugin_field, taskset_config_type

        narrow_plugin_field(data, "taskset", taskset_config_type)
        return data
