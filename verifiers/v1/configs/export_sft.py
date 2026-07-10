"""The `ExportSftConfig`: the config the `export-sft` CLI parses.

Export reads a finished run's saved traces and reshapes them into SFT rows — no taskset, no
runtime, no model — so it carries only selection knobs and a destination.
"""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class ExportSftConfig(BaseConfig):
    """Selection knobs + destination for exporting a run's traces as an SFT dataset."""

    min_reward: float | None = Field(
        None, validation_alias=AliasChoices("min_reward", "r")
    )
    """Keep only traces with `reward >= min_reward` (None = keep all). Generation-errored
    traces (`stop_condition == "error"`) are always dropped — a broken transcript is not a
    training sample. A scoring-only error keeps its finished transcript; its reward may be
    partial/zero, which this threshold handles."""
    drop_truncated: bool = False
    """Also drop traces cut off by a budget/limit (`max_turns`, token caps, timeouts)."""
    output_dir: Path | None = Field(
        None, validation_alias=AliasChoices("output_dir", "o")
    )
    """Where to write the dataset (`train.parquet`). None = `<run-dir>/sft`, which
    `load_dataset` (and prime-rl's `data.name`) reads directly."""
    push: str | None = None
    """A Hugging Face repo id to push the dataset to instead of writing parquet locally."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of the default info."""
