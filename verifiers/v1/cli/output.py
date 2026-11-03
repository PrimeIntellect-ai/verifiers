"""CLI output directories and saved launch/config files."""

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from verifiers.v1.configs.cli.run import RunConfig
from verifiers.v1.utils.trace_store import TRACES_FILE

CONFIG_DIR = "configs"
"""Directory inside a run dir holding its configs: the launch TOML copied verbatim to
`configs/<cli>.toml`, and the resolved config at `configs/resolved/<cli>.json`
(re-runnable via `@ <run-dir>/configs/resolved/<cli>.json`). Resolved configs are JSON,
not TOML: JSON keeps nulls, so explicit None settings round-trip exactly on re-parse."""

RESOLVED_DIR = "resolved"
"""Subdirectory of `configs/` holding the resolved per-component JSON dumps."""


class RunOutput(Protocol):
    """A config that owns a run directory: `output_dir / run.dir`."""

    output_dir: Path
    run: RunConfig


def output_path(config: RunOutput) -> Path:
    """Where this run writes: `output_dir / run.dir` — the same grouping convention as
    training. The run directory defaults to the auto-generated run name
    (`<env>--<model>--<harness>--<short-id>`)."""
    assert config.run.dir is not None
    return config.output_dir / config.run.dir


def write_config(
    config: BaseModel, results_dir: Path, filename: str = "eval.json"
) -> Path:
    """Write the run's resolved config to `configs/resolved/<filename>` (re-readable via
    `@ <path>`); return its path. The full model dump is written, nulls included, so the
    file round-trips exactly."""
    config_dir = results_dir / CONFIG_DIR / RESOLVED_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / filename
    config_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))
    return config_path


def write_launch_toml(results_dir: Path, name: str = "eval") -> None:
    """Copy the launch `@` TOML file(s) verbatim to `configs/<name>.toml`."""
    import sys

    argv = sys.argv[1:]
    paths = []
    for i, arg in enumerate(argv):
        # root config references only: `@ file`; a `--flag @ file` / `--flag @file`
        # is a nested reference and belongs under its flag, not in the launch copy
        if (
            arg == "@"
            and i + 1 < len(argv)
            and (i == 0 or not argv[i - 1].startswith("--"))
        ):
            paths.append(Path(argv[i + 1]))
    tomls = [(p, p.read_text()) for p in paths if p.suffix == ".toml" and p.is_file()]
    if not tomls:
        return
    texts = (
        [text for _, text in tomls]
        if len(tomls) == 1
        else [f"# @ {p}\n{text}" for p, text in tomls]
    )
    config_dir = results_dir / CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{name}.toml").write_text("\n".join(texts))


def save_config(
    config: BaseModel, results_dir: Path, filename: str = "eval.json"
) -> None:
    """Set up the run's output dir: write the resolved config, copy the launch TOML,
    and start a fresh (empty) `traces.jsonl`. Call once up front, before episodes start
    landing."""
    write_config(config, results_dir, filename)
    write_launch_toml(results_dir, Path(filename).stem)
    (results_dir / TRACES_FILE).write_text(
        ""
    )  # fresh; appended to as rollouts complete
