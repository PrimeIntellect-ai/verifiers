"""On-disk output: config.toml + results.jsonl (one full trace per line).

The trace is the full data dump — written verbatim, consumed by the platform
(visualization) and prime-rl (training). config.toml is the run's resolved EvalConfig,
written in the same format the CLI reads (`@ config.toml`), so a run is re-runnable from
its own output. Aggregates (avg reward, etc.) are cheap to recompute from results, so
they aren't stored.

The runner writes `config.toml` once up front (`save_config`) and then appends each
trace to `results.jsonl` as it completes (`append_trace`), so a long run's results are
durable as they land rather than only at the end.
"""

import asyncio
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import TypeAdapter

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import Trace


@dataclass(frozen=True)
class InvalidResultLine:
    line: int
    reason: str


@dataclass(frozen=True)
class EvalRunArtifacts:
    """The native on-disk run contract consumed by Prime and other hosts."""

    config: dict[str, Any]
    results: list[dict[str, Any]]
    invalid_results: list[InvalidResultLine]


def read_config(results_dir: Path) -> dict[str, Any]:
    """Read a native run's resolved config without importing its plugins."""
    path = results_dir / "config.toml"
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Invalid Verifiers eval config: {path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Invalid Verifiers eval config: {path}")
    return config


def read_results(path: Path) -> tuple[list[dict[str, Any]], list[InvalidResultLine]]:
    """Read JSONL results while reporting incomplete or invalid records to the caller."""
    results: list[dict[str, Any]] = []
    invalid: list[InvalidResultLine] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                invalid.append(InvalidResultLine(line_number, "invalid JSON"))
                continue
            if not isinstance(result, dict):
                invalid.append(
                    InvalidResultLine(
                        line_number, f"expected object, got {type(result).__name__}"
                    )
                )
                continue
            results.append(result)
    return results, invalid


def read_artifacts(results_dir: Path) -> EvalRunArtifacts:
    """Read the complete native run contract from a directory."""
    results, invalid = read_results(results_dir / "results.jsonl")
    return EvalRunArtifacts(
        config=read_config(results_dir),
        results=results,
        invalid_results=invalid,
    )


def output_path(config: EvalConfig) -> Path:
    """Where this run writes: `outputs/<taskset>--<model>--<harness>/<uuid>` (or the explicit
    `--output-dir`). The per-run `uuid` leaf means runs never overwrite each other."""
    if config.output_dir is not None:
        return config.output_dir
    if config.is_legacy:
        from verifiers.v1.utils.install import env_name

        assert config.id is not None
        name = f"{env_name(config.id)}--{config.model.replace('/', '--')}--legacy"
        return Path("outputs") / name / config.uuid
    name = f"{config.taskset.name}--{config.model.replace('/', '--')}--{config.harness.name}"
    return Path("outputs") / name / config.uuid


def write_config(config: EvalConfig, results_dir: Path) -> Path:
    """Write the run's resolved `config.toml` (re-readable via `@ config.toml`); return its
    path. mode="json" makes values TOML-friendly (Path -> str, etc.); exclude_none drops the
    nulls TOML can't represent."""
    results_dir.mkdir(parents=True, exist_ok=True)
    toml = tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    config_path = results_dir / "config.toml"
    config_path.write_text(toml)
    return config_path


def save_config(config: EvalConfig, results_dir: Path) -> None:
    """Set up the run's output dir: write `config.toml` and start a fresh (empty)
    `results.jsonl`. Call once up front, before traces start landing."""
    write_config(config, results_dir)
    (results_dir / "results.jsonl").write_text(
        ""
    )  # fresh; appended to as traces complete


def write_trace(results_dir: Path, trace: Trace) -> None:
    """Serialize and append one trace in the worker thread."""
    data = TypeAdapter(type(trace)).dump_json(trace, exclude_none=True)
    with (results_dir / "results.jsonl").open("ab") as f:
        f.write(data + b"\n")


async def append_trace(results_dir: Path, trace: Trace, lock: asyncio.Lock) -> None:
    """Append one finished trace without blocking the event loop. The run's shared lock
    preserves whole-line ordering, and awaiting the worker preserves per-trace durability."""

    async def persist() -> None:
        async with lock:
            await asyncio.to_thread(write_trace, results_dir, trace)

    # Shield lock acquisition and the worker so finalized traces survive cancellation.
    persist_task = asyncio.create_task(persist())
    try:
        await asyncio.shield(persist_task)
    except asyncio.CancelledError:
        await persist_task
        raise
