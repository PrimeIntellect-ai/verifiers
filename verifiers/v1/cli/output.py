"""On-disk output: run.json + config.toml + results.jsonl (one full trace per line).

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
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import Trace

PROTOCOL_VERSION = 1
TRACE_SCHEMA_VERSION = 1
RUN_SCHEMA = "verifiers.eval-run/v1"


class RunInfo(BaseModel):
    """Stable identity and schema versions for one on-disk eval run."""

    model_config = ConfigDict(extra="forbid")

    artifact_schema: Literal["verifiers.eval-run/v1"] = Field(
        RUN_SCHEMA, alias="schema"
    )
    protocol_version: Literal[1] = PROTOCOL_VERSION
    trace_schema_version: Literal[1] = TRACE_SCHEMA_VERSION
    run_id: str


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


def read_run_info(results_dir: Path) -> RunInfo:
    return RunInfo.model_validate_json((results_dir / "run.json").read_text())


def write_run_info(results_dir: Path, run_id: str) -> Path:
    """Create the immutable identity record shared by process hosts and artifact readers."""
    results_dir.mkdir(parents=True, exist_ok=True)
    run_path = results_dir / "run.json"
    info = RunInfo(run_id=run_id)
    if run_path.exists():
        if read_run_info(results_dir) != info:
            raise ValueError(f"{run_path} belongs to a different eval run")
        return run_path
    run_path.write_text(info.model_dump_json(indent=2, by_alias=True) + "\n")
    return run_path


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
