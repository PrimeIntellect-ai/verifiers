"""On-disk output: results.jsonl (one full trace per line) + config.toml.

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

import tomli_w
from pydantic import BaseModel, TypeAdapter

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import Trace


def output_path(config: EvalConfig) -> Path:
    """Where this run writes: `outputs/<taskset>--<model>--<harness>/<uuid>` (or the explicit
    `--output-dir`). The per-run `uuid` leaf means runs never overwrite each other."""
    if config.output_dir is not None:
        return config.output_dir
    name = f"{config.taskset.name}--{config.model.replace('/', '--')}--{config.harness.name}"
    return Path("outputs") / name / config.uuid


def write_config(config: BaseModel, results_dir: Path) -> Path:
    """Write the run's resolved `config.toml` (re-readable via `@ config.toml`); return its
    path. mode="json" makes values TOML-friendly (Path -> str, etc.); exclude_none drops the
    nulls TOML can't represent."""
    results_dir.mkdir(parents=True, exist_ok=True)
    toml = tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    config_path = results_dir / "config.toml"
    config_path.write_text(toml)
    return config_path


def save_config(config: BaseModel, results_dir: Path) -> None:
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
