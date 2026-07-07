"""On-disk output: results.jsonl + config.toml.

One record per line: a full trace (single-agent eval), or a full instance record with its
traces nested (a topology eval) — either way durable as each unit's rewards become final.

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
from pathlib import Path

import tomli_w
from pydantic import BaseModel, TypeAdapter

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.topology import AgentGraph
from verifiers.v1.trace import Trace
from verifiers.v1.utils.aio import run_shielded


def output_path(config: EvalConfig) -> Path:
    """Where this run writes: `outputs/<taskset>--<model>--<harness>/<uuid>` — a topology
    run uses `outputs/<topology>--<model>/<uuid>` (its agents bind their own harnesses) —
    or the explicit `--output-dir`. The per-run `uuid` leaf means runs never overwrite
    each other."""
    if config.output_dir is not None:
        return config.output_dir
    if config.topology is not None:
        name = f"{config.topology.name}--{config.model.replace('/', '--')}"
    else:
        name = f"{config.taskset.name}--{config.model.replace('/', '--')}--{config.harness.name}"
    return Path("outputs") / name / config.uuid


def write_config(config: BaseModel, results_dir: Path) -> Path:
    """Write the run's resolved `config.toml` (re-readable via `@ config.toml`); return its
    path. mode="json" makes values TOML-friendly (Path -> str, etc.); exclude_none drops the
    nulls TOML can't represent."""
    results_dir.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json", exclude_none=True)
    if getattr(config, "topology", None) is not None:
        # A topology run never consults the taskset × harness pair; persisting their
        # (default) sections would make the saved config claim knobs the run ignored —
        # and trip the harness-with-topology guard on an `eval @ config.toml` re-run.
        data.pop("taskset", None)
        data.pop("harness", None)
    toml = tomli_w.dumps(data)
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


def write_graph(results_dir: Path, graph: AgentGraph) -> None:
    """Serialize and append one instance record (traces nested) in the worker thread."""
    with (results_dir / "results.jsonl").open("ab") as f:
        f.write(json.dumps(graph.to_record()).encode() + b"\n")


def read_traces(results_dir: Path, trace_type: type) -> list[Trace]:
    """Load a run's saved traces from `results.jsonl`, typed as `trace_type` — the inverse of
    `write_trace`. Used by `replay` to re-score finished rollouts."""
    adapter = TypeAdapter(trace_type)
    traces: list[Trace] = []
    with (results_dir / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(adapter.validate_python(json.loads(line)))
    return traces


async def append_graph(
    results_dir: Path, graph: AgentGraph, lock: asyncio.Lock
) -> None:
    """Append one finished topology instance without blocking the event loop — the
    topology counterpart of `append_trace`: one record per *instance*, traces nested
    (an instance's rewards are only final once the whole instance is done)."""

    async def persist() -> None:
        async with lock:
            await asyncio.to_thread(write_graph, results_dir, graph)

    # Run lock acquisition and the worker to completion even under cancellation, so
    # finalized graphs are never lost mid-write (`run_shielded` re-raises the cancellation).
    await run_shielded(persist())


async def append_trace(results_dir: Path, trace: Trace, lock: asyncio.Lock) -> None:
    """Append one finished trace without blocking the event loop. The run's shared lock
    preserves whole-line ordering, and awaiting the worker preserves per-trace durability."""

    async def persist() -> None:
        async with lock:
            await asyncio.to_thread(write_trace, results_dir, trace)

    # Run lock acquisition and the worker to completion even under cancellation, so
    # finalized traces are never lost mid-write (`run_shielded` re-raises the cancellation).
    await run_shielded(persist())
