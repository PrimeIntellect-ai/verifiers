"""On-disk output: traces.jsonl (one rollout record per line) + config.toml.

A `RolloutRecord` is the full data dump for one rollout — its trace(s) nested, written
verbatim, consumed by the platform (visualization) and prime-rl (training). config.toml
is the run's resolved EvalConfig, written in the same format the CLI reads
(`@ config.toml`), so a run is re-runnable from its own output. Aggregates (avg reward,
etc.) are cheap to recompute from results, so they aren't stored.

The runner writes `config.toml` once up front (`save_config`) and then appends each
record to `traces.jsonl` as its rollout completes (`append_record`), so a long run's
results are durable as they land rather than only at the end. Files written before the
record atom (one bare trace per line) are still readable: `read_records` sniffs the
line shape.
"""

import asyncio
import json
from pathlib import Path

import tomli_w
from pydantic import BaseModel, TypeAdapter

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import RolloutRecord, Trace, WireRecord
from verifiers.v1.utils.aio import run_shielded

TRACES_FILE = "traces.jsonl"
"""Filename a run's rollout records are written to (one JSON record per line)."""

CONFIG_FILE = "config.toml"
"""Filename a run's resolved config is written to (re-runnable via `@ config.toml`)."""


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
    config_path = results_dir / CONFIG_FILE
    config_path.write_text(toml)
    return config_path


def save_config(config: BaseModel, results_dir: Path) -> None:
    """Set up the run's output dir: write `config.toml` and start a fresh (empty)
    `traces.jsonl`. Call once up front, before records start landing."""
    write_config(config, results_dir)
    (results_dir / TRACES_FILE).write_text(
        ""
    )  # fresh; appended to as rollouts complete


def write_record(results_dir: Path, record: RolloutRecord) -> None:
    """Serialize and append one rollout record in the worker thread."""
    # Preserve fields declared by typed Trace subclasses nested in the record.
    data = TypeAdapter(type(record)).dump_json(record, exclude_none=True)
    with (results_dir / TRACES_FILE).open("ab") as f:
        f.write(data + b"\n")


def sniff_record(row: dict) -> bool:
    """Whether a parsed traces.jsonl row is a `RolloutRecord` (vs a pre-record bare
    trace, recognizable by its message graph)."""
    return "traces" in row and "nodes" not in row


def read_records(results_dir: Path, trace_type: type) -> list[RolloutRecord]:
    """Load a run's saved rollouts from `traces.jsonl` with traces typed as
    `trace_type` — the inverse of `write_record`. A pre-record line (one bare trace)
    is wrapped as a single-trace record, so both file generations read uniformly.
    Used by `replay` to re-score finished rollouts (pass the task's typed
    `Trace[...]`, or `Trace[WireTaskData, ...]` to read any taskset's file without
    importing it)."""
    trace_adapter = TypeAdapter(trace_type)
    records: list[RolloutRecord] = []
    with (results_dir / TRACES_FILE).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if sniff_record(row):
                # Validate the shell wire-typed (unknown task fields preserved), then
                # re-type the traces as the caller asked.
                record = WireRecord.model_validate({**row, "traces": []})
                record.traces = [
                    trace_adapter.validate_python(t) for t in row.get("traces") or []
                ]
                records.append(record)
            else:
                trace = trace_adapter.validate_python(row)
                records.append(RolloutRecord.of(trace))
    return records


async def append_record(
    results_dir: Path, record: RolloutRecord, lock: asyncio.Lock
) -> None:
    """Append one finished rollout record without blocking the event loop. The run's
    shared lock preserves whole-line ordering, and awaiting the worker preserves
    per-record durability."""

    async def persist() -> None:
        async with lock:
            await asyncio.to_thread(write_record, results_dir, record)

    # Run lock acquisition and the worker to completion even under cancellation, so
    # finalized records are never lost mid-write (`run_shielded` re-raises the cancellation).
    await run_shielded(persist())


async def append_trace(
    results_dir: Path, trace: Trace, lock: asyncio.Lock, env: str = ""
) -> None:
    """Append one finished trace as a single-agent rollout record — the writers that
    complete trace-at-a-time (eval runners, gepa, replay, the legacy bridge) all go
    through here."""
    await append_record(results_dir, RolloutRecord.of(trace, env=env), lock)
