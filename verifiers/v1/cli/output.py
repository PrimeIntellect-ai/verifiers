"""On-disk output: traces.jsonl (one trace per line) + config.toml.

A trace is the atom: one JSON line each, appended the moment its episode
completes, so results are durable mid-run. Traces of one env-rollout link through
their `episode.id` stamp — a multi-agent episode is simply several consecutive
lines sharing it. config.toml is the run's resolved config in the format the CLI
reads (`@ config.toml`), so a run is re-runnable from its own output. Files
written before the episode stamp (bare traces without `episode`) still read fine:
each line is one trace either way.
"""

import asyncio
import json
from collections import defaultdict
from pathlib import Path

import tomli_w
from pydantic import BaseModel, TypeAdapter

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import Trace
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.install import env_name

TRACES_FILE = "traces.jsonl"
"""Filename a run's traces are written to (one JSON trace per line)."""

CONFIG_FILE = "config.toml"
"""Filename a run's resolved config is written to (re-runnable via `@ config.toml`)."""


def output_path(config: EvalConfig) -> Path:
    """Where this run writes: `outputs/<env>--<model>--<harness>/<uuid>` (or the explicit
    `--output-dir`). The per-run `uuid` leaf means runs never overwrite each other."""
    if config.output_dir is not None:
        return config.output_dir
    taskset = config.env.taskset
    env = taskset.name if taskset is not None else "no-taskset"
    if taskset is not None and taskset.id and config.env.id:
        # Same compounding as `EnvConfig.env_id`: a `best-of-n+gsm8k-v1` run must
        # not share a parent dir with a plain `gsm8k-v1` one.
        env = f"{env_name(config.env.id)}+{env}"
    # Every agent's resolved harness, distinct, in declaration order.
    harness = "+".join(
        dict.fromkeys(h.name for h in config.env.agent_harnesses().values())
    )
    name = f"{env}--{config.model.replace('/', '--')}--{harness or 'default'}"
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
    `traces.jsonl`. Call once up front, before traces start landing."""
    write_config(config, results_dir)
    (results_dir / TRACES_FILE).write_text(
        ""
    )  # fresh; appended to as rollouts complete


def write_traces(results_dir: Path, traces: list[Trace]) -> None:
    """Serialize and append traces (one line each) in the worker thread."""
    lines = [
        # Preserve fields declared by typed Trace subclasses.
        TypeAdapter(type(trace)).dump_json(trace, exclude_none=True) + b"\n"
        for trace in traces
    ]
    with (results_dir / TRACES_FILE).open("ab") as f:
        f.writelines(lines)


def read_traces(results_dir: Path, trace_type: type) -> list[Trace]:
    """Load a run's saved traces from `traces.jsonl` typed as `trace_type`
    (`Trace[WireTaskData, ...]` reads any taskset's file without importing it)."""
    trace_adapter = TypeAdapter(trace_type)
    traces: list[Trace] = []
    with (results_dir / TRACES_FILE).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(trace_adapter.validate_python(json.loads(line)))
    return traces


def group_episodes(traces: list[Trace]) -> list[list[Trace]]:
    """The episodes in a flat trace list, first-seen order, grouped by each
    trace's `episode.id` stamp; a stampless (pre-episode) trace is its own
    single-trace episode."""
    groups: dict[str, list[Trace]] = defaultdict(list)
    episodes: list[list[Trace]] = []
    for i, trace in enumerate(traces):
        key = trace.episode.id if trace.episode is not None else f"trace-{i}"
        if key not in groups:
            episodes.append(groups[key])
        groups[key].append(trace)
    return episodes


async def append_episode(
    results_dir: Path, traces: list[Trace], lock: asyncio.Lock
) -> None:
    """Append one finished episode's traces without blocking the event loop. The
    run's shared lock keeps an episode's lines consecutive and whole, and awaiting
    the worker preserves per-episode durability."""

    async def persist() -> None:
        async with lock:
            await asyncio.to_thread(write_traces, results_dir, traces)

    # Run lock acquisition and the worker to completion even under cancellation, so
    # finalized traces are never lost mid-write (`run_shielded` re-raises the cancellation).
    await run_shielded(persist())


async def append_trace(results_dir: Path, trace: Trace, lock: asyncio.Lock) -> None:
    """Append one finished trace — the writers that complete trace-at-a-time
    (gepa, replay, the legacy bridge) all go through here."""
    await append_episode(results_dir, [trace], lock)
