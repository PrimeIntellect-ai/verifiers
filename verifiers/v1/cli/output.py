"""On-disk output: traces.jsonl (one rollout episode per line) + config.toml.

A `Episode` is the full data dump for one rollout — its trace(s) nested, written
verbatim, consumed by the platform (visualization) and prime-rl (training). config.toml
is the run's resolved EvalConfig, written in the same format the CLI reads
(`@ config.toml`), so a run is re-runnable from its own output. Aggregates (avg reward,
etc.) are cheap to recompute from results, so they aren't stored.

The runner writes `config.toml` once up front (`save_config`) and then appends each
episode to `traces.jsonl` as its rollout completes (`append_episode`), so a long run's
results are durable as they land rather than only at the end. Files written before the
episode atom (one bare trace per line) are still readable: `read_episodes` sniffs the
line shape.
"""

import asyncio
import json
from pathlib import Path

import tomli_w
from pydantic import BaseModel, TypeAdapter

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import Episode, Trace, WireEpisode
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.install import env_name

TRACES_FILE = "traces.jsonl"
"""Filename a run's rollout episodes are written to (one JSON episode per line)."""

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
        # The env axis is part of the run's identity (the same compounding as
        # `EnvConfig.env_id`): a `best-of-n+gsm8k-v1` run must not share a parent
        # dir with a plain `gsm8k-v1` one.
        env = f"{env_name(config.env.id)}+{env}"
    # The harness leg of the name: every seat's resolved harness, distinct, in
    # role order — `default` for one plain seat, `default+direct` for a judge run.
    harness = "+".join(
        dict.fromkeys(h.name for h in config.env.seat_harnesses().values())
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
    `traces.jsonl`. Call once up front, before episodes start landing."""
    write_config(config, results_dir)
    (results_dir / TRACES_FILE).write_text(
        ""
    )  # fresh; appended to as rollouts complete


def write_episode(results_dir: Path, episode: Episode) -> None:
    """Serialize and append one rollout episode in the worker thread."""
    # Preserve fields declared by typed Trace subclasses nested in the episode.
    data = TypeAdapter(type(episode)).dump_json(episode, exclude_none=True)
    with (results_dir / TRACES_FILE).open("ab") as f:
        f.write(data + b"\n")


def sniff_episode(row: dict) -> bool:
    """Whether a parsed traces.jsonl row is an `Episode` (vs a pre-episode bare
    trace, recognizable by its message graph)."""
    return "traces" in row and "nodes" not in row


def read_episodes(results_dir: Path, trace_type: type) -> list[Episode]:
    """Load a run's saved rollouts from `traces.jsonl` with traces typed as
    `trace_type` — the inverse of `write_episode`. A pre-episode line (one bare trace)
    is wrapped as a single-trace episode, so both file generations read uniformly.
    Used by `replay` to re-score finished rollouts (pass the task's typed
    `Trace[...]`, or `Trace[WireTaskData, ...]` to read any taskset's file without
    importing it)."""
    trace_adapter = TypeAdapter(trace_type)
    episodes: list[Episode] = []
    with (results_dir / TRACES_FILE).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if sniff_episode(row):
                # Validate the shell wire-typed (unknown task fields preserved), then
                # re-type the traces as the caller asked.
                episode = WireEpisode.model_validate({**row, "traces": []})
                episode.traces = [
                    trace_adapter.validate_python(t) for t in row.get("traces") or []
                ]
                episodes.append(episode)
            else:
                trace = trace_adapter.validate_python(row)
                episodes.append(Episode.of(trace))
    return episodes


async def append_episode(
    results_dir: Path, episode: Episode, lock: asyncio.Lock
) -> None:
    """Append one finished rollout episode without blocking the event loop. The run's
    shared lock preserves whole-line ordering, and awaiting the worker preserves
    per-episode durability."""

    async def persist() -> None:
        async with lock:
            await asyncio.to_thread(write_episode, results_dir, episode)

    # Run lock acquisition and the worker to completion even under cancellation, so
    # finalized episodes are never lost mid-write (`run_shielded` re-raises the cancellation).
    await run_shielded(persist())


async def append_trace(
    results_dir: Path, trace: Trace, lock: asyncio.Lock, env: str = ""
) -> None:
    """Append one finished trace as a single-agent rollout episode — the writers that
    complete trace-at-a-time (eval runners, gepa, replay, the legacy bridge) all go
    through here."""
    await append_episode(results_dir, Episode.of(trace, env=env), lock)
