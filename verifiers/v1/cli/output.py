"""On-disk output: traces.jsonl (one rollout episode per line) + config.toml.

Each line is an `Episode` — the episode's standing (`id`/`env`/`errors`) inlined
next to its flat, self-contained traces — so an episode persists whole or not at all: a torn line is the
whole episode owed on resume, and a failure before any trace minted still leaves
its errors on disk. config.toml is the run's resolved config in the format the
CLI reads (`@ config.toml`), so a run is re-runnable from its own output. Lines
append as episodes complete, so results are durable mid-run. Files written
before the episode atom (one bare trace per line) are still readable:
`read_episodes` sniffs the line shape.
"""

import asyncio
import json
from pathlib import Path

import tomli_w
from pydantic import BaseModel, TypeAdapter

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.episode import Episode, WireEpisode
from verifiers.v1.trace import Trace
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
    env = taskset.name if taskset.id else "no-taskset"
    if taskset.id and config.env.id:
        # Same compounding as `EnvConfig.env_id`: a `best-of-n+gsm8k-v1` run must
        # not share a parent dir with a plain `gsm8k-v1` one.
        env = f"{env_name(config.env.id)}+{env}"
    # Every seat's resolved harness, distinct, in role order.
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
    """Whether a parsed traces.jsonl row is an `Episode` (vs a pre-episode
    bare trace, recognizable by its message graph)."""
    return "traces" in row and "nodes" not in row


def read_episodes(results_dir: Path, trace_type: type) -> list[Episode]:
    """Load a run's saved rollouts from `traces.jsonl` with traces typed as
    `trace_type` (`Trace[WireTaskData, ...]` reads any taskset's file without
    importing it). A pre-episode line (one bare trace) is wrapped as a single-trace
    record, so both file generations read uniformly."""
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
                record = WireEpisode.model_validate({**row, "traces": []})
                record.traces = [
                    trace_adapter.validate_python(t) for t in row.get("traces") or []
                ]
                episodes.append(record)
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
