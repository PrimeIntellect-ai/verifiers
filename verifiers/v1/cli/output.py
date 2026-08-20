"""On-disk output: traces.jsonl (one rollout episode per line) + config.toml.

Each line is an `Episode` — the episode's standing (`id`/`env`/`errors`) inlined
next to its flat, self-contained traces — so an episode persists whole or not at all: a torn line is the
whole episode owed on resume, and a failure before any trace minted still leaves
its errors on disk. config.toml is the run's resolved config in the format the
CLI reads (`@ config.toml`), so a run is re-runnable from its own output. Lines
append as episodes complete, so results are durable mid-run. Files written
by this surface contain episodes only.
"""

import asyncio
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.episode import EnvInfo, Episode, WireEpisode
from verifiers.v1.trace import Trace
from verifiers.v1.utils.aio import run_shielded

TRACES_FILE = "traces.jsonl"
"""Filename a run's rollout episodes are written to (one JSON episode per line)."""

CONFIG_DIR = "configs"
"""Directory inside a run dir holding its resolved config (one `<cli>.json`, re-runnable
via `@ <run-dir>/configs/<cli>.json`). Resolved configs are JSON, not TOML: JSON keeps
nulls, so explicit None settings round-trip exactly on re-parse."""


def config_digest(config: BaseModel) -> str:
    """Canonical hash of a resolved config's full model dump (nulls included)."""
    dump = config.model_dump(mode="json")
    canonical = json.dumps(dump, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def saved_config_path(run_dir: Path) -> Path | None:
    """The run's saved resolved config (`configs/<cli>.json`), None if absent."""
    candidates = (
        sorted((run_dir / CONFIG_DIR).glob("*.json"))
        if (run_dir / CONFIG_DIR).is_dir()
        else []
    )
    return candidates[0] if candidates else None


# Compiling an adapter is the expensive part; run output reuses only a few model classes.
type_adapter = cache(TypeAdapter)


def output_path(config: EvalConfig) -> Path:
    """Where this run writes: `output_dir / run.dir` — the same grouping convention as
    training. The run directory defaults to the auto-generated run name
    (`<env>--<model>--<harness>--<short-id>`)."""
    assert config.run.dir is not None
    return config.output_dir / config.run.dir


def write_config(
    config: BaseModel, results_dir: Path, filename: str = "eval.json"
) -> Path:
    """Write the run's resolved config to `configs/<filename>` (re-readable via
    `@ <path>`); return its path. The full model dump is written, nulls included, so the
    file round-trips exactly."""
    config_dir = results_dir / CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / filename
    config_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))
    return config_path


def save_config(
    config: BaseModel, results_dir: Path, filename: str = "eval.json"
) -> None:
    """Set up the run's output dir: write the resolved config and start a fresh (empty)
    `traces.jsonl`. Call once up front, before episodes start landing."""
    write_config(config, results_dir, filename)
    (results_dir / TRACES_FILE).write_text(
        ""
    )  # fresh; appended to as rollouts complete


def write_episode(results_dir: Path, episode: Episode) -> None:
    """Serialize and append one rollout episode in the worker thread."""
    # Preserve fields declared by typed Trace subclasses nested in the episode.
    data = type_adapter(type(episode)).dump_json(episode, exclude_none=True)
    with (results_dir / TRACES_FILE).open("ab") as f:
        f.write(data + b"\n")


def read_episodes(results_dir: Path, trace_type: type) -> list[Episode[Any, Any, Any]]:
    """Load a run's saved rollouts from `traces.jsonl` with traces typed as
    `trace_type` (`Trace[WireTaskData, ...]` reads any taskset's file without
    importing it)."""
    trace_adapter = type_adapter(trace_type)
    episodes: list[Episode[Any, Any, Any]] = []
    with (results_dir / TRACES_FILE).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            record = WireEpisode.model_validate({**row, "traces": []})
            record.traces = [
                trace_adapter.validate_python(trace) for trace in row["traces"]
            ]
            episodes.append(record)
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
    complete trace-at-a-time (eval runners, gepa, replay) all go
    through here."""
    episode = Episode(
        env=EnvInfo(id=env),
        task=trace.task,
        traces=[trace],
        ok=trace.ok,
    )
    await append_episode(results_dir, episode, lock)
