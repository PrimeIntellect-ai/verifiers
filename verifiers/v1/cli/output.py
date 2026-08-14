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
import hashlib
import json
from functools import cache
from pathlib import Path

import tomli_w
from pydantic import BaseModel, TypeAdapter

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.episode import Episode, WireEpisode
from verifiers.v1.trace import Trace
from verifiers.v1.utils.aio import run_shielded

TRACES_FILE = "traces.jsonl"
"""Filename a run's rollout episodes are written to (one JSON episode per line)."""

CONFIG_DIR = "configs"
"""Directory inside a run dir holding its resolved config (one `<cli>.toml`, re-runnable
via `@ <run-dir>/configs/<cli>.toml`)."""

DIGEST_PREFIX = "# digest: "
"""Comment line prepended to a saved config: the digest of the in-memory resolved config.
Comments are not config — this is provenance, letting `--resume` verify the invocation
against what actually ran, including explicit-None settings TOML cannot represent."""


def config_digest(config: BaseModel) -> str:
    """Canonical hash of a resolved config's full model dump (nulls included)."""
    dump = config.model_dump(mode="json")
    canonical = json.dumps(dump, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def saved_config_path(run_dir: Path) -> Path | None:
    """The run's saved resolved config (`configs/<cli>.toml`), None if absent."""
    candidates = (
        sorted((run_dir / CONFIG_DIR).glob("*.toml"))
        if (run_dir / CONFIG_DIR).is_dir()
        else []
    )
    return candidates[0] if candidates else None


def saved_config_digest(path: Path) -> str | None:
    """The digest recorded when the config was saved, None for a file without one."""
    first = path.read_text().split("\n", 1)[0]
    return (
        first.removeprefix(DIGEST_PREFIX) if first.startswith(DIGEST_PREFIX) else None
    )


# Compiling an adapter is the expensive part; run output reuses only a few model classes.
type_adapter = cache(TypeAdapter)


def output_path(config: EvalConfig) -> Path:
    """Where this run writes: `output_dir / run.dir` — the same grouping convention as
    training. The run directory defaults to the auto-generated run name
    (`<env>--<model>--<harness>--<short-id>`)."""
    assert config.run.dir is not None
    return config.output_dir / config.run.dir


def write_config(
    config: BaseModel, results_dir: Path, filename: str = "eval.toml"
) -> Path:
    """Write the run's resolved config to `configs/<filename>` (re-readable via
    `@ <path>`); return its path. mode="json" makes values TOML-friendly (Path -> str,
    etc.); exclude_none drops the nulls TOML can't represent — the prepended digest
    comment still covers them (see `DIGEST_PREFIX`)."""
    config_dir = results_dir / CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    toml = tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    config_path = config_dir / filename
    config_path.write_text(f"{DIGEST_PREFIX}{config_digest(config)}\n{toml}")
    return config_path


def save_config(
    config: BaseModel, results_dir: Path, filename: str = "eval.toml"
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


def sniff_episode(row: dict) -> bool:
    """Whether a parsed traces.jsonl row is an `Episode` (vs a pre-episode
    bare trace, recognizable by its message graph)."""
    return "traces" in row and "nodes" not in row


def read_episodes(results_dir: Path, trace_type: type) -> list[Episode]:
    """Load a run's saved rollouts from `traces.jsonl` with traces typed as
    `trace_type` (`Trace[WireTaskData, ...]` reads any taskset's file without
    importing it). A pre-episode line (one bare trace) is wrapped as a single-trace
    record, so both file generations read uniformly."""
    trace_adapter = type_adapter(trace_type)
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
    complete trace-at-a-time (eval runners, gepa, replay) all go
    through here."""
    await append_episode(results_dir, Episode.of(trace, env=env), lock)
