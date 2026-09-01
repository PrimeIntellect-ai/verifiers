"""On-disk output: rollout traces, resolved configs, and upload-redaction fingerprints.

Each line is an `Episode` — the episode's standing (`id`/`env`/`errors`) inlined
next to its flat, self-contained traces — so an episode persists whole or not at all: a torn line is the
whole episode owed on resume, and a failure before any trace minted still leaves
its errors on disk. The JSON file is the run's resolved config in the format the
CLI reads (`@ configs/<cli>.json`), so a run is re-runnable from its own output. Lines
append as episodes complete, so results are durable mid-run. Runtime credentials stay
out of the trace; a non-plaintext sidecar lets resumed uploads redact them if the agent
echoed one into review data.
"""

import asyncio
import json
import os
from functools import cache
from pathlib import Path

from prime_evals import SecretFingerprint, fingerprint_secret, secret_values
from pydantic import BaseModel, TypeAdapter

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.client import resolve_api_key, resolve_headers
from verifiers.v1.episode import EPISODE_EXCLUDE_FIELDS, EnvInfo, Episode, WireEpisode
from verifiers.v1.state import StateT
from verifiers.v1.task import DataT
from verifiers.v1.trace import AgentConfigT, Trace
from verifiers.v1.utils.aio import run_shielded

TRACES_FILE = "traces.jsonl"
"""Filename a run's rollout episodes are written to (one JSON episode per line)."""

UPLOAD_SECRET_FINGERPRINTS_FILE = "upload-secret-fingerprints.jsonl"
"""Non-plaintext lookup material for redacting generated capabilities after resume."""

CONFIG_DIR = "configs"
"""Directory inside a run dir holding its configs: the launch TOML copied verbatim to
`configs/<cli>.toml`, and the resolved config at `configs/resolved/<cli>.json`
(re-runnable via `@ <run-dir>/configs/resolved/<cli>.json`). Resolved configs are JSON,
not TOML: JSON keeps nulls, so explicit None settings round-trip exactly on re-parse."""

RESOLVED_DIR = "resolved"
"""Subdirectory of `configs/` holding the resolved per-component JSON dumps."""


def create_attempt_log_dir(run_dir: Path) -> Path:
    """Create `logs/attempt_<n>` for this launch attempt and atomically repoint the
    relative `logs/latest` symlink at it (prime-rl's log layout). Every launch —
    fresh or `--resume` — gets its own numbered log directory, so a resume never
    appends to an earlier attempt's logs."""
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    attempts = (
        int(p.name.removeprefix("attempt_"))
        for p in logs_dir.glob("attempt_*")
        if p.name.removeprefix("attempt_").isdigit()
    )
    attempt_dir = logs_dir / f"attempt_{1 + max(attempts, default=0)}"
    attempt_dir.mkdir()
    # Atomically repoint the relative `latest` symlink: create a temp link, then rename.
    tmp_link = logs_dir / f".{attempt_dir.name}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    os.symlink(attempt_dir.name, tmp_link)
    os.replace(tmp_link, logs_dir / "latest")
    return attempt_dir


def attempt_log_file(run_dir: Path) -> Path:
    """The current attempt's `eval.log`. The CLI creates the attempt dir once at
    startup; everyone after it — the runner's worker spawn, the dashboard's log
    tail — resolves through `logs/latest`, so the whole invocation shares one file.
    A direct `run_eval` call (no CLI) creates the first attempt itself."""
    latest = run_dir / "logs" / "latest"
    if not latest.exists():
        create_attempt_log_dir(run_dir)
    return latest / "eval.log"


def saved_config_path(run_dir: Path) -> Path | None:
    """The run's saved resolved config (`configs/resolved/<cli>.json`; legacy runs
    kept it at `configs/<cli>.json`), None if absent."""
    for config_dir in (run_dir / CONFIG_DIR / RESOLVED_DIR, run_dir / CONFIG_DIR):
        candidates = sorted(config_dir.glob("*.json")) if config_dir.is_dir() else []
        if candidates:
            return candidates[0]
    return None


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
    """Write the run's resolved config to `configs/resolved/<filename>` (re-readable via
    `@ <path>`); return its path. The full model dump is written, nulls included, so the
    file round-trips exactly."""
    config_dir = results_dir / CONFIG_DIR / RESOLVED_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / filename
    config_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))
    return config_path


def write_launch_toml(results_dir: Path, name: str = "eval") -> None:
    """Copy the launch `@` TOML file(s) verbatim to `configs/<name>.toml`."""
    import sys

    argv = sys.argv[1:]
    paths = []
    for i, arg in enumerate(argv):
        # root config references only: `@ file`; a `--flag @ file` / `--flag @file`
        # is a nested reference and belongs under its flag, not in the launch copy
        if (
            arg == "@"
            and i + 1 < len(argv)
            and (i == 0 or not argv[i - 1].startswith("--"))
        ):
            paths.append(Path(argv[i + 1]))
    tomls = [(p, p.read_text()) for p in paths if p.suffix == ".toml" and p.is_file()]
    if not tomls:
        return
    texts = (
        [toml[1] for toml in tomls]
        if len(tomls) == 1
        else [f"# @ {p}\n{text}" for p, text in tomls]
    )
    config_dir = results_dir / CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{name}.toml").write_text("\n".join(texts))


def save_config(
    config: BaseModel, results_dir: Path, filename: str = "eval.json"
) -> None:
    """Write run config and initialize fresh trace and upload-redaction output files."""
    write_config(config, results_dir, filename)
    write_launch_toml(results_dir, Path(filename).stem)
    (results_dir / TRACES_FILE).write_text(
        ""
    )  # fresh; appended to as rollouts complete
    (results_dir / UPLOAD_SECRET_FINGERPRINTS_FILE).write_text("")


def write_episode(
    results_dir: Path, episode: Episode[DataT, StateT, AgentConfigT]
) -> None:
    """Serialize and append one rollout episode in the worker thread."""
    agents = [trace.agent.config for trace in episode.traces]
    clients = [agent.client for agent in agents if agent.client is not None]
    sources = [
        agent.harness.resolved_env for agent in agents if agent.harness is not None
    ] + [resolve_headers(client) for client in clients]
    secrets = secret_values(
        *episode.upload_secrets,
        *(secret for trace in episode.traces for secret in trace.upload_secrets),
        *(resolve_api_key(client) for client in clients),
        secret_sources=sources,
    )
    fingerprint_record = {
        "episode_id": episode.id,
        "fingerprints": [fingerprint_secret(secret) for secret in secrets],
    }
    with (results_dir / UPLOAD_SECRET_FINGERPRINTS_FILE).open("a") as f:
        f.write(json.dumps(fingerprint_record, separators=(",", ":")) + "\n")
    # Preserve fields declared by typed Trace subclasses nested in the episode.
    data = type_adapter(type(episode)).dump_json(
        episode,
        exclude=EPISODE_EXCLUDE_FIELDS,
        exclude_none=True,
    )
    with (results_dir / TRACES_FILE).open("ab") as f:
        f.write(data + b"\n")


def read_upload_secret_fingerprints(
    results_dir: Path, episode_ids: set[str]
) -> tuple[SecretFingerprint, ...]:
    """Read the sidecar records needed by resumed episodes, failing if any are absent."""
    records = {}
    with (results_dir / UPLOAD_SECRET_FINGERPRINTS_FILE).open() as f:
        for line in f:
            record = json.loads(line)
            if record["episode_id"] in episode_ids:
                records[record["episode_id"]] = record["fingerprints"]
    if missing := episode_ids - records.keys():
        raise ValueError(
            f"missing upload secret fingerprints for {len(missing)} episode(s)"
        )
    return tuple(
        (length, digest)
        for fingerprints in records.values()
        for length, digest in fingerprints
    )


def read_episodes(results_dir: Path, trace_type: type) -> list[WireEpisode]:
    """Load a run's saved rollouts from `traces.jsonl` with traces typed as
    `trace_type` (`Trace[WireTaskData, ...]` reads any taskset's file without
    importing it)."""
    trace_adapter = type_adapter(trace_type)
    episodes: list[WireEpisode] = []
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
    results_dir: Path,
    episode: Episode[DataT, StateT, AgentConfigT],
    lock: asyncio.Lock,
) -> None:
    """Append one finished rollout episode without blocking the event loop. The run's
    shared lock preserves whole-line ordering. Callers shield the whole operation so
    cancellation cannot interrupt lock acquisition or the worker."""

    async with lock:
        await asyncio.to_thread(write_episode, results_dir, episode)


async def append_trace(
    results_dir: Path, trace: Trace, lock: asyncio.Lock, env: str = ""
) -> None:
    """Append one finished trace as a single-agent rollout episode — debug and replay,
    which complete trace-at-a-time, both go through here."""
    episode = Episode(
        env=EnvInfo(id=env),
        task=trace.task,
        traces=[trace],
        ok=trace.ok,
    )
    await run_shielded(append_episode(results_dir, episode, lock))
