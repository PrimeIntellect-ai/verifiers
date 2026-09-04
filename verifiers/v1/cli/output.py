"""On-disk output: traces.jsonl (one rollout episode per line) + configs/<cli>.json + summary.json.

Each line is an `Episode` — the episode's standing (`id`/`env`/`errors`) inlined
next to its flat, self-contained traces — so an episode persists whole or not at all: a torn line is the
whole episode owed on resume, and a failure before any trace minted still leaves
its errors on disk. The JSON file is the run's resolved config in the format the
CLI reads (`@ configs/<cli>.json`), so a run is re-runnable from its own output. Lines
append as episodes complete, so results are durable mid-run. Files written
by this surface contain episodes only. `summary.json` is how the completed run went,
read off those episodes once (`summarize`) so no consumer re-derives it.
"""

import asyncio
import json
import os
from collections import Counter
from collections.abc import Sequence
from functools import cache
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.episode import EnvInfo, Episode, WireEpisode
from verifiers.v1.state import StateT
from verifiers.v1.task import DataT, Task
from verifiers.v1.trace import AgentConfigT, Error, Trace
from verifiers.v1.utils.aio import run_shielded

TRACES_FILE = "traces.jsonl"
"""Filename a run's rollout episodes are written to (one JSON episode per line)."""

SUMMARY_FILE = "summary.json"
"""Filename a completed run's `Summary` is written to, beside `traces.jsonl`."""

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
    """The run's saved resolved config (`configs/resolved/<cli>.json`), None if absent."""
    config_dir = run_dir / CONFIG_DIR / RESOLVED_DIR
    candidates = sorted(config_dir.glob("*.json")) if config_dir.is_dir() else []
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
        [text for _, text in tomls]
        if len(tomls) == 1
        else [f"# @ {p}\n{text}" for p, text in tomls]
    )
    config_dir = results_dir / CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{name}.toml").write_text("\n".join(texts))


def save_config(
    config: BaseModel, results_dir: Path, filename: str = "eval.json"
) -> None:
    """Set up the run's output dir: write the resolved config, copy the launch TOML,
    and start a fresh (empty) `traces.jsonl`. Call once up front, before episodes start
    landing."""
    write_config(config, results_dir, filename)
    write_launch_toml(results_dir, Path(filename).stem)
    (results_dir / TRACES_FILE).write_text(
        ""
    )  # fresh; appended to as rollouts complete


def write_episode(
    results_dir: Path, episode: Episode[DataT, StateT, AgentConfigT]
) -> None:
    """Serialize and append one rollout episode in the worker thread."""
    # Preserve fields declared by typed Trace subclasses nested in the episode.
    data = type_adapter(type(episode)).dump_json(episode, exclude_none=True)
    with (results_dir / TRACES_FILE).open("ab") as f:
        f.write(data + b"\n")


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
    """Append one finished trace as a single-agent rollout episode — debug and replay,
    which complete trace-at-a-time, both go through here."""
    episode = Episode(
        env=EnvInfo(id=env),
        task=trace.task,
        traces=[trace],
        ok=trace.ok,
    )
    await append_episode(results_dir, episode, lock)


class TaskSummary(BaseModel):
    """One task's rollouts, counted as `Summary` counts the run's."""

    name: str | None
    """The task's `TaskData.name`, when it has one."""
    rollouts: int
    failed: int
    reward: float | None
    """Mean reward over the task's scored rollouts; None when none was scored."""


class Summary(BaseModel):
    """How a run went, read off its finished episodes. An episode failed iff it ended
    not ok. A rollout's reward is the mean `Trace.reward` over its scored policy
    traces — the `trainable` ones, or every trace when no seat is — where a trace is
    scored once any of its rewards landed (all None: scoring never ran); a rollout
    with no scored trace is unscored and sits outside every mean."""

    episodes: int
    failed: int
    errors: dict[str, int]
    """Failed episodes by the type of the error that failed them: a failed trace's
    last error, else the episode's own last error (a hook's) — the live errors
    `episode_should_retry` reads, since an ok trace's errors are history its per-agent
    retry recovered from. A failed episode that recorded no error is counted nowhere here."""
    reward: float | None
    """Mean reward over the run's scored rollouts; None when none was scored."""
    tasks: dict[str, TaskSummary]
    """Per task, keyed by `task.key` (its `hash` when unset), in first-seen order."""


def _task_key(episode: Episode) -> str:
    # Rows written before the key and hash were recorded hash their data, as resume does.
    return episode.task.key or episode.task.hash or Task(episode.task.data).hash


def _cause(episode: Episode) -> Error | None:
    """The error that failed `episode` (see `Summary.errors`)."""
    return next(
        (t.last_error for t in episode.traces if not t.ok and t.last_error),
        episode.last_error,
    )


def _rollout_reward(episode: Episode) -> float | None:
    """The episode's reward as one rollout (see `Summary`); None when unscored."""
    policy = [t for t in episode.traces if t.agent.trainable] or episode.traces
    rewards = [
        t.reward for t in policy if any(r is not None for r in t.rewards.values())
    ]
    return sum(rewards) / len(rewards) if rewards else None


def _mean_reward(episodes: Sequence[Episode]) -> float | None:
    rewards = [r for e in episodes if (r := _rollout_reward(e)) is not None]
    return sum(rewards) / len(rewards) if rewards else None


def summarize(episodes: Sequence[Episode]) -> Summary:
    """The run's `Summary` over its finished episodes."""
    by_task: dict[str, list[Episode]] = {}
    for episode in episodes:
        by_task.setdefault(_task_key(episode), []).append(episode)
    causes = [c for e in episodes if not e.ok and (c := _cause(e)) is not None]
    return Summary(
        episodes=len(episodes),
        failed=sum(not e.ok for e in episodes),
        errors=dict(Counter(c.type for c in causes)),
        reward=_mean_reward(episodes),
        tasks={
            key: TaskSummary(
                name=group[0].task.data.name,
                rollouts=len(group),
                failed=sum(not e.ok for e in group),
                reward=_mean_reward(group),
            )
            for key, group in by_task.items()
        },
    )


def write_summary(results_dir: Path, summary: Summary) -> Path:
    """Write the completed run's `summary.json` beside its traces; return its path. Nulls
    included: an unscored reward reads as `null`, never as a missing key."""
    path = results_dir / SUMMARY_FILE
    path.write_text(summary.model_dump_json(indent=2))
    return path
