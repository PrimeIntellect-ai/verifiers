"""Resume an interrupted eval: re-run only the rollouts a previous run didn't finish.

A run writes `config.toml` + `traces.jsonl` into its output dir. `--resume <dir>` reloads
that config verbatim (so it takes no other flags) and writes back into the same dir, running
only the rollouts still owed: the *missing* ones (never written — the run was interrupted) and
the *errored* ones (written with an error). Good rollouts are kept; errored ones are dropped
and redone. A group-scored task is resumed a whole task at a time (its rollouts are scored
together), so any task that isn't fully complete is redone from scratch.
"""

import json
import tomllib
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from pydantic_core import from_json

from verifiers.v1.cli.output import CONFIG_FILE, TRACES_FILE, read_traces
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.state import State
from verifiers.v1.task import WireTaskData
from verifiers.v1.trace import Trace


def split_resume(argv: list[str]) -> tuple[Path | None, list[str]]:
    """Pull `--resume <dir>` / `--resume=<dir>` out of argv, returning (dir, the other args).
    The caller rejects any leftover args, since resume re-runs the saved config verbatim."""
    for i, arg in enumerate(argv):
        if arg == "--resume":
            if i + 1 >= len(argv):
                raise SystemExit(
                    "--resume needs an output dir: uv run eval --resume <dir>"
                )
            return Path(argv[i + 1]), argv[:i] + argv[i + 2 :]
        if arg.startswith("--resume="):
            return Path(arg.split("=", 1)[1]), argv[:i] + argv[i + 1 :]
    return None, argv


def load_resume_config(resume_dir: Path) -> EvalConfig:
    """Rebuild the run's `EvalConfig` from its saved `config.toml`, pointed back at its own
    output dir so the resumed rollouts append to the same `traces.jsonl`."""
    config_path = resume_dir / CONFIG_FILE
    if not config_path.exists():
        raise SystemExit(
            f"--resume: no config.toml in {resume_dir} - not an eval output dir"
        )
    config = EvalConfig.model_validate(tomllib.loads(config_path.read_text()))
    config.resume = resume_dir
    config.output_dir = resume_dir
    return config


def _read_results(results_path: Path) -> Iterator[tuple[int, int, bool]]:
    """Stream `(file reference, task idx, errored)` without retaining decoded traces."""
    if not results_path.exists():
        return
    with results_path.open("rb") as results:
        while True:
            offset = results.tell()
            line = results.readline()
            if not line:
                break
            if line.strip():
                try:
                    row = from_json(line)
                except ValueError:
                    row = json.loads(line)
                yield offset, row["task"]["idx"], bool(row.get("errors"))


def plan(
    resume_dir: Path, selected_idxs: list[int], num_rollouts: int, group_scored: bool
) -> tuple[list[int], dict[int, int]]:
    """Diff the saved results against the run's target (`num_rollouts` per selected task).
    Returns (byte offsets of rows to keep, rollouts owed per task idx). An errored trace is
    dropped and re-run; on a group-scored run (one task type per taskset, so run-wide) a
    task is kept only if fully complete, else its whole group is redone."""
    # Retain only the offsets resume can reuse; trace payloads stay on disk.
    selected = set(selected_idxs)
    by_idx: dict[int, list[int]] = defaultdict(list)
    for offset, idx, errored in _read_results(resume_dir / TRACES_FILE):
        if idx in selected and not errored and len(by_idx[idx]) < num_rollouts:
            by_idx[idx].append(offset)
    keep: list[int] = []
    owed: dict[int, int] = {}
    for idx in selected_idxs:
        good = by_idx.get(idx, [])
        if group_scored:
            if len(good) >= num_rollouts:
                keep.extend(good)
            else:
                owed[idx] = num_rollouts  # re-run the whole group; keep none of it
        else:
            keep.extend(good)
            missing = num_rollouts - len(good)
            if missing:
                owed[idx] = missing
    return keep, owed


def rewrite_results(resume_dir: Path, keep: list[int]) -> None:
    """Replace `traces.jsonl` with just the kept (good) traces; resumed rollouts append. Via a
    temp file + atomic rename, so an interrupted resume can't corrupt the prior good results."""
    path = resume_dir / TRACES_FILE
    tmp = path.with_suffix(".jsonl.tmp")
    if not keep:
        tmp.write_bytes(b"")
        tmp.replace(path)
        return
    # Re-read retained rows by offset while building the atomic replacement.
    with path.open("rb") as results, tmp.open("wb") as output:
        for offset in keep:
            results.seek(offset)
            raw = results.readline()
            output.write(raw)
            if not raw.endswith(b"\n"):
                output.write(b"\n")
    tmp.replace(path)


def load_kept(resume_dir: Path) -> list[Trace]:
    """Reload the kept (good) traces as finished `Trace`s, so a resumed run's live dashboard counts
    them toward the whole run (progress, reward, err, and the usage/time breakdown). Call *after*
    `rewrite_results`, which leaves only the kept rows on disk. `WireTaskData` reads any taskset's saved
    task without a runtime or its `Task` type (mirrors `replay`); these traces are display-only, so
    no task upgrade is needed."""
    return read_traces(resume_dir, Trace[WireTaskData, State])


def nothing_to_resume_msg(resume_dir: Path, num_tasks: int, num_rollouts: int) -> str:
    """The message shown (and then exit 0 - the run is already complete) when every selected
    rollout already completed without error."""
    return (
        f"nothing to resume in {resume_dir}: all {num_tasks}x{num_rollouts} rollouts "
        f"already completed without error"
    )
