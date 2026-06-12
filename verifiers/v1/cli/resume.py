"""Resume an interrupted eval: re-run only the rollouts a previous run didn't finish.

A run writes `config.toml` + `results.jsonl` into its output dir. `--resume <dir>` reloads
that config verbatim (so it takes no other flags) and writes back into the same dir, running
only the rollouts still owed: the *missing* ones (never written — the run was interrupted) and
the *errored* ones (written with an error). Good rollouts are kept; errored ones are dropped
and redone. A group-scored taskset is resumed a whole task at a time (its rollouts are scored
together), so any task that isn't fully complete is redone from scratch.
"""

import json
import tomllib
from collections import defaultdict
from pathlib import Path

from verifiers.v1.configs.eval import EvalConfig


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
    output dir so the resumed rollouts append to the same `results.jsonl`."""
    config_path = resume_dir / "config.toml"
    if not config_path.exists():
        raise SystemExit(
            f"--resume: no config.toml in {resume_dir} - not an eval output dir"
        )
    config = EvalConfig.model_validate(tomllib.loads(config_path.read_text()))
    config.resume = resume_dir
    config.output_dir = resume_dir
    return config


def _read_results(results_path: Path) -> list[dict]:
    """The previous run's traces as raw dicts. The on-disk dump carries computed fields a
    strict `Trace` can't re-validate, so we read the JSON directly — resume only needs each
    trace's `task.idx` and whether it `errors`ed."""
    if not results_path.exists():
        return []
    return [
        json.loads(line)
        for line in results_path.read_text().split("\n")
        if line.strip()
    ]


def plan(
    resume_dir: Path, selected_idxs: list[int], num_rollouts: int, group: bool
) -> tuple[list[dict], dict[int, int]]:
    """Diff the saved results against the run's target (`num_rollouts` per selected task).
    Returns (rows to keep in `results.jsonl`, rollouts owed per task idx). An errored trace is
    dropped and re-run; a group-scored task is kept only if fully complete, else its whole
    group is redone."""
    by_idx: dict[int, list[dict]] = defaultdict(list)
    for row in _read_results(resume_dir / "results.jsonl"):
        by_idx[row["task"]["idx"]].append(row)
    keep: list[dict] = []
    owed: dict[int, int] = {}
    for idx in selected_idxs:
        good = [row for row in by_idx.get(idx, []) if not row.get("errors")]
        if group:
            if len(good) >= num_rollouts:
                keep.extend(good[:num_rollouts])
            else:
                owed[idx] = num_rollouts  # re-run the whole group; keep none of it
        else:
            keep.extend(good[:num_rollouts])
            missing = num_rollouts - min(len(good), num_rollouts)
            if missing:
                owed[idx] = missing
    return keep, owed


def rewrite_results(resume_dir: Path, keep: list[dict]) -> None:
    """Replace `results.jsonl` with just the kept (good) traces; resumed rollouts append. Via a
    temp file + atomic rename, so an interrupted resume can't corrupt the prior good results."""
    path = resume_dir / "results.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in keep:
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)


def nothing_to_resume_msg(resume_dir: Path, num_tasks: int, num_rollouts: int) -> str:
    """The message shown (and then exit 0 - the run is already complete) when every selected
    rollout already completed without error."""
    return (
        f"nothing to resume in {resume_dir}: all {num_tasks}x{num_rollouts} rollouts "
        f"already completed without error"
    )
