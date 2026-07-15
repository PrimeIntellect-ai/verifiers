"""Resume an interrupted eval: reload its finished rollouts and run only the rest.

A run writes `config.toml` + `traces.jsonl` into its output dir. `--resume <dir>` reloads
that config verbatim (so it takes no other flags) and writes back into the same dir. `load`
brings the good saved rollouts back into memory and re-runs only what's still owed: the
*missing* rollouts (never written — the run was interrupted) and the *errored* ones (written
with an error; dropped and redone). The loaded traces rejoin the run everywhere — counted,
displayed, pushed, and printed alongside this session's — so a resumed run picks up exactly
where the interrupted one stopped. A group-scored taskset is resumed a whole task at a time
(its rollouts are scored together), so any task that isn't fully complete is redone from
scratch.
"""

import json
import tomllib
from collections import defaultdict
from pathlib import Path

from pydantic_core import from_json

from verifiers.v1.cli.output import CONFIG_FILE, TRACES_FILE
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.rollout import Phase, Rollout
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace, WireTrace


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


class Finished(Rollout):
    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self.task = Task(trace.task.data)
        self.phase = Phase.DONE


def load(
    resume_dir: Path,
    selected_idxs: list[int],
    num_rollouts: int,
    group: bool,
    data_idx_to_pos: dict[int, int] | None = None,
) -> tuple[list[Trace], dict[int, int]]:
    """Load the good saved rollouts back into memory as finished traces and diff them against
    the run's target (`num_rollouts` per selected task): returns (the kept traces, rollouts
    owed per task idx). An errored trace is dropped and re-run; a group-scored task is kept
    only if fully complete, else its whole group is redone. Rewrites `traces.jsonl` to just
    the kept rows — verbatim, via a temp file + atomic rename, so an interrupted resume can't
    corrupt the prior good results — and the resumed rollouts then append. `WireTrace` reads
    any taskset's saved traces without importing it.

    For traces written before `info.task_idx` existed, `data_idx_to_pos` maps the saved
    `task.data.idx` to its current load position. If the mapping is absent, `data.idx` is
    compared directly against `selected_idxs` (the legacy behavior)."""
    path = resume_dir / TRACES_FILE
    selected = set(selected_idxs)
    good: dict[int, list[bytes]] = defaultdict(list)
    if path.exists():
        with path.open("rb") as results:
            for line in results:
                if not line.strip():
                    continue
                try:
                    row = from_json(line)
                except ValueError:
                    row = json.loads(line)
                idx = (row.get("info") or {}).get("task_idx")
                if idx is None:
                    data_idx = row["task"]["data"]["idx"]
                    if data_idx_to_pos is not None:
                        idx = data_idx_to_pos.get(data_idx)
                    else:
                        idx = data_idx
                if (
                    idx is not None
                    and idx in selected
                    and not row.get("errors")
                    and len(good[idx]) < num_rollouts
                ):
                    good[idx].append(line if line.endswith(b"\n") else line + b"\n")
    keep: list[bytes] = []
    owed: dict[int, int] = {}
    for idx in selected_idxs:
        rows = good.get(idx, [])
        if group and len(rows) < num_rollouts:
            owed[idx] = num_rollouts  # re-run the whole group; keep none of it
            continue
        keep.extend(rows)
        if missing := num_rollouts - len(rows):
            owed[idx] = missing
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_bytes(b"".join(keep))
    tmp.replace(path)
    return [WireTrace.model_validate_json(line) for line in keep], owed


def nothing_to_resume_msg(resume_dir: Path, num_tasks: int, num_rollouts: int) -> str:
    """The message shown (and then exit 0 - the run is already complete) when every selected
    rollout already completed without error."""
    return (
        f"nothing to resume in {resume_dir}: all {num_tasks}x{num_rollouts} rollouts "
        f"already completed without error"
    )
