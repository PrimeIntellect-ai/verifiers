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
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_core import from_json

from verifiers.v1.configs.eval import EvalConfig


@dataclass
class Baseline:
    """Scoring aggregates over a resume's kept (already-on-disk, non-errored) rollouts, so the
    dashboard reflects the whole run rather than only the resumed rollouts. `n` feeds the progress
    counter and the denominators; `reward_sum` feeds the headline reward; `comp_sum` holds
    per-component reward/metric sums keyed `source ("rewards"/"metrics") -> name` for the
    breakdown (each divided by the total `n`, matching the breakdown's missing-component-as-0
    convention). Token/time totals are deliberately omitted — they stay session-scoped, since kept
    rows weren't recomputed and so their resources weren't re-spent. Empty (`n == 0`) on a fresh,
    non-resumed run."""

    n: int = 0
    reward_sum: float = 0.0
    comp_sum: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, rewards: dict[str, float], metrics: dict[str, float]) -> None:
        """Fold one kept rollout's recorded `rewards`/`metrics` into the aggregates."""
        self.n += 1
        self.reward_sum += sum(rewards.values())
        for source, values in (("rewards", rewards), ("metrics", metrics)):
            sums = self.comp_sum.setdefault(source, {})
            for name, v in values.items():
                sums[name] = sums.get(name, 0.0) + v


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


def _read_results(
    results_path: Path,
) -> Iterator[tuple[int, int, bool, dict, dict]]:
    """Stream `(file offset, task idx, errored, rewards, metrics)` without retaining decoded
    traces (`rewards`/`metrics` are the row's recorded score dicts, used to seed the resume
    `Baseline`)."""
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
                yield (
                    offset,
                    row["task"]["idx"],
                    bool(row.get("errors")),
                    row.get("rewards") or {},
                    row.get("metrics") or {},
                )


def plan(
    resume_dir: Path, selected_idxs: list[int], num_rollouts: int, group: bool
) -> tuple[list[int], dict[int, int], Baseline]:
    """Diff the saved results against the run's target (`num_rollouts` per selected task).
    Returns (byte offsets of rows to keep, rollouts owed per task idx, scoring `Baseline` over
    the kept rows). An errored trace is dropped and re-run; a group-scored task is kept only if
    fully complete, else its whole group is redone."""
    # Retain only the offsets resume can reuse (with each kept row's scores, to seed the
    # baseline); trace payloads stay on disk.
    selected = set(selected_idxs)
    by_idx: dict[int, list[tuple[int, dict, dict]]] = defaultdict(list)
    for offset, idx, errored, rewards, metrics in _read_results(
        resume_dir / "results.jsonl"
    ):
        if idx in selected and not errored and len(by_idx[idx]) < num_rollouts:
            by_idx[idx].append((offset, rewards, metrics))
    keep: list[int] = []
    owed: dict[int, int] = {}
    baseline = Baseline()

    def keep_all(rows: list[tuple[int, dict, dict]]) -> None:
        for offset, rewards, metrics in rows:
            keep.append(offset)
            baseline.add(rewards, metrics)

    for idx in selected_idxs:
        good = by_idx.get(idx, [])
        if group:
            if len(good) >= num_rollouts:
                keep_all(good)
            else:
                owed[idx] = num_rollouts  # re-run the whole group; keep none of it
        else:
            keep_all(good)
            missing = num_rollouts - len(good)
            if missing:
                owed[idx] = missing
    return keep, owed, baseline


def rewrite_results(resume_dir: Path, keep: list[int]) -> None:
    """Replace `results.jsonl` with just the kept (good) traces; resumed rollouts append. Via a
    temp file + atomic rename, so an interrupted resume can't corrupt the prior good results."""
    path = resume_dir / "results.jsonl"
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


def nothing_to_resume_msg(resume_dir: Path, num_tasks: int, num_rollouts: int) -> str:
    """The message shown (and then exit 0 - the run is already complete) when every selected
    rollout already completed without error."""
    return (
        f"nothing to resume in {resume_dir}: all {num_tasks}x{num_rollouts} rollouts "
        f"already completed without error"
    )
