"""Resume an interrupted eval: reload its finished rollouts and run only the rest.

`--resume` re-enters the run dir the resolved config points at and writes back into
it. `load` keeps the good saved rollouts and re-runs what's owed: missing rollouts
(never written) and errored ones (dropped and redone).

A saved episode is matched to a selected task by its persisted `episode.task.hash`.
Tasks with identical data are interchangeable, a task whose data changed since the
interrupted run re-runs, and nothing depends on `data.idx`.
"""

from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from pydantic_core import from_json

from verifiers.v1.cli.output import TRACES_FILE
from verifiers.v1.episode import WireEpisode


def load(
    resume_dir: Path,
    selected_keys: list[str],
    num_rollouts: int,
    complete: Callable[[WireEpisode], bool] | None = None,
) -> tuple[list[WireEpisode], dict[str, int]]:
    """Load the good saved rollouts and diff them against the run's target: returns
    (kept episodes, rollouts owed per task key). `selected_keys` is one key per
    selected task (duplicates allowed — a key selected k times is owed up to
    `k * num_rollouts`; spread back over the tasks with `distribute`). `complete`
    is the keep-verdict (default `episode.ok`). Rewrites `traces.jsonl` to the
    kept rows via a temp file + atomic rename; a torn or malformed row is owed
    again, never a crash."""
    path = resume_dir / TRACES_FILE
    targets = {
        key: count * num_rollouts for key, count in Counter(selected_keys).items()
    }

    verdict = complete if complete is not None else (lambda episode: episode.ok)
    good: dict[str, list[tuple[bytes, WireEpisode]]] = defaultdict(list)
    if path.exists():
        with path.open("rb") as results:
            for line in results:
                if not line.strip():
                    continue
                try:
                    row = from_json(line)
                    if "traces" not in row:
                        continue
                    key = row["task"]["hash"]
                except (ValueError, KeyError, IndexError, TypeError):
                    # A torn final line (the run died mid-write) or a foreign shape
                    # is not a keepable rollout — it's owed again, never a crash.
                    continue
                if key not in targets or len(good[key]) >= targets[key]:
                    continue
                try:
                    episode = WireEpisode.model_validate(row)
                    if not verdict(episode):
                        continue
                # A malformed row from any task/episode plugin is owed again.
                except Exception:  # noqa: BLE001, S112
                    continue
                good[key].append(
                    (line if line.endswith(b"\n") else line + b"\n", episode)
                )
    keep: list[bytes] = []
    episodes: list[WireEpisode] = []
    owed: dict[str, int] = {}
    for key, target in targets.items():
        rows = good.get(key, [])
        keep.extend(line for line, _ in rows)
        episodes.extend(episode for _, episode in rows)
        if missing := target - len(rows):
            owed[key] = missing
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_bytes(b"".join(keep))
    tmp.replace(path)
    return episodes, owed
