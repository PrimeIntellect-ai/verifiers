"""Resume an interrupted eval: reload its finished rollouts and run only the rest.

`load` validates saved episodes, applies the environment's completion verdict, and
uses the shared rollout planner to retain only those matching the current tasks.
"""

from collections.abc import Callable
from pathlib import Path

from pydantic_core import from_json

from verifiers.v1.episode import WireEpisode
from verifiers.v1.task import TaskT
from verifiers.v1.utils.eval import plan_rollouts
from verifiers.v1.utils.trace_store import TRACES_FILE


def load(
    resume_dir: Path,
    tasks: list[TaskT],
    num_rollouts: int,
    complete: Callable[[WireEpisode], bool] | None = None,
) -> list[tuple[TaskT, list[WireEpisode], int]]:
    """Plan the remaining rollouts and atomically rewrite the stream to kept rows.

    ``complete`` is the keep-verdict (default ``episode.ok``). Torn or malformed
    rows are discarded; retained rows preserve their original serialized fields.
    """
    path = resume_dir / TRACES_FILE
    verdict = complete if complete is not None else (lambda episode: episode.ok)
    episodes: list[WireEpisode] = []
    lines: dict[str, bytes] = {}
    if path.exists():
        with path.open("rb") as results:
            for line in results:
                if not line.strip():
                    continue
                try:
                    row = from_json(line)
                    if "traces" not in row:
                        continue
                    episode = WireEpisode.model_validate(row)
                    if not verdict(episode):
                        continue
                # A malformed row from any task/episode plugin is owed again.
                except Exception:  # noqa: BLE001, S112
                    continue
                episodes.append(episode)
                lines.setdefault(
                    episode.id, line if line.endswith(b"\n") else line + b"\n"
                )
    plan = plan_rollouts(tasks, num_rollouts, episodes)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_bytes(
        b"".join(lines[episode.id] for _, kept, _ in plan for episode in kept)
    )
    tmp.replace(path)
    return plan
