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
from verifiers.v1.utils.trace_store import episode_lines, replace_episodes


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
    verdict = complete if complete is not None else (lambda episode: episode.ok)
    episodes: list[WireEpisode] = []
    lines: dict[str, bytes] = {}
    # Resolve episode revisions before applying completion: an earlier success
    # must never hide the latest failed revision. Preserve the exact wire row.
    for line in episode_lines(resume_dir):
        try:
            episode = WireEpisode.model_validate(from_json(line))
            if not verdict(episode):
                continue
        except Exception:  # noqa: BLE001, S112 - malformed plugin rows are owed again
            continue
        episodes.append(episode)
        lines[episode.id] = line
    plan = plan_rollouts(tasks, num_rollouts, episodes)
    replace_episodes(
        resume_dir, (lines[episode.id] for _, kept, _ in plan for episode in kept)
    )
    return plan
