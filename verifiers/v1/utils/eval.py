"""Rollout planning shared by in-process and served evaluations."""

from collections import defaultdict, deque
from collections.abc import Iterable

from verifiers.v1.episode import WireEpisode
from verifiers.v1.task import Task, TaskT


def plan_rollouts(
    tasks: Iterable[TaskT],
    num_rollouts: int,
    completed: Iterable[WireEpisode] = (),
) -> list[tuple[TaskT, list[WireEpisode], int]]:
    """Assign distinct accepted episodes to selected task occurrences by content hash.

    Each occurrence gets up to ``num_rollouts`` saved episodes and the count still
    missing. Call separately per evaluation source; callers own acceptance and I/O.
    """
    saved: dict[str, deque[WireEpisode]] = defaultdict(deque)
    seen: set[str] = set()
    for episode in completed:
        if episode.id in seen:
            continue
        seen.add(episode.id)
        key = episode.task.hash
        if key is None:
            key = Task(episode.task.data).hash
        saved[key].append(episode)

    plan = []
    for task in tasks:
        episodes = saved[task.hash]
        kept = [episodes.popleft() for _ in range(min(num_rollouts, len(episodes)))]
        plan.append((task, kept, num_rollouts - len(kept)))
    return plan
