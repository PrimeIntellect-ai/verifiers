"""Search judge TaskSet factories."""

from typing import Any

from verifiers.envs.experimental.composable import TaskSet


def make_search_judge_taskset(backend: str = "quest", **kwargs: Any) -> TaskSet:
    """Create a search-judge TaskSet from a backend name."""
    factories = {
        "quest": make_quest_taskset,
    }
    if backend not in factories:
        raise ValueError(
            f"Unknown search judge backend: {backend!r}. Available: {list(factories)}"
        )
    return factories[backend](**kwargs)


def make_quest_taskset(**kwargs: Any) -> TaskSet:
    """QUEST objective/open-ended judge TaskSet."""
    from verifiers.envs.experimental.composable.tasksets.search_judge.quest import (
        QuestTaskSet,
    )

    return QuestTaskSet(**kwargs)
