"""Search TaskSet factories."""

from typing import Any

from verifiers.envs.experimental.composable import TaskSet


def make_search_taskset(backend: str = "quest", **kwargs: Any) -> TaskSet:
    """Create a search/research TaskSet from a backend name."""
    factories = {
        "openseeker": make_openseeker_taskset,
        "quest": make_quest_taskset,
    }
    if backend not in factories:
        raise ValueError(
            f"Unknown search backend: {backend!r}. Available: {list(factories)}"
        )
    return factories[backend](**kwargs)


def make_quest_taskset(**kwargs: Any) -> TaskSet:
    """QUEST objective deep-research TaskSet."""
    from verifiers.envs.experimental.composable.tasksets.search.quest import (
        QuestTaskSet,
    )

    return QuestTaskSet(**kwargs)


def make_openseeker_taskset(**kwargs: Any) -> TaskSet:
    """OpenSeeker v1 deep-search QA TaskSet."""
    from verifiers.envs.experimental.composable.tasksets.search.openseeker import (
        OpenSeekerTaskSet,
    )

    return OpenSeekerTaskSet(**kwargs)
