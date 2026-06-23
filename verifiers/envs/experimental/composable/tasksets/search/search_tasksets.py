"""Search TaskSet factories."""

from typing import Any

from verifiers.envs.experimental.composable import TaskSet


def make_search_taskset(backend: str = "quest", **kwargs: Any) -> TaskSet:
    """Create a search/research TaskSet from a backend name."""
    factories = {
        "openseeker": make_openseeker_taskset,
        "redsearcher": make_redsearcher_taskset,
        "s1_deepresearch": make_s1_deepresearch_taskset,
    }
    if backend not in factories:
        raise ValueError(
            f"Unknown search backend: {backend!r}. Available: {list(factories)}"
        )
    return factories[backend](**kwargs)


def make_openseeker_taskset(**kwargs: Any) -> TaskSet:
    """OpenSeeker v1 deep-search QA TaskSet."""
    from verifiers.envs.experimental.composable.tasksets.search.openseeker import (
        OpenSeekerTaskSet,
    )

    return OpenSeekerTaskSet(**kwargs)


def make_redsearcher_taskset(**kwargs: Any) -> TaskSet:
    """REDSearcher RL query-set deep-search TaskSet."""
    from verifiers.envs.experimental.composable.tasksets.search.redsearcher import (
        RedSearcherTaskSet,
    )

    return RedSearcherTaskSet(**kwargs)


def make_s1_deepresearch_taskset(**kwargs: Any) -> TaskSet:
    """S1 DeepResearch closed-ended multi-hop TaskSet."""
    from verifiers.envs.experimental.composable.tasksets.search.s1_deepresearch import (
        S1DeepResearchTaskSet,
    )

    return S1DeepResearchTaskSet(**kwargs)
