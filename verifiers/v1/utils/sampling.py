"""Shared sampling: an optional fixed-seed shuffle, then an optional head-slice.

With `--shuffle`, a shuffle under the fixed `SEED` so the sampled subset is the *same*
every run (reproducible), then an optional slice to the first `limit`. Used by the paths
that sample plain index lists (the server eval path and the legacy bridge); tasksets
shuffle themselves (`Taskset.shuffle`, same default seed).
"""

import random
from collections.abc import Iterable
from typing import TypeVar

SEED = 0  # fixed so `--shuffle` samples the same items every run (reproducible)

T = TypeVar("T")


def sample(items: Iterable[T], shuffle: bool, limit: int | None = None) -> list[T]:
    """`items` optionally shuffled under the fixed seed, then optionally sliced to the first
    `limit`. Returns a new list; the input is left untouched."""
    items = list(items)
    if shuffle:
        random.Random(SEED).shuffle(items)
    return items if limit is None else items[:limit]
