"""Shared sampling: an optional fixed-seed shuffle, then an optional head-slice.

Every entrypoint narrows its items the same way: with `--shuffle`, a shuffle under a fixed
seed so the sampled subset is the *same* every run (reproducible), then an optional slice to
the first `limit`. Taskset entrypoints (eval, validate, debug, GEPA) go through
`Taskset.select`, which shares this shuffle; the server eval path and the legacy bridge
sample plain index lists here directly.
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
