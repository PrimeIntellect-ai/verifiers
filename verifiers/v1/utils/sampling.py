"""Shared task sampling: an optional fixed-seed shuffle, then an optional head-slice.

Every taskset entrypoint — eval, server eval, debug, validate, legacy eval, and GEPA — narrows
the loaded tasks (or their indices) the same way: with `--shuffle`, a shuffle under a fixed seed
so the sampled subset is the *same* every run (reproducible), then an optional slice to the
first `limit`. GEPA layers its disjoint train/val split on top of this shuffle (see
`verifiers.v1.gepa.dataset`).
"""

import random
from typing import TypeVar

_SHUFFLE_SEED = (
    0  # fixed so `--shuffle` samples the same items every run (reproducible)
)

T = TypeVar("T")


def sample(items: list[T], shuffle: bool, limit: int | None = None) -> list[T]:
    """`items` optionally shuffled under the fixed seed, then optionally sliced to the first
    `limit`. Returns a new list; the input is left untouched."""
    items = list(items)
    if shuffle:
        random.Random(_SHUFFLE_SEED).shuffle(items)
    return items if limit is None else items[:limit]
