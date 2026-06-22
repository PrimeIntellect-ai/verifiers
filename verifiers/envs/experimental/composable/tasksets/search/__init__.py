"""Composable search/research tasksets."""

from .search_tasksets import (
    make_openseeker_taskset,
    make_quest_taskset,
    make_redsearcher_taskset,
    make_s1_deepresearch_taskset,
    make_search_taskset,
)

__all__ = [
    "make_openseeker_taskset",
    "make_quest_taskset",
    "make_redsearcher_taskset",
    "make_s1_deepresearch_taskset",
    "make_search_taskset",
]
