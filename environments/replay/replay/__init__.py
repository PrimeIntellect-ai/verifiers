"""replay — resume old rollouts from tagged compaction points (a replay buffer).

Offline (``ReplayTaskset`` materializes tasks from the buffer) or online (``ReplayHarness``
samples the live buffer per rollout). Scoring reuses the original env's verifier.

Option A: compaction tags + snapshot refs are read from ``trace.info`` (see ``replay/selector.py``)."""

from replay.harness import ReplayHarness, ReplayHarnessConfig
from replay.taskset import ReplayTask, ReplayTaskset, ReplayTasksetConfig

__all__ = [
    "ReplayTaskset",
    "ReplayTasksetConfig",
    "ReplayTask",
    "ReplayHarness",
    "ReplayHarnessConfig",
]
