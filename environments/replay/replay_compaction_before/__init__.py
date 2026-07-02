"""replay_compaction_before — resume just before a compaction.

Seed the model with the pre-compaction context (the branch leaf the compaction summarized) so its
continuation *writes* the compaction itself, then keeps solving; score with the original env's
verifier (``config.inner``).
"""

from typing import ClassVar

from replay_common.base import BaseReplayHarness, BaseReplayTaskset


class CompactionBeforeTaskset(BaseReplayTaskset):
    KIND: ClassVar[str] = "compaction_before"


class CompactionBeforeHarness(BaseReplayHarness):
    KIND: ClassVar[str] = "compaction_before"


__all__ = ["CompactionBeforeTaskset", "CompactionBeforeHarness"]
