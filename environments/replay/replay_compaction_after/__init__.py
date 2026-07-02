"""replay_compaction_after — resume from a compaction message.

Seed the model with the post-compaction context (``[system, user(notes)]``) and let it continue
solving; score with the original env's verifier (``config.inner``).
"""

from typing import ClassVar

from replay_common.base import BaseReplayHarness, BaseReplayTaskset


class CompactionAfterTaskset(BaseReplayTaskset):
    KIND: ClassVar[str] = "compaction_after"


class CompactionAfterHarness(BaseReplayHarness):
    KIND: ClassVar[str] = "compaction_after"


__all__ = ["CompactionAfterTaskset", "CompactionAfterHarness"]
