"""replay_recheck — the "try again" taskset.

Re-roll a sampled rollout after appending a "check your work, fix if wrong" user turn; score
the corrected attempt with the original env's verifier (``config.inner``).
"""

from typing import ClassVar

from replay_common.base import BaseReplayHarness, BaseReplayTaskset


class RecheckTaskset(BaseReplayTaskset):
    KIND: ClassVar[str] = "recheck"


class RecheckHarness(BaseReplayHarness):
    KIND: ClassVar[str] = "recheck"


__all__ = ["RecheckTaskset", "RecheckHarness"]
