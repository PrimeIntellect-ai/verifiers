"""replay_judge — the correctness-judge taskset.

Present a sampled rollout's transcript and ask "was this correct? yes/no"; grade the model's
verdict against the rollout's actual reward (``original_reward > judge_threshold``). A
self-supervised correctness label — no inner verifier or sandbox needed.
"""

from typing import ClassVar

from replay_common.base import BaseReplayHarness, BaseReplayTaskset


class JudgeTaskset(BaseReplayTaskset):
    KIND: ClassVar[str] = "judge"


class JudgeHarness(BaseReplayHarness):
    KIND: ClassVar[str] = "judge"


__all__ = ["JudgeTaskset", "JudgeHarness"]
