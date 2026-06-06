from pathlib import Path

import verifiers.v1 as vf
from harnesses import ReplayHarness
from tasksets import ReplayTaskset, ReplayTasksetConfig


class SFTReplayTaskset(ReplayTaskset):
    data_dir = str(Path(__file__).parent / "data")


def load_taskset(config: ReplayTasksetConfig) -> SFTReplayTaskset:
    return SFTReplayTaskset(config=config)


def load_harness(config: vf.HarnessConfig) -> ReplayHarness:
    return ReplayHarness(config=config)
