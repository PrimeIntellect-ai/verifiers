from pathlib import Path

import verifiers as vf
from harnesses import ReplayHarness, ReplayHarnessConfig
from tasksets import ReplayTaskset, ReplayTasksetConfig


class SFTReplayTaskset(ReplayTaskset):
    data_dir = str(Path(__file__).parent / "data")


def load_taskset(config: ReplayTasksetConfig) -> SFTReplayTaskset:
    return SFTReplayTaskset(config=config)


def load_harness(config: ReplayHarnessConfig) -> ReplayHarness:
    return ReplayHarness(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
