import verifiers as vf
from harnesses import ReplayHarness, ReplayHarnessConfig
from tasksets import ReplayTaskset as BaseReplayTaskset
from tasksets import ReplayTasksetConfig


class ReplayTaskset(BaseReplayTaskset):
    pass


def load_taskset(config: ReplayTasksetConfig) -> ReplayTaskset:
    return ReplayTaskset(config=config)


def load_harness(config: ReplayHarnessConfig) -> ReplayHarness:
    return ReplayHarness(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
