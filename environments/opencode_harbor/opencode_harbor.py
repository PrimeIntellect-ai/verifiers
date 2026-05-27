import verifiers as vf
from harnesses import OpenCodeConfig
from tasksets import HarborTasksetConfig


class OpenCodeHarborEnvConfig(vf.EnvConfig):
    taskset: HarborTasksetConfig = HarborTasksetConfig()
    harness: OpenCodeConfig = OpenCodeConfig()


def load_environment(config: OpenCodeHarborEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset("tasksets.harbor", config=config.taskset),
        harness=vf.load_harness("harnesses.opencode", config=config.harness),
    )
