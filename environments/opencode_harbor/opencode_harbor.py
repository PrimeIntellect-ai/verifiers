import verifiers as vf
from verifiers.v1.packages.harnesses import OpenCode, OpenCodeConfig
from verifiers.v1.packages.tasksets import HarborTaskset, HarborTasksetConfig


class OpenCodeHarborEnvConfig(vf.EnvConfig):
    taskset: HarborTasksetConfig = HarborTasksetConfig()
    harness: OpenCodeConfig = OpenCodeConfig()


def load_environment(config: OpenCodeHarborEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=HarborTaskset(config=config.taskset),
        harness=OpenCode(config=config.harness),
    )
