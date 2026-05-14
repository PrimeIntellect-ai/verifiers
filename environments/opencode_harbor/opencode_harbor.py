import verifiers as vf


class OpenCodeHarborEnvConfig(vf.EnvConfig):
    taskset: vf.HarborTasksetConfig
    harness: vf.OpenCodeConfig


def load_taskset(config: vf.HarborTasksetConfig) -> vf.HarborTaskset:
    return vf.HarborTaskset(config=config)


def load_harness(config: vf.OpenCodeConfig) -> vf.OpenCode:
    return vf.OpenCode(config=config)


def load_environment(config: OpenCodeHarborEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config.taskset),
        harness=load_harness(config.harness),
    )
