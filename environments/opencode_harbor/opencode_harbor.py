import verifiers as vf


class OpenCodeHarborEnvConfig(vf.EnvConfig):
    taskset: vf.HarborTasksetConfig = vf.HarborTasksetConfig()
    harness: vf.OpenCodeConfig = vf.OpenCodeConfig()


def load_environment(config: OpenCodeHarborEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.HarborTaskset(config=config.taskset),
        harness=vf.OpenCode(config=config.harness),
    )
