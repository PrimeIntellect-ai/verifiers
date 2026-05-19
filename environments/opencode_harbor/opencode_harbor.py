import verifiers as vf


class OpenCodeHarborEnvConfig(vf.EnvConfig):
    taskset: vf.HarborTasksetConfig = vf.HarborTasksetConfig()
    harness: vf.OpenCodeConfig = vf.OpenCodeConfig()


def load_environment(
    config: OpenCodeHarborEnvConfig | None = None,
) -> vf.Env:
    return vf.Env(config, taskset=vf.HarborTaskset, harness=vf.OpenCode)
