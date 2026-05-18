import verifiers as vf


OpenCodeHarborEnvConfig = vf.Env.config(
    taskset=vf.HarborTaskset,
    harness=vf.OpenCode,
)


def load_environment(
    config: OpenCodeHarborEnvConfig | None = None,
) -> vf.Env:
    return vf.Env.from_config(
        config,
        taskset=vf.HarborTaskset,
        harness=vf.OpenCode,
        env_config=OpenCodeHarborEnvConfig,
    )
