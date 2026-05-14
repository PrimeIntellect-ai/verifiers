import verifiers as vf


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.HarborTaskset(config=config.taskset),
        harness=vf.OpenCode(config=config.harness),
    )
