import verifiers as vf
from harnesses import NeMoGymHarness, NeMoGymHarnessConfig
from tasksets import NeMoGymTaskset, NeMoGymTasksetConfig


NEMO_ENV = "example_single_tool_call"


class NeMoGymEnvConfig(vf.EnvConfig):
    taskset: NeMoGymTasksetConfig = NeMoGymTasksetConfig(nemo_env=NEMO_ENV)
    harness: NeMoGymHarnessConfig = NeMoGymHarnessConfig(nemo_env=NEMO_ENV)


def load_environment(config: NeMoGymEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=NeMoGymTaskset(config=config.taskset),
        harness=NeMoGymHarness(config=config.harness),
    )
