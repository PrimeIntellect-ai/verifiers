import verifiers as vf
from harnesses import OpenCodeConfig
from tasksets import HarborTasksetConfig


class OpenCodeHarborEnvConfig(vf.EnvConfig):
    taskset: HarborTasksetConfig = HarborTasksetConfig(bundle_package=__name__)
    harness: OpenCodeConfig = OpenCodeConfig()


def load_environment(config: OpenCodeHarborEnvConfig) -> vf.Env:
    taskset_config = config.taskset
    if taskset_config.dataset is None and taskset_config.bundle_package is None:
        taskset_config = taskset_config.model_copy(update={"bundle_package": __name__})
    return vf.Env(
        taskset=vf.load_taskset("tasksets.harbor", config=taskset_config),
        harness=vf.load_harness("harnesses.opencode", config=config.harness),
    )
