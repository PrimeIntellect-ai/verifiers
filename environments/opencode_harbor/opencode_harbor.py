import verifiers.v1 as vf
from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset, HarborTasksetConfig


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    taskset_config = config
    if taskset_config.dataset is None and taskset_config.bundle_package is None:
        taskset_config = taskset_config.model_copy(update={"bundle_package": __name__})
    return HarborTaskset(config=taskset_config)


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
        runtime=config.runtime,
    )
