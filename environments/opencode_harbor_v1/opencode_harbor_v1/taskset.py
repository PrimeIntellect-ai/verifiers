from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset, HarborTasksetConfig


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    taskset_config = config
    if taskset_config.dataset is None and taskset_config.bundle_package is None:
        taskset_config = taskset_config.model_copy(
            update={"bundle_package": "opencode_harbor_v1"}
        )
    return HarborTaskset(config=taskset_config)


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)
