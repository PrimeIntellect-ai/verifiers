from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset, HarborTasksetConfig


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    taskset_config = config
    if (
        "source" not in config.model_fields_set
        and "dataset" not in config.model_fields_set
    ):
        taskset_config = taskset_config.model_copy(
            update={"source": "package", "dataset": "opencode_harbor_v1"}
        )
    return HarborTaskset(config=taskset_config)


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)
