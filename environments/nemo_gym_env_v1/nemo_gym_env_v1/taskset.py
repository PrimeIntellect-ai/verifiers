from harnesses import NeMoGymHarness, NeMoGymHarnessConfig
from tasksets import NeMoGymTaskset, NeMoGymTasksetConfig


NEMO_ENV = "example_single_tool_call"


def load_taskset(config: NeMoGymTasksetConfig) -> NeMoGymTaskset:
    if "nemo_env" not in config.model_fields_set:
        config = config.model_copy(update={"nemo_env": NEMO_ENV})
    return NeMoGymTaskset(config=config)


def load_harness(config: NeMoGymHarnessConfig) -> NeMoGymHarness:
    if "nemo_env" not in config.model_fields_set:
        config = config.model_copy(update={"nemo_env": NEMO_ENV})
    return NeMoGymHarness(config=config)
