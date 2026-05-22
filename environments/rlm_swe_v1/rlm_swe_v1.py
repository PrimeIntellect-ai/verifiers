import verifiers as vf

DEFAULT_REPO_PATH = "/testbed"
DEFAULT_RLM_TOOLS = ("bash", "edit")

RlmSweTasksetConfig = vf.SWETasksetConfig
R2ESWETaskset = vf.SWETaskset


def load_taskset(config: RlmSweTasksetConfig) -> R2ESWETaskset:
    return vf.SWETaskset(config=config)


def load_harness(config: vf.RLMConfig) -> vf.RLM:
    base_data = vf.RLMConfig(
        workdir=DEFAULT_REPO_PATH,
        rlm_tools=list(DEFAULT_RLM_TOOLS),
    ).model_dump()
    config = vf.RLMConfig.model_validate(
        {
            **base_data,
            **config.model_dump(exclude_unset=True, exclude_none=True),
        }
    )
    return vf.RLM(config=config)


class RlmSweEnvConfig(vf.EnvConfig):
    taskset: RlmSweTasksetConfig = RlmSweTasksetConfig()
    harness: vf.RLMConfig = vf.RLMConfig()


def load_environment(config: RlmSweEnvConfig) -> vf.Env:
    taskset = load_taskset(config=config.taskset)
    harness = load_harness(config=config.harness)
    return vf.Env(taskset=taskset, harness=harness)
