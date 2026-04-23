from tasksets import TerminalBench2TaskSet
from verifiers.envs.composable_env import NamedComposableEnv


class TerminalBench2Env(
    NamedComposableEnv,
    taskset=TerminalBench2TaskSet,
    default_harness_config={"agent": "openclaw"},
):
    """Terminal-Bench 2 environment composed from tasksets + harnesses."""


def load_environment(**env_args) -> TerminalBench2Env:
    return TerminalBench2Env(**env_args)
