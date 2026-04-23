from tasksets import SWEBenchProTaskSet
from verifiers.envs.composable_env import NamedComposableEnv


class SWEBenchProEnv(NamedComposableEnv, taskset=SWEBenchProTaskSet):
    """SWE-bench Pro environment composed from tasksets + harnesses."""


def load_environment(**env_args) -> SWEBenchProEnv:
    return SWEBenchProEnv(**env_args)
