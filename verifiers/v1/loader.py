from verifiers.utils import env_utils

from .config import HarnessConfig, TasksetConfig
from .harness import Harness
from .taskset import Taskset
from .types import ConfigMap


def load_taskset(
    env_id: str,
    *,
    config: TasksetConfig | ConfigMap | None = None,
) -> Taskset:
    """Load a taskset by env id using the package load_taskset annotation."""
    return env_utils.load_taskset(env_id, config=config)


def load_harness(
    env_id: str,
    *,
    config: HarnessConfig | ConfigMap | None = None,
) -> Harness:
    """Load a harness by env id using the package load_harness annotation."""
    return env_utils.load_harness(env_id, config=config)
