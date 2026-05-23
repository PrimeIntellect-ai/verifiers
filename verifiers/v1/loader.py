from verifiers.utils.env_utils import load_harness as _load_harness
from verifiers.utils.env_utils import load_taskset as _load_taskset

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
    return _load_taskset(env_id, config=config)


def load_harness(
    env_id: str,
    *,
    config: HarnessConfig | ConfigMap | None = None,
) -> Harness:
    """Load a harness by env id using the package load_harness annotation."""
    return _load_harness(env_id, config=config)
