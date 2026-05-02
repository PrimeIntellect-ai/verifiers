"""v1 workspace for the next verifier environment architecture."""

from verifiers.decorators import cleanup, metric, reward, teardown

from .config import Config, HarnessConfig, TasksetConfig
from .env import Env
from .harness import Harness
from .scoring import (
    add_metric,
    add_reward,
    build_signals,
    collect_signals,
    score_group,
    score_rollout,
)
from .state import State
from .task import Task
from .taskset import Taskset
from .toolset import Toolset

__all__ = [
    "Config",
    "Env",
    "Harness",
    "HarnessConfig",
    "State",
    "Task",
    "Taskset",
    "TasksetConfig",
    "Toolset",
    "add_metric",
    "add_reward",
    "build_signals",
    "cleanup",
    "collect_signals",
    "metric",
    "reward",
    "score_group",
    "score_rollout",
    "teardown",
]
