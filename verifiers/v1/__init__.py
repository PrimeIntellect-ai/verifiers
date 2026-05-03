"""v1 workspace for the next verifier environment architecture."""

from verifiers.decorators import cleanup, metric, reward, teardown

from .config import Config, HarnessConfig, TasksetConfig
from .env import Env
from .harness import Harness
from .utils.scoring_utils import (
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
from .toolset import MCPTool, Toolset
from .user import User

__all__ = [
    "Config",
    "Env",
    "Harness",
    "HarnessConfig",
    "MCPTool",
    "State",
    "Task",
    "Taskset",
    "TasksetConfig",
    "Toolset",
    "User",
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
