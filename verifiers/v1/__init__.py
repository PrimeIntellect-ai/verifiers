"""Taskset/harness authoring API."""

from verifiers.decorators import (
    advantage,
    cleanup,
    metric,
    reward,
    setup,
    stop,
    teardown,
    update,
)

from .config import Config, HarnessConfig, TasksetConfig
from .env import Env
from .harness import Harness
from .packages.harnesses import CLIHarness, MiniSWEAgent, OpenCode, Pi, RLM
from .utils.scoring_utils import (
    add_metric,
    add_reward,
    add_advantage,
    build_signals,
    collect_signals,
    score_group,
    score_rollout,
)
from .state import State
from .task import Task
from .taskset import Taskset
from .packages.tasksets import (
    HarborTaskset,
    HarborTasksetConfig,
)
from .toolset import MCPTool, Toolset
from .user import User

__all__ = [
    "Config",
    "Env",
    "Harness",
    "HarnessConfig",
    "HarborTaskset",
    "HarborTasksetConfig",
    "CLIHarness",
    "MCPTool",
    "MiniSWEAgent",
    "OpenCode",
    "Pi",
    "RLM",
    "State",
    "Task",
    "Taskset",
    "TasksetConfig",
    "Toolset",
    "User",
    "add_metric",
    "add_reward",
    "add_advantage",
    "advantage",
    "build_signals",
    "cleanup",
    "collect_signals",
    "metric",
    "reward",
    "score_group",
    "score_rollout",
    "setup",
    "stop",
    "teardown",
    "update",
]
