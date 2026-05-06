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

from .config import (
    Config,
    EnvConfig,
    HarnessConfig,
    MCPToolConfig,
    SandboxConfig,
    TasksetConfig,
    ToolsetConfig,
    UserConfig,
)
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
    "EnvConfig",
    "Harness",
    "HarnessConfig",
    "HarborTaskset",
    "HarborTasksetConfig",
    "CLIHarness",
    "MCPTool",
    "MCPToolConfig",
    "MiniSWEAgent",
    "OpenCode",
    "Pi",
    "RLM",
    "SandboxConfig",
    "State",
    "Task",
    "Taskset",
    "TasksetConfig",
    "Toolset",
    "ToolsetConfig",
    "User",
    "UserConfig",
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
