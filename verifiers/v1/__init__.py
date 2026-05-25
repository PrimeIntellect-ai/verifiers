"""Taskset/harness authoring API."""

import importlib

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
from verifiers.types import (
    AssistantMessage,
    Message,
    Messages,
    SystemMessage,
    TextMessage,
    ToolMessage,
    UserMessage,
)
from verifiers.utils.message_utils import get_messages

from .config import (
    CallableConfig,
    Config,
    EnvConfig,
    HarnessConfig,
    MCPToolConfig,
    ProgramConfig,
    SandboxConfig,
    SignalConfig,
    TasksetConfig,
    ToolsetConfig,
    UserConfig,
)
from .env import Env
from .harness import Harness
from .packages.harnesses import (
    MiniSWEAgent,
    MiniSWEAgentConfig,
    OpenCode,
    OpenCodeConfig,
    Pi,
    PiConfig,
    RLM,
    RLMConfig,
    Terminus2,
    Terminus2Config,
)
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
from .taskset import Taskset, discover_sibling_dir
from .packages.tasksets import (
    HarborTaskset,
    HarborTasksetConfig,
)
from .toolset import MCPTool, Toolset
from .types import (
    ConfigData,
    ConfigMap,
    GroupHandler,
    Handler,
    MutableConfigMap,
    Objects,
    TaskRow,
    Tasks,
)
from .user import User

__all__ = [
    "ConfigData",
    "CallableConfig",
    "Config",
    "ConfigMap",
    "Env",
    "EnvConfig",
    "AssistantMessage",
    "GroupHandler",
    "Harness",
    "HarnessConfig",
    "HarborTaskset",
    "HarborTasksetConfig",
    "Handler",
    "MutableConfigMap",
    "MCPTool",
    "MCPToolConfig",
    "Message",
    "Messages",
    "MiniSWEAgent",
    "MiniSWEAgentConfig",
    "OpenCode",
    "OpenCodeConfig",
    "Objects",
    "Pi",
    "PiConfig",
    "ProgramConfig",
    "RLM",
    "RLMConfig",
    "Terminus2",
    "Terminus2Config",
    "TextArenaTaskset",
    "TextArenaTasksetConfig",
    "SandboxConfig",
    "SignalConfig",
    "State",
    "Task",
    "TaskRow",
    "Tasks",
    "Taskset",
    "TasksetConfig",
    "SystemMessage",
    "TextMessage",
    "Toolset",
    "ToolsetConfig",
    "ToolMessage",
    "User",
    "UserMessage",
    "UserConfig",
    "add_metric",
    "add_reward",
    "add_advantage",
    "advantage",
    "build_signals",
    "cleanup",
    "collect_signals",
    "discover_sibling_dir",
    "metric",
    "get_messages",
    "load_harness",
    "load_taskset",
    "reward",
    "score_group",
    "score_rollout",
    "setup",
    "stop",
    "teardown",
    "update",
]


def __getattr__(name: str):
    if name in ("load_harness", "load_taskset"):
        module = importlib.import_module("verifiers.utils.env_utils")
        return getattr(module, name)
    if name in ("TextArenaTaskset", "TextArenaTasksetConfig"):
        module = importlib.import_module("verifiers.v1.packages.tasksets.textarena")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
