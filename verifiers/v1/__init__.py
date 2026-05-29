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
    EndpointConfig,
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
    SignalConfig,
)
from .env import Env, EnvConfig
from .harness import Harness, HarnessConfig
from .model import ModelConfig
from .program import Program, ProgramConfig, ProgramOptionMap, ProgramValue
from .sandbox import SandboxConfig
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
from .taskset import Taskset, TasksetConfig, discover_sibling_dir
from .toolset import MCPTool, MCPToolConfig, Toolset, ToolsetConfig, Toolsets
from .utils.endpoint_utils import Endpoint
from .types import (
    ConfigData,
    ConfigMap,
    GroupHandler,
    Handler,
    MutableConfigMap,
    Objects,
    PromptInput,
    SystemPrompt,
    TaskRow,
    TaskSplit,
    Tasks,
)
from .user import User, UserConfig

__all__ = [
    "ConfigData",
    "CallableConfig",
    "Config",
    "ConfigMap",
    "Env",
    "EnvConfig",
    "Endpoint",
    "EndpointConfig",
    "AssistantMessage",
    "GroupHandler",
    "Harness",
    "HarnessConfig",
    "Handler",
    "MutableConfigMap",
    "MCPTool",
    "MCPToolConfig",
    "Message",
    "Messages",
    "ModelConfig",
    "Objects",
    "Program",
    "ProgramConfig",
    "ProgramOptionMap",
    "ProgramValue",
    "PromptInput",
    "SandboxConfig",
    "SignalConfig",
    "State",
    "SystemPrompt",
    "Task",
    "TaskRow",
    "TaskSplit",
    "Tasks",
    "Taskset",
    "TasksetConfig",
    "SystemMessage",
    "TextMessage",
    "Toolset",
    "ToolsetConfig",
    "Toolsets",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
