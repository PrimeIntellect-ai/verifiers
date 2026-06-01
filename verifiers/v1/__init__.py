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
    ToolLike,
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
from .artifact import ArtifactConfig, Artifacts, ArtifactsConfig
from .harness import Harness, HarnessConfig
from .model import ModelConfig
from .program import ProgramConfig, ProgramValue
from .runtime import TrajectoryVisibility
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
from .toolset import (
    MCPTool,
    MCPToolConfig,
    Toolset,
    ToolsetConfig,
    Toolsets,
    VisibilityConfig,
)
from .utils.endpoint_utils import Endpoint
from .utils.binding_utils import BindingsConfig, ObjectsConfig
from .utils.prompt_utils import SystemPrompt, SystemPromptConfig, SystemPromptStrategy
from .types import (
    ConfigData,
    Handler,
    JsonData,
    Objects,
    PromptInput,
    TaskSplit,
    Tasks,
)
from .user import User, UserConfig

__all__ = [
    "BindingsConfig",
    "ArtifactConfig",
    "Artifacts",
    "ArtifactsConfig",
    "ConfigData",
    "CallableConfig",
    "Config",
    "Env",
    "EnvConfig",
    "Endpoint",
    "EndpointConfig",
    "AssistantMessage",
    "Harness",
    "HarnessConfig",
    "Handler",
    "JsonData",
    "MCPTool",
    "MCPToolConfig",
    "Message",
    "Messages",
    "ModelConfig",
    "Objects",
    "ObjectsConfig",
    "ProgramConfig",
    "ProgramValue",
    "PromptInput",
    "SandboxConfig",
    "SignalConfig",
    "State",
    "SystemPrompt",
    "SystemPromptConfig",
    "SystemPromptStrategy",
    "Task",
    "TaskSplit",
    "Tasks",
    "Taskset",
    "TasksetConfig",
    "SystemMessage",
    "TextMessage",
    "ToolLike",
    "Toolset",
    "ToolsetConfig",
    "Toolsets",
    "ToolMessage",
    "TrajectoryVisibility",
    "User",
    "UserMessage",
    "UserConfig",
    "VisibilityConfig",
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
