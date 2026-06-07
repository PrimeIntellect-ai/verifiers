"""Taskset/harness authoring API."""

import importlib

from .decorators import (
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
    MessageContent,
    Messages,
    SystemMessage,
    TextMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from verifiers.utils.message_utils import get_messages

from . import advantages
from .config import (
    Config,
)
from .env import Env, EnvConfig
from .harness import Harness, HarnessConfig
from .interception import (
    EndpointProtocol,
    InterceptedRequest,
    InterceptionServer,
    ProtocolRoute,
)
from .protocols import (
    AnthropicMessagesProtocol,
    OpenAIChatCompletionsProtocol,
    OpenAICompletionsProtocol,
    OpenAIResponsesProtocol,
    default_protocols,
)
from .mcp import MCPToolRegistry, ServerResponse
from .runtime import (
    CommandResult,
    DockerRuntimeConfig,
    DockerRuntimeProvider,
    DockerRuntime,
    LocalRuntimeConfig,
    LocalRuntimeProvider,
    LocalRuntime,
    PrimeRuntimeConfig,
    PrimeRuntimeProvider,
    PrimeRuntime,
    RuntimeConfig,
    RuntimeConfigValue,
    RuntimeProvider,
    Runtime,
    TrajectoryVisibility,
    make_runtime_provider,
    resolve_runtime_config,
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
from .state import (
    Extras,
    State,
    Timing,
    TimeSpan,
    Turn,
    TurnTokens,
    TurnUsage,
)
from .task import Task
from .taskset import Taskset, TasksetConfig, discover_sibling_dir
from .toolset import (
    Scope,
    ServerConfig,
    Toolset,
    ToolsetConfig,
    Toolsets,
    VisibilityConfig,
    tool,
)
from .utils.prompt_utils import SystemPrompt, SystemPromptConfig, SystemPromptStrategy
from .types import (
    Handler,
    JsonData,
    JsonValue,
    ModelClient,
    ModelConfig,
    PromptInput,
    Context,
    TaskSplit,
    Tasks,
)
from .user import User, UserConfig

__all__ = [
    "Config",
    "Env",
    "EnvConfig",
    "Extras",
    "EndpointProtocol",
    "AssistantMessage",
    "Harness",
    "HarnessConfig",
    "Handler",
    "InterceptedRequest",
    "InterceptionServer",
    "JsonData",
    "JsonValue",
    "MCPToolRegistry",
    "ServerResponse",
    "Message",
    "MessageContent",
    "Messages",
    "OpenAIChatCompletionsProtocol",
    "OpenAICompletionsProtocol",
    "OpenAIResponsesProtocol",
    "AnthropicMessagesProtocol",
    "ProtocolRoute",
    "Context",
    "ModelClient",
    "ModelConfig",
    "RuntimeConfig",
    "RuntimeConfigValue",
    "RuntimeProvider",
    "Runtime",
    "Scope",
    "ServerConfig",
    "CommandResult",
    "DockerRuntimeConfig",
    "DockerRuntimeProvider",
    "DockerRuntime",
    "LocalRuntimeConfig",
    "LocalRuntimeProvider",
    "LocalRuntime",
    "PrimeRuntimeConfig",
    "PrimeRuntimeProvider",
    "PrimeRuntime",
    "PromptInput",
    "State",
    "SystemPrompt",
    "SystemPromptConfig",
    "SystemPromptStrategy",
    "Task",
    "TaskSplit",
    "Tasks",
    "Taskset",
    "TasksetConfig",
    "TimeSpan",
    "Timing",
    "SystemMessage",
    "TextMessage",
    "ToolCall",
    "Toolset",
    "ToolsetConfig",
    "Toolsets",
    "ToolMessage",
    "TrajectoryVisibility",
    "Turn",
    "TurnTokens",
    "TurnUsage",
    "UserMessage",
    "User",
    "UserConfig",
    "VisibilityConfig",
    "add_metric",
    "add_reward",
    "add_advantage",
    "advantages",
    "advantage",
    "build_signals",
    "cleanup",
    "collect_signals",
    "discover_sibling_dir",
    "default_protocols",
    "metric",
    "make_runtime_provider",
    "resolve_runtime_config",
    "get_messages",
    "load_environment",
    "load_harness",
    "load_taskset",
    "reward",
    "score_group",
    "score_rollout",
    "setup",
    "stop",
    "teardown",
    "tool",
    "update",
]


def __getattr__(name: str):
    if name in ("load_environment", "load_harness", "load_taskset"):
        module = importlib.import_module("verifiers.v1.loaders")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
