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
    ClientConfig,
    Message,
    MessageContent,
    Messages,
    SystemMessage,
    TextMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from . import advantages
from .advantages import AdvantageConfig
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
from .lifecycle import EnvRun, Group
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
    DaytonaRuntimeConfig,
    DaytonaRuntimeProvider,
    DaytonaRuntime,
    DockerRuntimeConfig,
    DockerRuntimeProvider,
    DockerRuntime,
    ModalRuntimeConfig,
    ModalRuntimeProvider,
    ModalRuntime,
    PrimeRuntimeConfig,
    PrimeRuntimeProvider,
    PrimeRuntime,
    RuntimeConfig,
    RuntimeConfigValue,
    RuntimeProvider,
    Runtime,
    SubprocessRuntimeConfig,
    SubprocessRuntimeProvider,
    SubprocessRuntime,
    make_runtime_provider,
    resolve_runtime_config,
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
from .task import Resources, Task, TaskVisibility
from .taskset import Taskset, TasksetConfig, discover_sibling_dir
from .toolset import (
    Scope,
    ServerPlacement,
    ServerConfig,
    Toolset,
    ToolsetConfig,
    ToolsetConfigs,
    VisibilityConfig,
    resource,
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
from .user import User, UserConfig, user

__all__ = [
    "Config",
    "Env",
    "EnvConfig",
    "EnvRun",
    "Extras",
    "EndpointProtocol",
    "AssistantMessage",
    "ClientConfig",
    "Harness",
    "HarnessConfig",
    "Handler",
    "Group",
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
    "AdvantageConfig",
    "ProtocolRoute",
    "Context",
    "ModelClient",
    "ModelConfig",
    "RuntimeConfig",
    "RuntimeConfigValue",
    "RuntimeProvider",
    "Runtime",
    "Resources",
    "Scope",
    "ServerPlacement",
    "ServerConfig",
    "CommandResult",
    "DaytonaRuntimeConfig",
    "DaytonaRuntimeProvider",
    "DaytonaRuntime",
    "DockerRuntimeConfig",
    "DockerRuntimeProvider",
    "DockerRuntime",
    "ModalRuntimeConfig",
    "ModalRuntimeProvider",
    "ModalRuntime",
    "PrimeRuntimeConfig",
    "PrimeRuntimeProvider",
    "PrimeRuntime",
    "SubprocessRuntimeConfig",
    "SubprocessRuntimeProvider",
    "SubprocessRuntime",
    "PromptInput",
    "State",
    "SystemPrompt",
    "SystemPromptConfig",
    "SystemPromptStrategy",
    "Task",
    "TaskVisibility",
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
    "ToolsetConfigs",
    "ToolMessage",
    "Turn",
    "TurnTokens",
    "TurnUsage",
    "UserMessage",
    "User",
    "UserConfig",
    "VisibilityConfig",
    "advantages",
    "advantage",
    "cleanup",
    "discover_sibling_dir",
    "default_protocols",
    "metric",
    "make_runtime_provider",
    "resolve_runtime_config",
    "load_environment",
    "load_harness",
    "load_taskset",
    "reward",
    "setup",
    "stop",
    "teardown",
    "resource",
    "tool",
    "user",
    "update",
]


def __getattr__(name: str):
    if name in ("load_environment", "load_harness", "load_taskset"):
        module = importlib.import_module("verifiers.v1.loaders")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
