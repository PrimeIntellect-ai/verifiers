"""verifiers v1 — a clean-slate, heavily-typed reimplementation.

Public surface is re-exported here so environments can `import verifiers.v1 as vf`
and reach everything they need. Built up milestone by milestone.
"""

import logging as _logging

from pydantic_config import BaseConfig

from verifiers.v1.clients import (
    BaseClientConfig,
    Client,
    ClientConfig,
    RolloutContext,
    resolve_client,
)
from verifiers.v1.decorators import group_reward, metric, reward, stop, tool
from verifiers.v1.env import (
    ElasticPoolConfig,
    EnvConfig,
    EnvServerConfig,
    Environment,
    StaticPoolConfig,
    TimeoutConfig,
    pool_serve_kwargs,
)
from verifiers.v1.episode import Episode
from verifiers.v1.errors import (
    HarnessError,
    InterceptionError,
    ProgramError,
    ProviderError,
    RolloutError,
    ToolError,
    TunnelError,
)
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.loaders import (
    harness_config_type,
    import_harness,
    import_taskset,
    load_harness,
    load_taskset,
    task_type,
    taskset_config_type,
)
from verifiers.v1.retries import CallRetryConfig, RetryConfig, RolloutRetryConfig
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import (
    DockerConfig,
    PrimeConfig,
    ProgramResult,
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
)
from verifiers.v1.state import State, StateT
from verifiers.v1.task import Task, TaskResources, TaskTimeout, WireTask
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.mcp import (
    Toolset,
    ToolsetConfig,
    User,
    UserConfig,
)
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import (
    Branch,
    Error,
    TimeSpan,
    Timing,
    Trace,
    WireTrace,
)
from verifiers.v1.types import (
    AssistantMessage,
    ContentPart,
    EnvId,
    ImageUrlContentPart,
    ImageUrlSource,
    Message,
    MessageContent,
    Messages,
    Response,
    Sampling,
    SamplingConfig,
    StrictBaseModel,
    SystemMessage,
    TextContentPart,
    Tool,
    ToolCall,
    TurnTokens,
    ToolMessage,
    Usage,
    UserMessage,
)

__all__ = [
    # types
    "EnvId",
    "AssistantMessage",
    "ContentPart",
    "ImageUrlContentPart",
    "ImageUrlSource",
    "Message",
    "MessageContent",
    "Messages",
    "Response",
    "Sampling",
    "SamplingConfig",
    "StrictBaseModel",
    "SystemMessage",
    "TextContentPart",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "Usage",
    "UserMessage",
    # task / trace / state
    "Task",
    "WireTask",
    "TaskResources",
    "TaskTimeout",
    "Trace",
    "WireTrace",
    "State",
    "StateT",
    "MessageNode",
    "Branch",
    "TurnTokens",
    "Timing",
    "TimeSpan",
    "Error",
    # decorators
    "stop",
    "tool",
    "metric",
    "reward",
    "group_reward",
    # errors
    "RolloutError",
    "ProviderError",
    "HarnessError",
    "ToolError",
    "ProgramError",
    "InterceptionError",
    "TunnelError",
    # clients
    "Client",
    "BaseClientConfig",
    "ClientConfig",
    "resolve_client",
    # taskset / harness / runtime / environment
    "Taskset",
    "TasksetConfig",
    "BaseConfig",
    "Harness",
    "HarnessConfig",
    "RolloutContext",
    "Runtime",
    "RuntimeConfig",
    "ProgramResult",
    "SubprocessConfig",
    "DockerConfig",
    "PrimeConfig",
    "Environment",
    "EnvConfig",
    "EnvServerConfig",
    "StaticPoolConfig",
    "ElasticPoolConfig",
    "pool_serve_kwargs",
    "CallRetryConfig",
    "RetryConfig",
    "RolloutRetryConfig",
    "TimeoutConfig",
    "Episode",
    "Rollout",
    # loaders
    "import_taskset",
    "import_harness",
    "load_taskset",
    "load_harness",
    "task_type",
    "taskset_config_type",
    "harness_config_type",
    # mcp
    "Toolset",
    "ToolsetConfig",
    # user simulator
    "User",
    "UserConfig",
]

# The library logs via stdlib logging (per-module `getLogger(__name__)`), but is
# silent until an app opts in: a NullHandler on the package root absorbs records
# so nothing is emitted (and no "no handler" warning) unless handlers are added.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())
