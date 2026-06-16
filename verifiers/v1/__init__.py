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
from verifiers.v1.decorators import group_reward, metric, reward, stop
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
from verifiers.v1.errors import ModelError, ProgramError, RolloutError, ToolError
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.ids import EnvId, ensure_installed, env_name
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
from verifiers.v1.task import Resources, Task, WireTask
from verifiers.v1.taskset import Taskset, TasksetConfig, ToolsConfig
from verifiers.v1.tools import Tools, run_mcp_server
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import (
    Branch,
    Error,
    TimeSpan,
    Timing,
    Trace,
)
from verifiers.v1.types import (
    AssistantMessage,
    ContentPart,
    ImageUrlContentPart,
    ImageUrlSource,
    Message,
    MessageContent,
    Messages,
    Response,
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
from verifiers.v1.user import User

__all__ = [
    # types
    "EnvId",
    "ensure_installed",
    "env_name",
    "AssistantMessage",
    "ContentPart",
    "ImageUrlContentPart",
    "ImageUrlSource",
    "Message",
    "MessageContent",
    "Messages",
    "Response",
    "SamplingConfig",
    "StrictBaseModel",
    "SystemMessage",
    "TextContentPart",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "Usage",
    "UserMessage",
    # task / trace
    "Task",
    "WireTask",
    "Resources",
    "Trace",
    "MessageNode",
    "Branch",
    "TurnTokens",
    "Timing",
    "TimeSpan",
    "Error",
    # decorators
    "stop",
    "metric",
    "reward",
    "group_reward",
    # errors
    "RolloutError",
    "ModelError",
    "ToolError",
    "ProgramError",
    # clients
    "Client",
    "BaseClientConfig",
    "ClientConfig",
    "resolve_client",
    # taskset / harness / runtime / environment
    "Taskset",
    "TasksetConfig",
    "ToolsConfig",
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
    # mcp
    "Tools",
    "run_mcp_server",
    # user simulator
    "User",
]

# The library logs via stdlib logging (per-module `getLogger(__name__)`), but is
# silent until an app opts in: a NullHandler on the package root absorbs records
# so nothing is emitted (and no "no handler" warning) unless handlers are added.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())
