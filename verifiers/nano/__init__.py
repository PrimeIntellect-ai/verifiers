"""verifiers nano — a clean-slate, heavily-typed reimplementation.

Public surface is re-exported here so environments can `import verifiers.nano as vf`
and reach everything they need. Built up milestone by milestone.
"""

from verifiers.nano.agent import (
    Agent,
    AgentConfig,
    DefaultAgent,
    DefaultAgentConfig,
    RLMAgent,
    RLMAgentConfig,
    make_agent,
)
from verifiers.nano.clients import Client
from verifiers.nano.context import RolloutContext
from verifiers.nano.decorators import cleanup, metric, reward, setup, stop
from verifiers.nano.environment import EnvConfig, Environment
from verifiers.nano.errors import ModelError, ProgramError, RolloutError, ToolError
from verifiers.nano.eval import ClientConfig, EvalConfig, resolve_client, run_eval
from verifiers.nano.loaders import (
    import_env,
    load_agent,
    load_environment,
    load_taskset,
)
from verifiers.nano.runtime import (
    DockerConfig,
    PrimeConfig,
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
)
from verifiers.nano.output import EvalMetadata, save_results
from verifiers.nano.task import Task
from verifiers.nano.taskset import Taskset, TasksetConfig
from verifiers.nano.transcript import (
    Error,
    TimeSpan,
    Timing,
    Transcript,
    Turn,
    TurnTokens,
)
from verifiers.nano.toolset import Toolset
from verifiers.nano.types import (
    AssistantMessage,
    Message,
    Messages,
    Response,
    SamplingConfig,
    StrictBaseModel,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from verifiers.nano.user import User, UserConfig

__all__ = [
    # types
    "AssistantMessage",
    "Message",
    "Messages",
    "Response",
    "SamplingConfig",
    "StrictBaseModel",
    "SystemMessage",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "Usage",
    "UserMessage",
    # task / transcript
    "Task",
    "Transcript",
    "Turn",
    "TurnTokens",
    "Timing",
    "TimeSpan",
    "Error",
    # decorators
    "setup",
    "cleanup",
    "stop",
    "metric",
    "reward",
    # errors
    "RolloutError",
    "ModelError",
    "ToolError",
    "ProgramError",
    # clients
    "Client",
    "ClientConfig",
    "resolve_client",
    # tools / user
    "Toolset",
    "User",
    "UserConfig",
    # taskset / agent / runtime / environment
    "Taskset",
    "TasksetConfig",
    "Agent",
    "AgentConfig",
    "make_agent",
    "DefaultAgent",
    "DefaultAgentConfig",
    "RLMAgent",
    "RLMAgentConfig",
    "RolloutContext",
    "Runtime",
    "RuntimeConfig",
    "SubprocessConfig",
    "DockerConfig",
    "PrimeConfig",
    "Environment",
    "EnvConfig",
    # loaders
    "import_env",
    "load_environment",
    "load_taskset",
    "load_agent",
    # eval
    "EvalConfig",
    "run_eval",
    # output
    "EvalMetadata",
    "save_results",
]
