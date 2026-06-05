"""verifiers v2 — a clean-slate, heavily-typed reimplementation.

Public surface is re-exported here so environments can `import verifiers.v2 as vf`
and reach everything they need. Built up milestone by milestone.
"""

from verifiers.v2.clients import Client
from verifiers.v2.decorators import cleanup, metric, reward, setup, stop
from verifiers.v2.environment import EnvConfig, Environment
from verifiers.v2.errors import ModelError, RolloutError, ToolError
from verifiers.v2.eval import ClientConfig, EvalConfig, resolve_client, run_eval
from verifiers.v2.harness import Harness, HarnessConfig, RolloutContext
from verifiers.v2.loaders import (
    import_env,
    load_environment,
    load_harness,
    load_taskset,
)
from verifiers.v2.output import EvalMetadata, save_results
from verifiers.v2.scoring import score
from verifiers.v2.task import Task
from verifiers.v2.taskset import Taskset, TasksetConfig
from verifiers.v2.transcript import (
    Error,
    TimeSpan,
    Timing,
    Transcript,
    Turn,
    TurnTokens,
)
from verifiers.v2.tools import Toolset
from verifiers.v2.types import (
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
from verifiers.v2.user import User, UserConfig

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
    # scoring
    "score",
    # errors
    "RolloutError",
    "ModelError",
    "ToolError",
    # clients
    "Client",
    "ClientConfig",
    "resolve_client",
    # tools / user
    "Toolset",
    "User",
    "UserConfig",
    # taskset / harness / environment
    "Taskset",
    "TasksetConfig",
    "Harness",
    "HarnessConfig",
    "RolloutContext",
    "Environment",
    "EnvConfig",
    # loaders
    "import_env",
    "load_environment",
    "load_taskset",
    "load_harness",
    # eval
    "EvalConfig",
    "run_eval",
    # output
    "EvalMetadata",
    "save_results",
]
