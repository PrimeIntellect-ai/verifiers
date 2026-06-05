"""verifiers nano — a clean-slate, heavily-typed reimplementation.

Public surface is re-exported here so environments can `import verifiers.nano as vf`
and reach everything they need. Built up milestone by milestone.
"""

from verifiers.nano.clients import Client
from verifiers.nano.decorators import cleanup, metric, reward, setup, stop
from verifiers.nano.environment import EnvConfig, Environment
from verifiers.nano.errors import ModelError, ProgramError, RolloutError, ToolError
from verifiers.nano.eval import ClientConfig, EvalConfig, resolve_client, run_eval
from verifiers.nano.harness import Harness, HarnessConfig, RolloutContext
from verifiers.nano.loaders import (
    import_env,
    load_environment,
    load_harness,
    load_taskset,
)
from verifiers.nano.output import EvalMetadata, save_results
from verifiers.nano.program import ProgramConfig, ProgramHarness
from verifiers.nano.scoring import score
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
from verifiers.nano.tools import Toolset
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
    # scoring
    "score",
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
    # taskset / harness / environment
    "Taskset",
    "TasksetConfig",
    "Harness",
    "HarnessConfig",
    "RolloutContext",
    "ProgramHarness",
    "ProgramConfig",
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
