"""Public v1 API."""

import logging as _logging

from pydantic_config import BaseConfig

from verifiers.v1.agent import Agent, ChatSession, Reply
from verifiers.v1.clients import (
    BaseClientConfig,
    Client,
    ClientConfig,
    EvalClientConfig,
    ModelContext,
    TrainClientConfig,
    resolve_client,
)
from verifiers.v1.decorators import metric, reward, stop, tool
from verifiers.v1.env import (
    AgentConfig,
    ElasticPoolConfig,
    EnvConfig,
    EnvParams,
    EnvServerConfig,
    Environment,
    Role,
    StaticPoolConfig,
    TimeoutConfig,
    pool_serve_kwargs,
)
from verifiers.v1.errors import (
    HarnessError,
    InterceptionError,
    ProviderError,
    RolloutError,
    SandboxError,
    TaskError,
    ToolsetError,
    TunnelError,
    UserError,
)
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.judge import (
    Judge,
    JudgeConfig,
    JudgeResponse,
    Judges,
    JudgeSamplingConfig,
    JudgeView,
)
from verifiers.v1.judges import (
    ReferenceJudge,
    ReferenceJudgeConfig,
    Criterion,
    RubricJudge,
    RubricJudgeConfig,
)
from verifiers.v1.loaders import (
    default_harness_id,
    env_params_type,
    environment_class,
    harness_config_type,
    import_environment,
    import_harness,
    import_judge,
    import_taskset,
    judge_config_type,
    load_environment,
    load_harness,
    load_judge,
    load_taskset,
    task_type,
    taskset_config_type,
)
from verifiers.v1.scoring import (
    compare_stdout_results as compare_stdout_results,
    extract_boxed_answer as extract_boxed_answer,
    parse_judge_choice as parse_judge_choice,
    parse_pytest_outcomes as parse_pytest_outcomes,
    read_answer_file_or_last_reply as read_answer_file_or_last_reply,
    verify_boxed_math_answer as verify_boxed_math_answer,
)
from verifiers.v1.retries import RetryConfig, RolloutRetryConfig
from verifiers.v1.runtimes import (
    DockerConfig,
    PrimeConfig,
    ProgramResult,
    Runtime,
    RuntimeConfig,
    RuntimeInfo,
    SubprocessConfig,
)
from verifiers.v1.state import State, StateT
from verifiers.v1.task import (
    Task,
    TaskConfig,
    TaskData,
    TaskResources,
    TaskTimeout,
    WireTaskData,
)
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.mcp import (
    Toolset,
    SharedToolsetConfig,
    ToolsetConfig,
)
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import (
    Branch,
    Error,
    RolloutRecord,
    TimeSpan,
    Timing,
    Trace,
    TraceTask,
    WireRecord,
    WireTrace,
)
from verifiers.v1.types import (
    AssistantMessage,
    ContentPart,
    ID,
    ImageUrlContentPart,
    ImageUrlSource,
    KeptTokens,
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
    "ID",
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
    "TaskData",
    "WireTaskData",
    "TaskResources",
    "TaskTimeout",
    "Trace",
    "TraceTask",
    "WireTrace",
    "RolloutRecord",
    "WireRecord",
    "State",
    "StateT",
    "MessageNode",
    "Branch",
    "TurnTokens",
    "KeptTokens",
    "Timing",
    "TimeSpan",
    "Error",
    # decorators
    "stop",
    "tool",
    "metric",
    "reward",
    # errors
    "RolloutError",
    "ProviderError",
    "HarnessError",
    "ToolsetError",
    "UserError",
    "SandboxError",
    "TaskError",
    "InterceptionError",
    "TunnelError",
    # clients
    "Client",
    "BaseClientConfig",
    "ClientConfig",
    "EvalClientConfig",
    "TrainClientConfig",
    "resolve_client",
    # taskset / harness / runtime / environment
    "Taskset",
    "TaskConfig",
    "TasksetConfig",
    "BaseConfig",
    "Harness",
    "HarnessConfig",
    "ModelContext",
    "Runtime",
    "RuntimeConfig",
    "RuntimeInfo",
    "ProgramResult",
    "SubprocessConfig",
    "DockerConfig",
    "PrimeConfig",
    "Environment",
    "EnvConfig",
    "EnvServerConfig",
    "EnvParams",
    "AgentConfig",
    "Role",
    "StaticPoolConfig",
    "ElasticPoolConfig",
    "pool_serve_kwargs",
    "RetryConfig",
    "RolloutRetryConfig",
    "TimeoutConfig",
    # agent
    "Agent",
    # loaders
    "import_taskset",
    "import_harness",
    "import_judge",
    "import_environment",
    "load_environment",
    "load_taskset",
    "load_harness",
    "load_judge",
    "environment_class",
    "task_type",
    "taskset_config_type",
    "harness_config_type",
    "judge_config_type",
    "env_params_type",
    "default_harness_id",
    # judge
    "Judge",
    "JudgeConfig",
    "Judges",
    "JudgeSamplingConfig",
    "JudgeResponse",
    "JudgeView",
    "ReferenceJudge",
    "ReferenceJudgeConfig",
    "RubricJudge",
    "RubricJudgeConfig",
    "Criterion",
    # scoring
    "compare_stdout_results",
    "extract_boxed_answer",
    "parse_judge_choice",
    "parse_pytest_outcomes",
    "read_answer_file_or_last_reply",
    "verify_boxed_math_answer",
    # mcp
    "Toolset",
    "SharedToolsetConfig",
    "ToolsetConfig",
    # the user channel
    "ChatSession",
    "Reply",
]

# The library logs via stdlib logging (per-module `getLogger(__name__)`), but is
# silent until an app opts in: a NullHandler on the package root absorbs records
# so nothing is emitted (and no "no handler" warning) unless handlers are added.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())
