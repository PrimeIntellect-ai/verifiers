import logging as _logging

from pydantic_config import BaseConfig

from verifiers.v1.acp import ACP
from verifiers.v1.agent import Agent, Agents, Interaction, Segment, make_agent
from verifiers.v1.clients import (
    BaseClientConfig,
    Client,
    ClientConfig,
    EvalClientConfig,
    ModelContext,
    TrainClientConfig,
    resolve_client,
)
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.cli.env import narrowed_env_annotation, resolve_env_field
from verifiers.v1.configs.env import EnvConfig, default_agent_harness
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.configs.judge import JudgeConfig, Judges
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.configs.serve import (
    ElasticPoolConfig,
    ServeConfig,
    StaticPoolConfig,
    pool_serve_kwargs,
)
from verifiers.v1.configs.task import (
    DecoratedFunctionConfig,
    FunctionConfig,
    RewardFunctionConfig,
    TaskConfig,
)
from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.env import Env
from verifiers.v1.envs.single_agent import SingleAgentEnv, SingleAgentEnvConfig
from verifiers.v1.episode import Episode, WireEpisode
from verifiers.v1.errors import (
    EnvError,
    HarnessError,
    InterceptionError,
    ProviderError,
    RolloutError,
    SandboxError,
    TaskError,
    ToolsetError,
    TunnelError,
)
from verifiers.v1.graph import MessageNode
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.judge import Judge, JudgeResponse, JudgeView
from verifiers.v1.judges import (
    Criterion,
    ReferenceJudge,
    ReferenceJudgeConfig,
    RubricJudge,
    RubricJudgeConfig,
)
from verifiers.v1.mcp import (
    SharedToolsetConfig,
    Toolset,
    ToolsetConfig,
)
from verifiers.v1.runtimes import (
    DockerConfig,
    PrimeConfig,
    ProgramResult,
    Runtime,
    RuntimeConfig,
    RuntimeInfo,
    RuntimeProcess,
    SubprocessConfig,
)
from verifiers.v1.state import State, StateT
from verifiers.v1.task import Task, TaskData, TaskResources, TaskTimeout, WireTaskData
from verifiers.v1.taskset import Taskset
from verifiers.v1.trace import (
    TRACE_VERSION,
    AgentInfo,
    AgentSpan,
    Branch,
    Error,
    EvalRunInfo,
    ModelCall,
    Reward,
    RunInfo,
    TimeSpan,
    TimeSplit,
    Timing,
    Trace,
    TraceTask,
    TrainRunInfo,
    VersionInfo,
    WireTrace,
)
from verifiers.v1.types import (
    ID,
    AssistantMessage,
    ContentPart,
    ImageUrlContentPart,
    ImageUrlSource,
    KeptTokens,
    Message,
    MessageContent,
    Messages,
    Response,
    Sampling,
    SamplingConfig,
    SystemMessage,
    TextContentPart,
    Tool,
    ToolCall,
    ToolMessage,
    TurnTokens,
    Usage,
    UserMessage,
)
from verifiers.v1.utils.artifacts import (
    ARTIFACTS_DIR,
    Artifact,
    collect,
    restore,
)
from verifiers.v1.utils.decorators import metric, reward, stop, tool
from verifiers.v1.utils.git import (
    PATCH_CAP_BYTES as PATCH_CAP_BYTES,
)
from verifiers.v1.utils.git import (
    capture_patch as capture_patch,
)
from verifiers.v1.utils.git import (
    resolve_head as resolve_head,
)
from verifiers.v1.utils.loaders import (
    default_harness_id,
    env_config_type,
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
    resolve_env_config,
    task_type,
    taskset_config_type,
)
from verifiers.v1.utils.score import (
    compare_stdout_results as compare_stdout_results,
)
from verifiers.v1.utils.score import (
    extract_boxed_answer as extract_boxed_answer,
)
from verifiers.v1.utils.score import (
    parse_judge_choice as parse_judge_choice,
)
from verifiers.v1.utils.score import (
    parse_pytest_outcomes as parse_pytest_outcomes,
)
from verifiers.v1.utils.score import (
    read_answer_file_or_last_reply as read_answer_file_or_last_reply,
)
from verifiers.v1.utils.score import (
    verify_boxed_math_answer as verify_boxed_math_answer,
)

__all__ = [  # noqa: RUF022 - grouped by public API area
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
    "Reward",
    "Episode",
    "WireEpisode",
    "TRACE_VERSION",
    "AgentInfo",
    "RunInfo",
    "EvalRunInfo",
    "ModelCall",
    "TrainRunInfo",
    "VersionInfo",
    "State",
    "StateT",
    "MessageNode",
    "Branch",
    "TurnTokens",
    "KeptTokens",
    "Timing",
    "TimeSpan",
    "TimeSplit",
    "AgentSpan",
    "Error",
    # decorators
    "stop",
    "tool",
    "metric",
    "reward",
    # errors
    "RolloutError",
    "EnvError",
    "ProviderError",
    "HarnessError",
    "ToolsetError",
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
    "FunctionConfig",
    "DecoratedFunctionConfig",
    "RewardFunctionConfig",
    "BaseConfig",
    "Harness",
    "HarnessSession",
    "HarnessConfig",
    "ACP",
    "ModelContext",
    "Runtime",
    "RuntimeProcess",
    "RuntimeConfig",
    "RuntimeInfo",
    "ProgramResult",
    "SubprocessConfig",
    "DockerConfig",
    "PrimeConfig",
    "Env",
    "SingleAgentEnv",
    "EnvConfig",
    "ServeConfig",
    "resolve_env_field",
    "narrowed_env_annotation",
    "SingleAgentEnvConfig",
    "AgentConfig",
    "StaticPoolConfig",
    "ElasticPoolConfig",
    "default_agent_harness",
    "pool_serve_kwargs",
    "RetryConfig",
    # agent
    "Agent",
    "Agents",
    "make_agent",
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
    "env_config_type",
    "resolve_env_config",
    "default_harness_id",
    # judge
    "Judge",
    "JudgeConfig",
    "Judges",
    "JudgeResponse",
    "JudgeView",
    "ReferenceJudge",
    "ReferenceJudgeConfig",
    "RubricJudge",
    "RubricJudgeConfig",
    "Criterion",
    # git patch capture
    "PATCH_CAP_BYTES",
    "capture_patch",
    "resolve_head",
    # grading artifacts
    "ARTIFACTS_DIR",
    "Artifact",
    "collect",
    "restore",
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
    "Interaction",
    "Segment",
]

# The library logs via stdlib logging (per-module `getLogger(__name__)`), but is
# silent until an app opts in: a NullHandler on the package root absorbs records
# so nothing is emitted (and no "no handler" warning) unless handlers are added.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())
