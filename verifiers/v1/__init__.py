"""Public v1 API."""

import logging as _logging

from pydantic_config import BaseConfig

from verifiers.v1.agent import Agent
from verifiers.v1.clients import (
    BaseClientConfig,
    Client,
    ClientConfig,
    EvalClientConfig,
    ModelContext,
    TrainClientConfig,
    resolve_client,
)
from verifiers.v1.decorators import group_reward, metric, reward, stop, tool
from verifiers.v1.env import (
    ElasticPoolConfig,
    EnvConfig,
    Environment,
    EnvServerConfig,
    StaticPoolConfig,
    TimeoutConfig,
    pool_serve_kwargs,
)
from verifiers.v1.episode import Episode
from verifiers.v1.errors import (
    HarnessError,
    InterceptionError,
    ProviderError,
    RolloutError,
    SandboxError,
    TaskError,
    ToolsetError,
    TopologyError,
    TunnelError,
    UserError,
)
from verifiers.v1.graph import MessageNode
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.judge import (
    Judge,
    JudgeConfig,
    JudgeResponse,
    Judges,
    JudgeSamplingConfig,
    JudgeView,
    judge_verdict,
)
from verifiers.v1.judges import (
    Criterion,
    ReferenceJudge,
    ReferenceJudgeConfig,
    RubricJudge,
    RubricJudgeConfig,
)
from verifiers.v1.loaders import (
    default_harness_id,
    harness_config_type,
    import_harness,
    import_taskset,
    import_topology,
    load_harness,
    load_taskset,
    load_topology,
    task_type,
    taskset_config_type,
    topology_config_type,
)
from verifiers.v1.mcp import (
    Toolset,
    ToolsetConfig,
    User,
    UserConfig,
)
from verifiers.v1.retries import RetryConfig, RolloutRetryConfig
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import (
    DockerConfig,
    PrimeConfig,
    ProgramResult,
    Runtime,
    RuntimeConfig,
    RuntimeInfo,
    SubprocessConfig,
)
from verifiers.v1.scoring import (
    compare_stdout_results as compare_stdout_results,
)
from verifiers.v1.scoring import (
    extract_boxed_answer as extract_boxed_answer,
)
from verifiers.v1.scoring import (
    parse_judge_choice as parse_judge_choice,
)
from verifiers.v1.scoring import (
    parse_pytest_outcomes as parse_pytest_outcomes,
)
from verifiers.v1.scoring import (
    read_answer_file_or_last_reply as read_answer_file_or_last_reply,
)
from verifiers.v1.scoring import (
    verify_boxed_math_answer as verify_boxed_math_answer,
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
from verifiers.v1.agent import Session, SessionEnded
from verifiers.v1.services import RunServices
from verifiers.v1.topology import (
    AgentBinding,
    AgentConfig,
    AgentGraph,
    DirectAgentConfig,
    NullAgentConfig,
    Topology,
    TopologyAgent,
    TopologyConfig,
    TopologyRun,
    TopologyRunner,
)
from verifiers.v1.mcp import (
    SharedToolsetConfig,
)
from verifiers.v1.trace import (
    Branch,
    Error,
    TimeSpan,
    Timing,
    Trace,
    TraceTask,
    WireTrace,
)
from verifiers.v1.types import (
    ID,
    AssistantMessage,
    ContentPart,
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
    ToolMessage,
    TurnTokens,
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
    "ToolsetError",
    "UserError",
    "SandboxError",
    "TaskError",
    "TopologyError",
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
    "StaticPoolConfig",
    "ElasticPoolConfig",
    "pool_serve_kwargs",
    "RetryConfig",
    "RolloutRetryConfig",
    "TimeoutConfig",
    "Episode",
    "Rollout",
    # agents / topology
    "Agent",
    "AgentBinding",
    "AgentConfig",
    "AgentGraph",
    "DirectAgentConfig",
    "NullAgentConfig",
    "RunServices",
    "Session",
    "SessionEnded",
    "Topology",
    "TopologyAgent",
    "TopologyConfig",
    "TopologyRunner",
    "TopologyRun",
    # loaders
    "import_taskset",
    "import_harness",
    "import_topology",
    "load_taskset",
    "load_harness",
    "load_topology",
    "task_type",
    "taskset_config_type",
    "harness_config_type",
    "topology_config_type",
    "default_harness_id",
    # judge (a single-call utility — judge-as-agent is the llm-judge/agentic-judge topologies)
    "Judge",
    "JudgeConfig",
    "JudgeSamplingConfig",
    "JudgeResponse",
    "judge_verdict",
    "Judges",
    "JudgeView",
    "Criterion",
    "ReferenceJudge",
    "ReferenceJudgeConfig",
    "RubricJudge",
    "RubricJudgeConfig",
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
    # user simulator
    "User",
    "UserConfig",
]

# The library logs via stdlib logging (per-module `getLogger(__name__)`), but is
# silent until an app opts in: a NullHandler on the package root absorbs records
# so nothing is emitted (and no "no handler" warning) unless handlers are added.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())
