from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Literal,
    TypeAlias,
)

from datasets import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from verifiers.clients.client import Client as VfClient
    from verifiers.errors import Error as VfError

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


# openai types
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,  # noqa: F401
)
from openai.types.chat.chat_completion_role import ChatCompletionRole  # noqa: F401
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,  # noqa: F401
)
from openai.types.shared_params import (  # noqa: F401
    FunctionDefinition,
    FunctionParameters,
)
from pydantic import BaseModel, ConfigDict

# typing aliases
ClientType = Literal["openai", "anthropic"]
MessageType = Literal["chat", "completion"]

TextMessage = str
TextMessages = str


class CustomBaseModel(BaseModel):
    """Allow extras and dict-like attribute access."""

    model_config = ConfigDict(extra="allow")

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)


class SystemMessage(CustomBaseModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(CustomBaseModel):
    role: Literal["user"] = "user"
    content: str


class ToolCall(CustomBaseModel):
    id: str
    name: str
    arguments: str


class AssistantMessage(CustomBaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(CustomBaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


ChatMessage: TypeAlias = SystemMessage | UserMessage | AssistantMessage | ToolMessage
ChatMessages = list[ChatMessage]

Message = ChatMessage
Messages = ChatMessages


class Tool(CustomBaseModel):
    name: str
    description: str
    parameters: dict[str, object]
    strict: bool | None = None


class Usage(CustomBaseModel):
    prompt_tokens: int
    reasoning_tokens: int
    completion_tokens: int
    total_tokens: int


FinishReason = Literal["stop", "length", "tool_calls"] | None


class ResponseTokens(CustomBaseModel):
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]


class ResponseMessage(AssistantMessage):
    finish_reason: FinishReason
    is_truncated: bool | None
    tokens: ResponseTokens | None = None


class Response(CustomBaseModel):
    id: str
    created: int
    model: str
    usage: Usage | None = None
    message: ResponseMessage  # can call tools


Info = dict[str, Any]
SamplingArgs = dict[str, Any]
IndividualRewardFunc = Callable[..., float | Awaitable[float]]
GroupRewardFunc = Callable[..., list[float] | Awaitable[list[float]]]
RewardFunc = IndividualRewardFunc | GroupRewardFunc
DatasetBuilder = Callable[[], Dataset]


class TrajectoryStepTokens(TypedDict):
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    overlong_prompt: bool
    is_truncated: bool


class TrajectoryStep(TypedDict):
    prompt: Messages
    completion: Messages
    response: Response
    tokens: TrajectoryStepTokens | None
    reward: float | None
    advantage: float | None
    is_truncated: bool
    trajectory_id: str
    extras: dict[str, Any]


class BaseRolloutInput(TypedDict):
    prompt: Messages
    example_id: int
    task: str


class RolloutInput(BaseRolloutInput, total=False):
    # required: prompt, example_id, task
    # optional: answer, info
    answer: str
    info: Info


class RolloutTiming(TypedDict, total=False):
    start_time: float
    generation_ms: float
    scoring_ms: float
    total_ms: float


class State(dict):
    INPUT_FIELDS = ["prompt", "answer", "task", "info", "example_id"]
    # rollout inputs
    input: RolloutInput
    client: VfClient
    model: str
    sampling_args: SamplingArgs | None
    # created during rollout
    is_completed: bool
    is_truncated: bool
    stop_condition: str | None
    oai_tools: list[Tool]
    trajectory: list[TrajectoryStep]
    completion: Messages | None
    reward: float | None
    advantage: float | None
    metrics: dict[str, float] | None
    timing: RolloutTiming | None
    error: VfError | None

    def __getitem__(self, key: str) -> Any:
        # forward to input if exists
        if key in self.INPUT_FIELDS and "input" in self:
            input_obj = super().__getitem__("input")
            if key in input_obj:
                return input_obj[key]
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        # forward to input if exists
        if key in self.INPUT_FIELDS and "input" in self:
            input_obj = super().__getitem__("input")
            if key in input_obj:
                input_obj[key] = value
                return
        super().__setitem__(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


# oai tools
JsonPrimitive = Literal["string", "number", "integer", "boolean", "array", "object"]

# callbacks
StartCallback = Callable[[int], None]  # total rollouts
ProgressCallback = Callable[[list[State], list[State]], None]  # all_states, new_states
LogCallback = Callable[[str], None]  # log messages


class GenerateMetadata(TypedDict):
    """Pydantic model for generation metadata."""

    env_id: str
    env_args: dict
    model: str
    base_url: str
    num_examples: int
    rollouts_per_example: int
    sampling_args: SamplingArgs
    date: str
    time_ms: float
    avg_reward: float
    avg_metrics: dict[str, float]
    state_columns: list[str]
    path_to_save: Path
    tools: list[Tool] | None


class GenerateOutputs(TypedDict):
    """TypedDict for generation outputs."""

    states: list[State]
    metadata: GenerateMetadata


class RolloutScore(TypedDict):
    """TypedDict for rollout scores."""

    reward: float
    metrics: dict[str, float]


class RolloutScores(TypedDict):
    """TypedDict for rubric outputs."""

    reward: list[float]
    metrics: dict[str, list[float]]


Endpoint = TypedDict(
    "Endpoint", {"key": str, "url": str, "model": str, "client_type": ClientType}
)
Endpoints = dict[str, Endpoint]


class ClientConfig(BaseModel):
    """Pydantic model for OpenAI client configuration."""

    client_type: ClientType = "openai"
    api_key_var: str = "PRIME_API_KEY"
    api_base_url: str = "https://api.pinference.ai/api/v1"
    timeout: float = 3600.0
    max_connections: int = 28000
    max_keepalive_connections: int = 28000
    max_retries: int = 10
    extra_headers: dict[str, str] = {}


class EvalConfig(BaseModel):
    """Pydantic model for evaluation configuration."""

    # environment
    env_id: str
    env_args: dict
    env_dir_path: str
    # evaluation
    model: str
    client_config: ClientConfig
    sampling_args: SamplingArgs
    num_examples: int
    rollouts_per_example: int
    max_concurrent: int
    max_concurrent_generation: int | None = None
    max_concurrent_scoring: int | None = None
    independent_scoring: bool = False
    interleaved_thinking: bool = True
    extra_env_kwargs: dict = {}
    max_retries: int = 0
    # logging
    verbose: bool = False
    use_tqdm: bool = True
    # saving
    state_columns: list[str] | None = None
    save_results: bool = False
    save_every: int = -1
    save_to_hf_hub: bool = False
    hf_hub_dataset_name: str | None = None


class EvalRunConfig(BaseModel):
    """Pydantic model for evaluation run configuration."""

    evals: list[EvalConfig]
