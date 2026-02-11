from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Literal,
    TypeAlias,
)


if TYPE_CHECKING:
    from datasets import Dataset

    import verifiers as vf
    from verifiers.errors import Error

if sys.version_info < (3, 12):
    from typing_extensions import NotRequired, TypedDict
else:
    from typing import NotRequired, TypedDict
from pydantic import BaseModel, ConfigDict, Field
from pydantic import field_validator

# Client / message type literals
ClientType = Literal["openai", "anthropic"]
MessageType = Literal["chat", "completion"]


# Provider-agnostic message + response types
class CustomBaseModel(BaseModel):
    """Allow extras and dict-like attribute access."""

    model_config = ConfigDict(extra="allow")

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.model_dump() == dict(other)
        return super().__eq__(other)


class TextMessage(CustomBaseModel):
    role: Literal["text"] = "text"
    content: str


class TextContentPart(CustomBaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrlSource(CustomBaseModel):
    url: str


class ImageUrlContentPart(CustomBaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlSource


class InputAudioSource(CustomBaseModel):
    data: str
    format: str


class InputAudioContentPart(CustomBaseModel):
    type: Literal["input_audio"] = "input_audio"
    input_audio: InputAudioSource


class GenericContentPart(CustomBaseModel):
    type: str


ContentPart: TypeAlias = (
    TextContentPart
    | ImageUrlContentPart
    | InputAudioContentPart
    | GenericContentPart
    | dict[str, Any]
)
MessageContent: TypeAlias = str | list[ContentPart]


class SystemMessage(CustomBaseModel):
    role: Literal["system"] = "system"
    content: MessageContent


class UserMessage(CustomBaseModel):
    role: Literal["user"] = "user"
    content: MessageContent


class ToolCall(CustomBaseModel):
    id: str
    name: str
    arguments: str


class AssistantMessage(CustomBaseModel):
    role: Literal["assistant"] = "assistant"
    content: MessageContent | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(CustomBaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: MessageContent


Message: TypeAlias = (
    SystemMessage | UserMessage | AssistantMessage | ToolMessage | TextMessage
)
Messages = list[Message]


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


class ResponseTokens(CustomBaseModel):
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]


FinishReason = Literal["stop", "length", "tool_calls"] | None


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


# Core data types
Info = dict[str, Any]
SamplingArgs = dict[str, Any]
IndividualRewardFunc = Callable[..., float | Awaitable[float]]
GroupRewardFunc = Callable[..., list[float] | Awaitable[list[float]]]
RewardFunc = IndividualRewardFunc | GroupRewardFunc
DatasetBuilder: TypeAlias = "Callable[[], Dataset]"


class TrajectoryStepTokens(TypedDict):
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    overlong_prompt: bool
    is_truncated: bool


class TokenUsage(TypedDict):
    input_tokens: float
    output_tokens: float


class VersionInfo(TypedDict):
    vf_version: str
    vf_commit: str | None
    env_version: str | None
    env_commit: str | None


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
    info: Info | str


class RolloutTiming(TypedDict, total=False):
    start_time: float
    generation_ms: float
    scoring_ms: float
    total_ms: float


class ErrorInfo(TypedDict):
    error: str
    error_chain_repr: str
    error_chain_str: str


class RolloutOutput(dict):
    """Serialized output from a rollout (mirrors RolloutInput).

    A dict subclass that allows typed access to known fields while supporting
    arbitrary additional fields from state_columns. All values must be
    JSON-serializable.

    Required fields: example_id, task, prompt, completion, reward, timing,
                     is_completed, is_truncated, metrics
    Optional fields: answer, info, error, stop_condition, trajectory, tool_defs,
                     token_usage
    Additional fields: arbitrary serializable state_columns
    """

    # Required fields
    example_id: int
    task: str
    prompt: Messages | None
    completion: Messages | None
    reward: float
    timing: RolloutTiming
    is_completed: bool
    is_truncated: bool
    metrics: dict[str, float]
    # Optional fields
    answer: str
    info: Info
    error: ErrorInfo | None
    stop_condition: str | None
    trajectory: list["TrajectoryStep"]
    tool_defs: list[Tool]
    token_usage: TokenUsage


class State(dict):
    INPUT_FIELDS = ["prompt", "answer", "task", "info", "example_id"]
    # rollout inputs
    input: RolloutInput
    client: vf.Client
    model: str
    sampling_args: SamplingArgs | None
    # created during rollout
    is_completed: bool
    is_truncated: bool
    stop_condition: str | None
    tool_defs: list[Tool]
    trajectory: list[TrajectoryStep]
    completion: Messages | None
    reward: float | None
    advantage: float | None
    metrics: dict[str, float] | None
    timing: RolloutTiming | None
    error: Error | None
    usage: TokenUsage | None
    usage_tracker: object

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
StartCallback = Callable[
    [list[RolloutInput], list[RolloutInput] | list[list[RolloutInput]]], None
]
ProgressCallback = Callable[
    [list[RolloutOutput], list[RolloutOutput], "GenerateMetadata"], None
]  # all_outputs, new_outputs, new_metadata
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
    avg_error: float
    usage: TokenUsage | None
    version_info: VersionInfo
    state_columns: list[str]
    path_to_save: Path
    tools: list[Tool] | None


class GenerateOutputs(TypedDict):
    """TypedDict for generation outputs (results)."""

    outputs: list[RolloutOutput]
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
    "Endpoint",
    {
        "key": str,
        "url": str,
        "model": str,
        "api_client_type": NotRequired[ClientType],
    },
)
Endpoints = dict[str, list[Endpoint]]


class ClientConfig(BaseModel):
    """Pydantic model for client configuration."""

    client_idx: int = 0
    client_type: ClientType = "openai"
    api_key_var: str = "PRIME_API_KEY"
    api_base_url: str = "https://api.pinference.ai/api/v1"
    endpoint_configs: list["EndpointClientConfig"] = Field(default_factory=list)
    timeout: float = 3600.0
    max_connections: int = 28000
    max_keepalive_connections: int = 28000
    max_retries: int = 10
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("endpoint_configs", mode="before")
    @classmethod
    def validate_non_recursive_endpoints(cls, value):
        if not isinstance(value, list):
            return value

        normalized_endpoints = []
        for endpoint in value:
            if isinstance(endpoint, ClientConfig):
                if endpoint.endpoint_configs:
                    raise ValueError(
                        "ClientConfig.endpoint_configs entries cannot include endpoint_configs"
                    )
                normalized_endpoints.append(
                    endpoint.model_dump(
                        mode="python",
                        exclude={"endpoint_configs"},
                        exclude_unset=True,
                    )
                )
                continue

            if (
                isinstance(endpoint, dict)
                and "endpoint_configs" in endpoint
                and endpoint["endpoint_configs"]
            ):
                raise ValueError(
                    "ClientConfig.endpoint_configs entries cannot include endpoint_configs"
                )

            nested = getattr(endpoint, "endpoint_configs", None)
            if nested:
                raise ValueError(
                    "ClientConfig.endpoint_configs entries cannot include endpoint_configs"
                )

            normalized_endpoints.append(endpoint)

        return normalized_endpoints


class EndpointClientConfig(BaseModel):
    """Leaf endpoint config used inside ClientConfig.endpoint_configs."""

    client_idx: int = 0
    api_key_var: str = "PRIME_API_KEY"
    api_base_url: str = "https://api.pinference.ai/api/v1"
    timeout: float = 3600.0
    max_connections: int = 28000
    max_keepalive_connections: int = 28000
    max_retries: int = 10
    extra_headers: dict[str, str] = Field(default_factory=dict)


ClientConfig.model_rebuild()


class EvalConfig(BaseModel):
    """Pydantic model for evaluation configuration."""

    # environment
    env_id: str
    env_args: dict
    env_dir_path: str
    # evaluation
    endpoint_id: str | None = None
    model: str
    client_config: ClientConfig
    sampling_args: SamplingArgs
    num_examples: int
    rollouts_per_example: int
    max_concurrent: int
    independent_scoring: bool = False
    extra_env_kwargs: dict = {}
    max_retries: int = 0
    # logging
    verbose: bool = False
    debug: bool = False
    # saving
    state_columns: list[str] | None = None
    save_results: bool = False
    resume_path: Path | None = None
    save_to_hf_hub: bool = False
    hf_hub_dataset_name: str | None = None


class EvalRunConfig(BaseModel):
    """Pydantic model for evaluation run configuration."""

    evals: list[EvalConfig]
