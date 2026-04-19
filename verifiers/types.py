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

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from anthropic.types import RedactedThinkingBlock
    from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
    from datasets import Dataset

    from verifiers.clients import Client
    from verifiers.errors import Error
else:
    RedactedThinkingBlock = Any
    AnthropicThinkingBlock = Any

if sys.version_info < (3, 12):
    from typing_extensions import NotRequired, TypedDict
else:
    from typing import NotRequired, TypedDict

# Client / message type literals
ClientType = Literal[
    "openai_completions",
    "openai_chat_completions",
    "openai_chat_completions_token",
    "anthropic_messages",
    "nemorl_chat_completions",
]
MessageType = Literal["chat", "completion"]  # deprecated


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


ThinkingBlock: TypeAlias = AnthropicThinkingBlock | RedactedThinkingBlock


class AssistantMessage(CustomBaseModel):
    role: Literal["assistant"] = "assistant"
    content: MessageContent | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[ThinkingBlock] | None = None
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
    routed_experts: list[list[list[int]]] | None = None  # [seq_len, layers, topk]


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
    routed_experts: list[list[list[int]]] | None  # [seq_len, layers, topk]


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
    example_id: int | str
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
                     is_completed, is_truncated, metrics, sampling_args,
                     trajectory_id
    Optional fields: answer, info, error, stop_condition, trajectory, tool_defs,
                     token_usage, mar_score
    Additional fields: arbitrary serializable state_columns
    """

    # Required fields
    example_id: int | str
    task: str
    prompt: Messages | None
    completion: Messages | None
    reward: float
    timing: RolloutTiming
    is_completed: bool
    is_truncated: bool
    metrics: dict[str, float]
    sampling_args: "SamplingArgs"  # required by multi_agent_bridge
    trajectory_id: str  # required by multi_agent_bridge
    # Optional fields
    answer: str
    info: Info
    error: ErrorInfo | None
    stop_condition: str | None
    trajectory: list["TrajectoryStep"]
    tool_defs: list[Tool]
    token_usage: TokenUsage
    mar_score: "MARScore"  # populated by MultiAgentRubric subclasses


class MemberRollout(TypedDict):
    """RolloutOutput-compatible dict with per-member multi-agent metadata."""

    # Training-path fields (read by pretokenize -> interleave -> TrainingSample).
    example_id: int | str
    task: str
    trajectory: list[TrajectoryStep]
    sampling_args: dict[str, Any]
    error: ErrorInfo | None
    reward: float

    # Multi-agent metadata
    episode_id: str
    member_id: str


class State(dict):
    INPUT_FIELDS = ["prompt", "answer", "task", "info", "example_id"]
    # rollout inputs
    input: RolloutInput
    client: Client
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
        if key in self.INPUT_FIELDS and super().__contains__("input"):
            input_obj = super().__getitem__("input")
            if key in input_obj:
                return input_obj[key]
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        # forward to input if exists
        if key in self.INPUT_FIELDS and super().__contains__("input"):
            input_obj = super().__getitem__("input")
            if key in input_obj:
                input_obj[key] = value
                return
        super().__setitem__(key, value)

    def __contains__(self, key: object) -> bool:
        # Mirror __getitem__ forwarding so ``"answer" in state`` matches
        # ``state["answer"]`` access. Without this override, membership
        # checks see only top-level dict keys while reads silently reach
        # into state["input"] — a subtle trap that bit gpqa_debate's
        # build_marscore.
        if (
            isinstance(key, str)
            and key in self.INPUT_FIELDS
            and super().__contains__("input")
        ):
            input_obj = super().__getitem__("input")
            if key in input_obj:
                return True
        return super().__contains__(key)

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
    pass_at_k: dict[str, float]
    pass_all_k: dict[str, float]
    pass_threshold: float
    usage: TokenUsage | None
    version_info: VersionInfo
    state_columns: list[str]
    path_to_save: Path
    tools: list[Tool] | None


class GenerateOutputs(TypedDict):
    """TypedDict for generation outputs (results)."""

    outputs: list[RolloutOutput]
    metadata: GenerateMetadata


class MemberScore(CustomBaseModel):
    """Per-member outcome of one episode in a multi-agent rollout.

    The bridge consumes ``reward`` (training signal) and ``member_id``
    (advantage-baseline partition key in RAE). ``metrics`` projects to
    wandb as ``f"{k}/{member_id}"`` at the serialization boundary.
    ``parse_error_count`` propagates to ``f"parse_errors/{member_id}"``
    when non-zero.

    ``metrics`` is intentionally float-only — bools and ints must be
    converted at the producer site so the aggregator can average without
    a type check. Categorical/string telemetry doesn't belong here.
    """

    member_id: str
    reward: float
    parse_error_count: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)


_RESERVED_FLAT_KEYS: frozenset[str] = frozenset(
    {
        # RolloutOutput canonical fields — kept in sync with RolloutOutput
        # above. Projected MARScore keys must not collide with these,
        # otherwise state_to_output's ``output[k] = v`` flattening loop
        # silently rewrites the canonical field with metric data.
        "example_id",
        "task",
        "prompt",
        "completion",
        "reward",
        "timing",
        "is_completed",
        "is_truncated",
        "metrics",
        "answer",
        "info",
        "error",
        "stop_condition",
        "trajectory",
        "tool_defs",
        "token_usage",
        "mar_score",
        "trajectory_id",
        "sampling_args",
    }
)


class MARScore(CustomBaseModel):
    """Member-Attributed Reward — single source of truth for episode scoring.

    Replaces the legacy multi-key spread:

    - ``state["reward"]``         -> ``MARScore.episode_scalar``
    - ``state["metrics"]``        -> ``MARScore.to_wandb_flat()`` (boundary)
    - ``state["member_rewards"]`` -> ``{m.member_id: m.reward for m in members}``
    - ``state["commits"]``        -> deleted (recomputable from trajectory)
    - ``state["error_info"]``     -> deleted (state["error"] is the channel)

    Episode-level telemetry is split by purpose so the wandb projection
    can't accidentally average a categorical sentinel like a winner-index
    encoded as a float (a ZIP-code mistake):

    - ``episode_metrics``      averageable scalars only (accuracy, agreement)
    - ``episode_categorical``  codes/labels (winner = "debater_a"/"tie"/None)
    - ``episode_error``        error metadata when the rollout failed

    Schema invariants enforced at construction (duplicate member_id, empty
    members) — drift between rubric writer and bridge reader is structurally
    impossible after this lands.
    """

    members: list[MemberScore]
    episode_scalar: float
    episode_metrics: dict[str, float] = Field(default_factory=dict)
    episode_categorical: dict[str, str | None] = Field(default_factory=dict)
    episode_error: dict[str, str] | None = None

    @field_validator("members")
    @classmethod
    def _validate_members(cls, v: list[MemberScore]) -> list[MemberScore]:
        if not v:
            raise ValueError("MARScore.members cannot be empty")
        ids = [m.member_id for m in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate member_id in MARScore: {ids}")
        return v

    def by_id(self) -> dict[str, MemberScore]:
        """O(N) lookup table; small N, no need to memoize."""
        return {m.member_id: m for m in self.members}

    def to_metrics_flat(self) -> dict[str, float]:
        """Projection of averageable scalars to flat wandb keys.

        Episode-level scalars stay top-level. Per-member metrics project
        to ``f"{k}/{member_id}"``. Reward and parse_error_count get
        canonical prefixes. Categorical / error fields are NOT included
        — those live under their own MARScore fields and are projected
        separately at the serialization boundary.

        Reserved keys (canonical ``RolloutOutput`` fields) are rejected:
        without this guard, an ``episode_metrics={'reward': -1}`` would
        silently overwrite ``output['reward']`` at the wire boundary
        (``state_to_output`` does ``output[k] = v`` for each flat key),
        corrupting training signal.
        """
        out: dict[str, float] = {}
        for k, v in self.episode_metrics.items():
            if k in _RESERVED_FLAT_KEYS:
                raise ValueError(
                    f"MARScore.episode_metrics key {k!r} collides with a "
                    f"reserved RolloutOutput field; rename it (e.g. "
                    f"{k!s}_metric)"
                )
            out[k] = v
        for m in self.members:
            out[f"reward/{m.member_id}"] = m.reward
            if m.parse_error_count:
                out[f"parse_errors/{m.member_id}"] = float(m.parse_error_count)
            for k, v in m.metrics.items():
                if k in _RESERVED_FLAT_KEYS:
                    raise ValueError(
                        f"MemberScore.metrics key {k!r} collides with a "
                        f"reserved RolloutOutput field; rename it (e.g. "
                        f"{k!s}_metric)"
                    )
                out[f"{k}/{m.member_id}"] = v
        return out


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
        "extra_headers": NotRequired[dict[str, str]],
    },
)
Endpoints = dict[str, list[Endpoint]]


def _validate_extra_headers_value(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("extra_headers must be a dict")
    out: dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError("extra_headers keys must be non-empty strings")
        if not isinstance(v, str):
            raise ValueError("extra_headers values must be strings")
        out[k] = v
    return out


class ClientConfig(BaseModel):
    """Pydantic model for client configuration."""

    client_idx: int = 0
    client_type: ClientType = "openai_chat_completions"
    api_key_var: str = "PRIME_API_KEY"
    api_base_url: str = "https://api.pinference.ai/api/v1"
    endpoint_configs: list["EndpointClientConfig"] = Field(default_factory=list)
    timeout: float = 3600.0
    connect_timeout: float = 5.0
    max_connections: int = 28000
    max_keepalive_connections: int = 28000
    max_retries: int = 10
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_headers_from_state: dict[str, str] = Field(
        default_factory=dict,
        description="Maps HTTP header names to state field names. "
        "For each request, the header value is read from the state dict. "
        'e.g. {"X-Session-ID": "example_id"} adds a X-Session-ID header '
        "with the value of state['example_id'].",
    )

    @field_validator("extra_headers", mode="before")
    @classmethod
    def validate_extra_headers(cls, value: object) -> dict[str, str]:
        return _validate_extra_headers_value(value)

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
    connect_timeout: float = 5.0
    max_connections: int = 28000
    max_keepalive_connections: int = 28000
    max_retries: int = 10
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("extra_headers", mode="before")
    @classmethod
    def validate_extra_headers(cls, value: object) -> dict[str, str]:
        return _validate_extra_headers_value(value)


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
    num_workers: int | str = "auto"
    independent_scoring: bool = False
    extra_env_kwargs: dict = {}
    max_retries: int = 0
    disable_env_server: bool = False
    # logging
    verbose: bool = False
    debug: bool = False
    # saving
    output_dir: str | None = None
    state_columns: list[str] | None = None
    save_results: bool = False
    resume_path: Path | None = None
    save_to_hf_hub: bool = False
    hf_hub_dataset_name: str | None = None


class EvalRunConfig(BaseModel):
    """Pydantic model for evaluation run configuration."""

    evals: list[EvalConfig]
    heartbeat_url: str | None = None
