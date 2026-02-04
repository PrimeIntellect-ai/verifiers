from __future__ import annotations

import sys
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Literal,
    TypeAlias,
)

from verifiers.errors import Error

if TYPE_CHECKING:
    from datasets import Dataset

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

# openai types
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,  # noqa: F401
)
from openai.types.chat.chat_completion_role import ChatCompletionRole  # noqa: F401
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,  # noqa: F401
)
from openai.types.completion import Completion
from openai.types.shared_params import (  # noqa: F401
    FunctionDefinition,
    FunctionParameters,
)
from pydantic import BaseModel

# typing aliases
ChatMessage = ChatCompletionMessageParam
MessageType = Literal["chat", "completion"]
ModelResponse = Completion | ChatCompletion | None

ChatMessages = list[ChatMessage]
Message = str | ChatMessage

Messages = str | list[ChatMessage]
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


class TrajectoryStep(TypedDict):
    prompt: Messages
    completion: Messages
    response: ModelResponse
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
    Optional fields: answer, info, error, stop_condition, trajectory, oai_tools,
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
    oai_tools: list["ChatCompletionToolParam"]
    token_usage: TokenUsage


class State(dict):
    INPUT_FIELDS = ["prompt", "answer", "task", "info", "example_id"]
    # rollout inputs
    input: RolloutInput
    client: AsyncOpenAI
    model: str
    sampling_args: SamplingArgs | None
    # created during rollout
    is_completed: bool
    is_truncated: bool
    stop_condition: str | None
    oai_tools: list[ChatCompletionToolParam]
    trajectory: list[TrajectoryStep]
    completion: Messages | None
    reward: float | None
    advantage: float | None
    metrics: dict[str, float] | None
    timing: RolloutTiming | None
    error: Error | None

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
ProgressCallback = Callable[
    [list[RolloutOutput], list[RolloutOutput]], None
]  # all_outputs, new_outputs
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
    usage: TokenUsage | None
    state_columns: list[str]
    path_to_save: Path
    tools: list[ChatCompletionToolParam] | None


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


Endpoint = TypedDict("Endpoint", {"key": str, "url": str, "model": str})
Endpoints = dict[str, Endpoint]


class ClientConfig(BaseModel):
    """Pydantic model for OpenAI client configuration."""

    client_idx: int = 0
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
    independent_scoring: bool = False
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
