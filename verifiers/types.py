import sys
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
)

from datasets import Dataset

from verifiers.errors import Error

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
    info: Info


class RolloutTiming(TypedDict, total=False):
    start_time: float
    generation_ms: float
    scoring_ms: float
    total_ms: float


class RolloutResultTrajectoryStep(TypedDict, total=False):
    """Serializable trajectory step for RolloutResult."""

    prompt: Messages
    completion: Messages
    tokens: TrajectoryStepTokens
    advantage: float


class _RolloutResultRequired(TypedDict):
    """Required fields for RolloutResult."""

    prompt: Messages
    completion: Messages
    example_id: int
    task: str
    metrics: dict[str, float]


class RolloutResult(_RolloutResultRequired, total=False):
    """Serializable result from a rollout, used in GenerateOutputs.

    This is the public contract for what evaluation/generation returns.
    Unlike State, this contains only pickleable fields suitable for IPC.
    """

    # Optional input fields
    answer: str
    info: Info
    # From rollout execution
    reward: float
    is_truncated: bool
    stop_condition: str | None
    trajectory: list[RolloutResultTrajectoryStep]
    timing: RolloutTiming
    error: str | None  # Stringified error (not the original Exception)


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


class GenerateMetadata(TypedDict):
    """Metadata about a generation run."""

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


class GenerateOutputs(TypedDict):
    """Output from generate() - rollouts plus metadata."""

    rollouts: list[RolloutResult]
    metadata: GenerateMetadata


class RolloutScore(TypedDict):
    """TypedDict for rollout scores."""

    reward: float
    metrics: dict[str, float]


class RolloutScores(TypedDict):
    """TypedDict for rubric outputs."""

    reward: list[float]
    metrics: dict[str, list[float]]


class ProcessedOutputs(TypedDict):
    """TypedDict for processed outputs."""

    prompt_ids: list[list[int]]
    prompt_mask: list[list[int]]
    completion_ids: list[list[int]]
    completion_mask: list[list[int]]
    completion_logprobs: list[list[float]]
    rewards: list[float]
    is_truncated: list[bool]


Endpoint = TypedDict("Endpoint", {"key": str, "url": str, "model": str})
Endpoints = dict[str, Endpoint]


class ClientConfig(BaseModel):
    """Pydantic model for OpenAI client configuration."""

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
    extra_env_kwargs: dict = {}
    num_workers: int = 1
    # logging
    print_results: bool = False
    verbose: bool = False
    # saving
    state_columns: list[str] = []
    save_results: bool = False
    save_every: int = -1
    save_to_hf_hub: bool = False
    hf_hub_dataset_name: str | None = None
