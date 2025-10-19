from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    TypedDict,
)

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
from pydantic import BaseModel, Field

# typing aliases
ChatMessage = ChatCompletionMessageParam
MessageType = Literal["chat", "completion"]
ModelResponse = Completion | ChatCompletion | None

Message = str | ChatMessage

Messages = str | list[ChatMessage]
Info = dict[str, Any]


State = dict[str, Any]
SamplingArgs = dict[str, Any]
RewardFunc = Callable[..., float | Awaitable[float]]

# oai tools
JsonPrimitive = Literal["string", "number", "integer", "boolean", "array", "object"]


class GenerateInputs(BaseModel):
    """Pydantic model for generation inputs."""

    prompt: list[Messages]
    completion: list[Messages] | None = None
    answer: list[str] | None = None
    task: list[str] | None = None
    info: list[Info] | None = None
    example_id: list[int] | None = None


class GenerateMetadata(BaseModel):
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


class GenerateOutputs(BaseModel):
    """Pydantic model for generation outputs."""

    prompt: list[Messages]
    completion: list[Messages]
    answer: list[str]
    state: list[State]
    task: list[str]
    info: list[Info]
    example_id: list[int]
    reward: list[float]
    metrics: dict[str, list[float]] = Field(default_factory=dict)
    metadata: GenerateMetadata


class RolloutScore(BaseModel):
    """Pydantic model for rollout scores."""

    reward: float
    metrics: dict[str, float] = Field(default_factory=dict)


class RolloutScores(BaseModel):
    """Pydantic model for rubric outputs."""

    reward: list[float]
    metrics: dict[str, list[float]] = Field(default_factory=dict)


class ProcessedOutputs(BaseModel):
    """Pydantic model for processed outputs."""

    prompt_ids: list[list[int]]
    prompt_mask: list[list[int]]
    completion_ids: list[list[int]]
    completion_mask: list[list[int]]
    completion_logprobs: list[list[float]]
    rewards: list[float]


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
    extra_headers: dict[str, str] | None = None


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
    interleave_scoring: bool = True
    # logging
    print_results: bool = False
    verbose: bool = False
    # saving
    state_columns: list[str] | None = None
    save_results: bool = False
    save_every: int = -1
    save_to_hf_hub: bool = False
    hf_hub_dataset_name: str | None = None
