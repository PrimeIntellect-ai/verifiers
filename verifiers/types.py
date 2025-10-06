from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
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
from pydantic import BaseModel, Field, field_validator

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
    answer: list[str] | None = None
    info: list[dict] | None = None
    task: list[str] | None = None
    completion: list[Messages] | None = None

    # This patch is necessary to deal with Iterable types in OAI ChatCompletionMessageParam, typically used for multi-modal inputs
    # For more details, see https://github.com/pydantic/pydantic/issues/9467#issuecomment-2442097291
    @field_validator("prompt", mode="after")
    def materialize_prompt_messages(cls, prompt) -> Iterable:
        def materialize_message_content(message):
            if isinstance(message["content"], str):
                return message
            elif isinstance(message["content"], Iterable):
                materialized_content = []
                while True:
                    try:
                        materialized_content.append(next(message["content"]))
                    except StopIteration:
                        break
                return {**message, "content": materialized_content}
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}"
                )

        return [[materialize_message_content(m) for m in p] for p in prompt]


class GenerateOutputs(BaseModel):
    """Pydantic model for generation outputs."""

    prompt: list[Messages]
    completion: list[Messages]
    answer: list[str]
    state: list[State]
    info: list[Info]
    task: list[str]
    reward: list[float]
    metrics: dict[str, list[float]] = Field(default_factory=dict)


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
