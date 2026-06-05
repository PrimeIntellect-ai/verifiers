"""Provider-agnostic wire types: messages, tools, model responses.

These are the only types that cross the boundary between verifiers and a model
provider. They are pydantic models — never dicts with normalizers. The single
place raw provider dicts enter is the client implementation, which validates
them into these models explicitly.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """A pydantic base that rejects unknown fields. Use for all closed data types."""

    model_config = ConfigDict(extra="forbid")


# --- messages -----------------------------------------------------------------


class SystemMessage(StrictBaseModel):
    """A system instruction message."""

    role: Literal["system"] = "system"
    content: str


class UserMessage(StrictBaseModel):
    """A user message."""

    role: Literal["user"] = "user"
    content: str


class ToolCall(StrictBaseModel):
    """A tool/function call requested by the model."""

    id: str
    name: str
    arguments: str
    """Raw JSON string of arguments, exactly as the model emitted it."""


class AssistantMessage(StrictBaseModel):
    """An assistant message, optionally carrying reasoning and tool calls."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(StrictBaseModel):
    """The result of a tool call, keyed to the originating call id."""

    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]
Messages = list[Message]


# --- tools --------------------------------------------------------------------


class Tool(StrictBaseModel):
    """A function tool advertised to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool | None = None


# --- responses ----------------------------------------------------------------


FinishReason = Literal["stop", "length", "tool_calls"] | None


class Usage(StrictBaseModel):
    """Token usage for one model response (total_tokens is derived)."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Response(StrictBaseModel):
    """One model completion, provider-agnostic."""

    id: str
    created: int
    model: str
    message: AssistantMessage
    finish_reason: FinishReason
    usage: Usage | None = None


# --- sampling -----------------------------------------------------------------


class SamplingConfig(BaseModel):
    """Typed sampling knobs; provider-specific keys pass through (extra='allow')."""

    model_config = ConfigDict(extra="allow")
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
