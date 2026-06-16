"""Provider-agnostic wire types: messages, tools, model responses.

These are the only types that cross the boundary between verifiers and a model
provider. They are pydantic models — never dicts with normalizers. The single
place raw provider dicts enter is the client implementation, which validates
them into these models explicitly.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


@dataclass
class PlaceholderRange:
    offset: int
    length: int


@dataclass
class MultiModalData:
    """Multimodal trace data shared by all V1 clients."""

    mm_hashes: dict[str, list[str]] = field(default_factory=dict)
    mm_placeholders: dict[str, list[PlaceholderRange]] = field(default_factory=dict)
    mm_items: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.mm_hashes or self.mm_placeholders or self.mm_items)


class RoutedExperts(TypedDict):
    """The raw MoE expert-routing data a `generate` response carries for router replay:
    base64 `data` (uint8 `[tokens, layers, top_k]`), its `shape`, and `start` — the prompt
    offset where the routing begins (0 = full prompt+completion). Kept opaque (`Any` data)
    so pydantic never validates the encoded blob."""

    data: Any
    shape: list[int]
    start: int


class StrictBaseModel(BaseModel):
    """A pydantic base that rejects unknown fields. Use for all closed data types."""

    model_config = ConfigDict(extra="forbid")


# --- messages -----------------------------------------------------------------


class TextContentPart(StrictBaseModel):
    """A text span in a multimodal message body."""

    type: Literal["text"] = "text"
    text: str


class ImageUrlSource(StrictBaseModel):
    """An image reference — a URL or a `data:` URI."""

    url: str


class ImageUrlContentPart(StrictBaseModel):
    """An image in a multimodal message body (OpenAI `image_url` shape)."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlSource


ContentPart = Annotated[
    TextContentPart | ImageUrlContentPart, Field(discriminator="type")
]
MessageContent = str | list[ContentPart]
"""A message body: plain text, or a list of content parts (text + images)."""


def content_to_parts(content) -> MessageContent:
    """Parse raw OpenAI message content into typed content — a `str` stays a `str`; a list of
    parts becomes typed `ContentPart`s (text + image_url), dropping unknown part types. The
    shared multimodal-ingress parser used by the interception server and the v0 legacy bridge."""
    if not isinstance(content, list):
        return content or ""
    parts: list[ContentPart] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            parts.append(TextContentPart(text=p.get("text", "")))
        elif p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            parts.append(ImageUrlContentPart(image_url=ImageUrlSource(url=url)))
    return parts


class SystemMessage(StrictBaseModel):
    """A system instruction message."""

    role: Literal["system"] = "system"
    content: MessageContent


class UserMessage(StrictBaseModel):
    """A user message (text, or text + images)."""

    role: Literal["user"] = "user"
    content: MessageContent


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
    content: MessageContent


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


class TurnTokens(StrictBaseModel):
    """Token ids + sampling logprobs for one response, for training. Populated by the
    renderer client (client-side tokenization) or the chat client (parsed from vLLM's
    token ids); None when the provider returns neither."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    prompt_ids: list[int] = Field(default_factory=list)
    completion_ids: list[int] = Field(default_factory=list)
    completion_logprobs: list[float] = Field(default_factory=list)

    # Transient carrier (excluded): per-message token spans into `prompt_ids` from the renderer,
    # consumed by `graph.add_turn` to attribute tokens per message, then dropped.
    message_spans: list[tuple[int, int] | None] | None = Field(
        default=None, exclude=True
    )
    # Transient carrier (excluded): the renderer's multimodal sidecar (image tensors + offsets),
    # attributed per node by `graph.add_turn`, then dropped — never persisted.
    multi_modal_data: MultiModalData | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): the MoE expert-routing data from `generate` (expert ids
    # per token), attributed per node by `graph.add_turn` into `MessageNode.routed_experts`,
    # then dropped. None unless the engine ran with `enable_return_routed_experts`.
    routed_experts: RoutedExperts | None = Field(default=None, exclude=True)


class Response(StrictBaseModel):
    """One model completion, provider-agnostic."""

    id: str
    created: int
    model: str
    message: AssistantMessage
    finish_reason: FinishReason
    usage: Usage | None = None
    tokens: TurnTokens | None = None
    """Token ids + logprobs for training (set by the renderer client)."""
    raw: dict | None = Field(default=None, exclude=True, repr=False)
    """The wire response the interception server hands back to the program 1:1: the provider's
    verbatim bytes (proxy, so no field is lost) or the client's serialized completion (renderer,
    which generates and has none to relay). Transient: excluded from the trace dump."""


# --- sampling -----------------------------------------------------------------


class SamplingConfig(BaseModel):
    """Typed sampling knobs; provider-specific keys pass through (extra='allow')."""

    model_config = ConfigDict(extra="allow")
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None
    max_tokens: int | None = Field(
        None, validation_alias=AliasChoices("max_tokens", "max_completion_tokens")
    )
