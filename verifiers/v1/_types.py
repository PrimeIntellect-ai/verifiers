from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_serializer
from pydantic.json_schema import WithJsonSchema
from typing_extensions import TypedDict


def encode_ndarray(arr: np.ndarray) -> dict:
    """A numpy array as a msgpack-safe dict (dtype + shape + raw bytes). The bytes ride a
    binary wire natively (msgpack's `bin` type) — no base64 — so a payload for such a wire
    must come from `model_dump(mode="python")` (`mode="json"` would coerce the bytes to str)."""
    arr = np.ascontiguousarray(arr)
    return {
        "__nd__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": arr.tobytes(),
    }


def decode_ndarray(d: dict) -> np.ndarray:
    """Reverse :func:`encode_ndarray`."""
    return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"])


def as_list(values: list | np.ndarray) -> list:
    """A token stream as a list, whichever form it is in."""
    return values.tolist() if isinstance(values, np.ndarray) else values


def _token_stream(dtype: type[np.generic], item_type: str):
    """The field type of a per-token stream: a numpy array of `dtype`.

    One list entry per token is a Python object the cyclic garbage collector walks and a
    serializer packs one by one; an array is one object it skips and raw bytes on the wire, so
    a process holding many long episodes (an RL orchestrator) keeps full collections short.
    Validation takes a list, an array or an `__nd__` dict; the models that hold streams
    serialize them (`MessageNode._serialize_token_stream`). Typed as an array alone,
    because a `list[int] | np.ndarray` union makes pydantic try the list branch element by
    element on every array it validates."""

    def validate(value: Any) -> np.ndarray:
        if isinstance(value, dict) and value.get("__nd__"):
            value = decode_ndarray(value)
        return np.asarray(value, dtype=dtype)

    return Annotated[
        np.ndarray,
        BeforeValidator(validate),
        WithJsonSchema({"type": "array", "items": {"type": item_type}}),
    ]


IntStream = _token_stream(np.int32, "integer")
"""Token ids: int32."""
BoolStream = _token_stream(np.bool_, "boolean")
"""Per-token flags: bool."""
FloatStream = _token_stream(np.float32, "number")
"""Per-token values (logprobs, advantages, loss weights): float32."""


def empty_stream(dtype: type[np.generic]) -> Callable[[], np.ndarray]:
    """A `default_factory` for an empty stream of `dtype`."""
    return lambda: np.zeros(0, dtype=dtype)


class TextContentPart(BaseModel):
    """A text part of multimodal message content."""

    type: Literal["text"] = "text"
    text: str
    """Text of the part."""


class ImageUrlSource(BaseModel):
    """Location of an image in an image content part."""

    url: str
    """Image URL, either remote or a base64 data URL."""


class ImageUrlContentPart(BaseModel):
    """An image part of multimodal message content."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlSource
    """Source of the image."""


ContentPart = Annotated[
    TextContentPart | ImageUrlContentPart, Field(discriminator="type")
]
MessageContent = str | list[ContentPart]
"""Plain text or typed multimodal content parts."""


def content_to_parts(content) -> MessageContent:
    """Type OpenAI content parts, dropping unsupported part types."""
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


def content_text(content: "MessageContent | None") -> str:
    """Extract text from message content, dropping images."""
    if isinstance(content, str):
        return content
    return "\n".join(
        part.text for part in content or [] if isinstance(part, TextContentPart)
    )


class SystemMessage(BaseModel):
    """A system message that instructs the model."""

    role: Literal["system"] = "system"
    """Message role, always "system"."""
    content: MessageContent
    """Text or content parts of the message."""


class UserMessage(BaseModel):
    """A user message sent to the model."""

    role: Literal["user"] = "user"
    """Message role, always "user"."""
    content: MessageContent
    """Text or content parts of the message."""


class ToolCall(BaseModel):
    """A tool call that the model emitted."""

    id: str
    """Provider ID of the tool call."""
    type: Literal["function", "custom"] = "function"
    name: str
    """Name of the called tool."""
    namespace: str | None = None
    """Provider namespace qualifying the tool name, when sent separately."""
    arguments: str
    """Raw function arguments or custom-tool input, exactly as the model emitted it."""


class AssistantMessage(BaseModel):
    """A message that the model generated."""

    role: Literal["assistant"] = "assistant"
    """Message role, always "assistant"."""
    content: str | None = None
    """Text the model generated."""
    text_block_offsets: list[int] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Character offsets between native text blocks concatenated into content."""
    reasoning_content: str | None = None
    """Reasoning text the model generated."""
    tool_calls: list[ToolCall] | None = None
    """Tool calls the model emitted."""
    provider_state: list[dict[str, Any]] | None = None
    """Opaque native items replayed to preserve signed or encrypted reasoning state."""


class ToolMessage(BaseModel):
    """The result of a tool call, sent back to the model."""

    role: Literal["tool"] = "tool"
    """Message role, always "tool"."""
    tool_call_id: str
    """ID of the tool call this message answers."""
    content: MessageContent
    """Text or content parts of the tool result."""
    name: str | None = None
    """Needed by templates such as Harmony when bridge tails omit the issuing call."""


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]
Messages = list[Message]


class Tool(BaseModel):
    """A tool that the model can call."""

    # Native declarations carry format, discovery, and provider-specific settings.
    model_config = ConfigDict(extra="allow")

    type: str = "function"
    """Tool kind, such as "function"."""
    name: str
    """Name the model uses to call the tool."""
    namespace: str | None = None
    """Provider namespace that qualifies the tool name."""
    description: str = ""
    """What the tool does, as shown to the model."""
    parameters: dict[str, Any] = Field(default_factory=dict)
    """JSON schema of the tool's arguments."""
    strict: bool | None = None
    """Whether the provider enforces the argument schema strictly."""


class Request(BaseModel):
    """The typed conversation about to cross a model or harness boundary."""

    messages: Messages
    """Conversation messages of the request."""
    tools: list[Tool] | None = None
    """Tools the model can call."""


FinishReason = Literal["stop", "length", "tool_calls"] | None


class Usage(BaseModel):
    """Provider token accounting.

    `prompt_tokens` excludes cache reads; `input_tokens` adds them back. Reasoning tokens
    are a subset of completion tokens and are not added to totals again.
    """

    prompt_tokens: int
    """Uncached input tokens."""
    completion_tokens: int
    """Generated tokens, including reasoning tokens."""
    cached_input_tokens: int | None = None
    """Input tokens read from the provider cache."""
    reasoning_tokens: int | None = None
    """Reasoning tokens, as a subset of the completion tokens."""
    cost: float | None = None
    """Cost the provider reported for the request."""

    @classmethod
    def from_openai(cls, usage: Any | None) -> "Usage | None":
        """Build usage while splitting cached tokens out of `prompt_tokens`."""
        if usage is None:
            return None
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached = prompt_details.cached_tokens if prompt_details else None
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning = completion_details.reasoning_tokens if completion_details else None
        return cls(
            prompt_tokens=usage.prompt_tokens - (cached or 0),
            completion_tokens=usage.completion_tokens,
            cached_input_tokens=cached,
            reasoning_tokens=reasoning,
            cost=getattr(usage, "cost", None),
        )

    @classmethod
    def aggregate(cls, usages: Iterable["Usage"]) -> "Usage | None":
        """Sum per-response usage while preserving whether cache usage was reported."""
        values = list(usages)
        if not values:
            return None
        # For the optional fields (cached / reasoning / cost), sum the responses that report them
        # and yield None only when *no* response does — so one response omitting a field (e.g. a
        # judge whose provider doesn't report reasoning or cost) doesn't null out the whole total.
        cached = [
            u.cached_input_tokens for u in values if u.cached_input_tokens is not None
        ]
        reasoning = [
            u.reasoning_tokens for u in values if u.reasoning_tokens is not None
        ]
        costs = [u.cost for u in values if u.cost is not None]
        return cls(
            prompt_tokens=sum(usage.prompt_tokens for usage in values),
            completion_tokens=sum(usage.completion_tokens for usage in values),
            cached_input_tokens=sum(cached) if cached else None,
            reasoning_tokens=sum(reasoning) if reasoning else None,
            cost=sum(costs) if costs else None,
        )

    @property
    def input_tokens(self) -> int:
        """Input tokens, including cached tokens."""
        return self.prompt_tokens + (self.cached_input_tokens or 0)

    @property
    def total_tokens(self) -> int:
        """Input and completion tokens together."""
        return self.input_tokens + self.completion_tokens


class RoutedExperts(TypedDict):
    """Base64 uint8 `[tokens, layers, top_k]` routing and its prompt offset."""

    data: Any
    """Base64 routing data."""
    shape: list[int]
    """Shape of the routing data."""
    start: int
    """Token offset where the routing data starts."""


class WeightVersionSpan(BaseModel):
    """Completion tokens `[start, end)` were sampled by the engine weights labelled `version`."""

    version: str
    """Weight version label of the engine."""
    start: int
    """First completion token of the span."""
    end: int
    """End of the span, exclusive."""


@dataclass
class SamplerSupport:
    """The sampler's support per token, in CSR form.

    Row `i` owns `counts[i]` entries of the flat int32 `ids` and, when the engine returned
    them, of the float32 `logprobs` (the sampler's logprob of each id). A `replayed` row is
    the exact set its token was sampled from under truncated sampling, with logprobs
    normalized over that set. Any other row is a head of the sampler's full distribution,
    independent of the sampled token.
    """

    ids: Any
    """Flat token ids of all rows."""
    counts: Any
    """Number of entries in each row."""
    replayed: Any
    """Per-row flags for rows replayed from truncated sampling."""
    logprobs: Any = None
    """Sampler logprob of each id, when the engine returned them."""

    @classmethod
    def from_response(cls, support: Mapping[str, Any]) -> "SamplerSupport":
        """Build from the renderer client's `sampler_support`: one row per completion token."""
        counts = np.asarray(support["counts"], dtype=np.int32)
        logprobs = support["logprobs"]
        return cls(
            ids=np.asarray(support["ids"], dtype=np.int32),
            counts=counts,
            replayed=np.full(len(counts), bool(support["replayed"])),
            logprobs=None if logprobs is None else np.asarray(logprobs, np.float32),
        )


class TurnTokens(BaseModel):
    """Training tokens from renderer tokenization or provider-returned token IDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: IntStream = Field(default_factory=empty_stream(np.int32))
    """Effective prompt IDs evaluated by the model, after multimodal expansion."""
    completion_ids: IntStream = Field(default_factory=empty_stream(np.int32))
    """Completion token IDs the model sampled."""
    completion_logprobs: FloatStream = Field(default_factory=empty_stream(np.float32))
    """Logprob of each completion token."""

    # Transient graph-construction metadata, consumed by the turn's commit and excluded
    # from serialized responses.
    renderer_prompt_ids: IntStream | None = Field(default=None, exclude=True)
    """Logical renderer IDs before multimodal expansion, retained for bridge extension."""
    bridged: bool = Field(default=False, exclude=True)
    """Whether the renderer constructed this prompt by extending a stored prefix."""
    # Transient carrier (excluded): per-message token spans into `prompt_ids` from the renderer,
    # consumed by the turn's `commit` to attribute tokens per message, then dropped.
    message_spans: list[tuple[int, int] | None] | None = Field(
        default=None, exclude=True
    )
    """Token span of each message in the prompt. Not serialized."""
    is_content: BoolStream | None = Field(default=None, exclude=True)
    """Per-token flag: True for message content. Not serialized."""
    # Authoritative effective-prompt ranges returned by vLLM, flattened across modalities.
    mm_placeholders: list[tuple[int, int]] | None = Field(default=None, exclude=True)
    """Token ranges of multimodal inputs in the prompt. Not serialized."""
    # Transient carrier (excluded): the renderer's special-token id -> modality marker map,
    # stamped onto `Trace.mm_token_type_id_map` by the turn's `commit`. None unless the
    # rendering renderer is multimodal.
    mm_token_type_id_map: dict[int, int] | None = Field(default=None, exclude=True)
    """Map from special token id to modality. Not serialized."""
    # Transient carrier (excluded): the MoE expert-routing data from `generate` (expert ids
    # per token), attributed per node by the turn's `commit` into `MessageNode.routed_experts`,
    # then dropped. None unless the engine ran with `enable_return_routed_experts`.
    routed_experts: RoutedExperts | None = Field(default=None, exclude=True)
    """MoE expert ids per token. Not serialized."""
    # Transient carrier (excluded): the sampler's support per completion token,
    # attributed to the assistant node by the turn's `commit`, then dropped.
    sampler_support: SamplerSupport | None = Field(default=None, exclude=True)
    """Sampler support per completion token. Not serialized."""
    # Transient carrier (excluded): the engine's weight-version spans over the completion,
    # attributed to the assistant node by the turn's `commit`, then dropped.
    weight_versions: list[WeightVersionSpan] | None = Field(default=None, exclude=True)
    """Weight versions that produced the completion. Not serialized."""

    @field_serializer("prompt_ids", "completion_ids", "completion_logprobs")
    def _serialize_stream(self, values: list | np.ndarray) -> list:
        return as_list(values)


class Response(BaseModel):
    """A model response for one turn."""

    id: str
    """Provider ID of the response."""
    created: int
    """Unix timestamp when the response was created."""
    model: str
    """Model that generated the response."""
    message: AssistantMessage
    """Message the model generated."""
    finish_reason: FinishReason
    """Why the model stopped generating."""
    usage: Usage | None = None
    """Token usage of the response."""
    tokens: TurnTokens | None = None
    """Token IDs and logprobs of the turn, when available."""
    raw: dict | None = Field(default=None, exclude=True, repr=False)
    """Full native response object returned to the program; excluded from traces."""
