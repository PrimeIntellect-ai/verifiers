import base64
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_serializer
from pydantic.json_schema import WithJsonSchema
from typing_extensions import TypedDict


def encode_ndarray(arr: np.ndarray) -> dict:
    """A numpy array as a msgpack-safe dict (dtype + shape + raw bytes). The bytes ride the
    env-server wire natively via msgpack's `bin` type — no base64 — so the response must be
    packed from `model_dump(mode="python")` (`mode="json"` would coerce the bytes to str)."""
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

    One list entry per token is a Python object the cyclic garbage collector walks and the
    msgpack wire packs one by one; an array is one object it skips and raw bytes on the wire, so
    a process holding many long episodes (an RL orchestrator) keeps full collections short.
    Validation takes a list, an array or an `__nd__` dict; the models that hold streams
    serialize them (`graph.MessageNode.serialize_token_stream`). Typed as an array alone,
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
    type: Literal["text"] = "text"
    text: str


class ImageUrlSource(BaseModel):
    url: str


class ImageUrlContentPart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlSource


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
    role: Literal["system"] = "system"
    content: MessageContent


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: MessageContent


class ToolCall(BaseModel):
    id: str
    type: Literal["function", "custom"] = "function"
    name: str
    namespace: str | None = None
    """Provider namespace qualifying the tool name, when sent separately."""
    arguments: str
    """Raw function arguments or custom-tool input, exactly as the model emitted it."""


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    provider_state: list[dict[str, Any]] | None = None
    """Opaque native items replayed to preserve signed or encrypted reasoning state."""


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: MessageContent
    name: str | None = None
    """Needed by templates such as Harmony when bridge tails omit the issuing call."""


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]
Messages = list[Message]


class Tool(BaseModel):
    # Native declarations carry format, discovery, and provider-specific settings.
    model_config = ConfigDict(extra="allow")

    type: str = "function"
    name: str
    namespace: str | None = None
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None


class Request(BaseModel):
    """The typed conversation about to cross a model or harness boundary."""

    messages: Messages
    tools: list[Tool] | None = None


FinishReason = Literal["stop", "length", "tool_calls"] | None


class Usage(BaseModel):
    """Provider token accounting.

    `prompt_tokens` excludes cache reads; `input_tokens` adds them back. Reasoning tokens
    are a subset of completion tokens and are not added to totals again.
    """

    prompt_tokens: int
    completion_tokens: int
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = None

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
        return self.prompt_tokens + (self.cached_input_tokens or 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.completion_tokens


class RoutedExperts(TypedDict):
    """Base64 uint8 `[tokens, layers, top_k]` routing and its prompt offset."""

    data: Any
    shape: list[int]
    start: int


class WeightVersionSpan(BaseModel):
    """Completion tokens `[start, end)` were sampled by the engine weights labelled `version`."""

    version: str
    start: int
    end: int


@dataclass
class SamplingMask:
    """Sampling masks stored as flat int32 `ids` and `counts` arrays.

    Each row contains the token ids that survived sampling filters for one completion
    token. Row boundaries are recovered from `counts`.
    """

    ids: Any
    counts: Any

    @classmethod
    def from_sampling_mask(
        cls, sampling_mask: list[list[int]] | dict[str, Any]
    ) -> "SamplingMask":
        """Build from vLLM's one-list-per-token masks, or from prime-rl's packed CSR
        form: `{"ids", "counts"}`, each a base64 `{data, shape, dtype}` int32 array."""
        if isinstance(sampling_mask, dict):
            return cls(
                ids=_decode_packed(sampling_mask["ids"], np.int32),
                counts=_decode_packed(sampling_mask["counts"], np.int32),
            )
        counts = np.fromiter(
            (len(row) for row in sampling_mask),
            dtype=np.int32,
            count=len(sampling_mask),
        )
        ids = (
            np.concatenate([np.asarray(row, dtype=np.int32) for row in sampling_mask])
            if int(counts.sum())
            else np.empty(0, dtype=np.int32)
        )
        return cls(ids=ids, counts=counts)


def _decode_packed(packed: dict[str, Any], dtype: type) -> np.ndarray:
    array = np.frombuffer(base64.b64decode(packed["data"]), dtype=packed["dtype"])
    return array.astype(dtype, copy=False)


@dataclass
class TopLogprobs:
    """Top-k sampling head stored as flat `ids`, `logprobs` and `counts` arrays.

    Each row holds one completion token's top-k sampler candidates in descending
    probability order, independent of the sampled action. Row boundaries come from
    `counts`; `ids` and `logprobs` share the same flat layout.
    """

    ids: Any
    logprobs: Any
    counts: Any

    @classmethod
    def from_top_logprobs(
        cls, ids: list[list[int]], logprobs: list[list[float]]
    ) -> "TopLogprobs":
        if len(ids) != len(logprobs):
            raise ValueError("top ids and logprobs must have one row per token")
        counts = np.fromiter(
            (len(row) for row in ids),
            dtype=np.int32,
            count=len(ids),
        )
        if any(len(ids[i]) != len(logprobs[i]) for i in range(len(ids))):
            raise ValueError("top ids and logprobs rows must be the same length")
        return cls(
            ids=np.fromiter((token for row in ids for token in row), dtype=np.int32),
            logprobs=np.fromiter(
                (p for row in logprobs for p in row), dtype=np.float32
            ),
            counts=counts,
        )


class TurnTokens(BaseModel):
    """Training tokens from renderer tokenization or provider-returned token IDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: IntStream = Field(default_factory=empty_stream(np.int32))
    """Effective prompt IDs evaluated by the model, after multimodal expansion."""
    completion_ids: IntStream = Field(default_factory=empty_stream(np.int32))
    completion_logprobs: FloatStream = Field(default_factory=empty_stream(np.float32))

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
    is_content: BoolStream | None = Field(default=None, exclude=True)
    # Authoritative effective-prompt ranges returned by vLLM, flattened across modalities.
    mm_placeholders: list[tuple[int, int]] | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): the renderer's special-token id -> modality marker map,
    # stamped onto `Trace.mm_token_type_id_map` by the turn's `commit`. None unless the
    # rendering renderer is multimodal.
    mm_token_type_id_map: dict[int, int] | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): the MoE expert-routing data from `generate` (expert ids
    # per token), attributed per node by the turn's `commit` into `MessageNode.routed_experts`,
    # then dropped. None unless the engine ran with `enable_return_routed_experts`.
    routed_experts: RoutedExperts | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): per-completion-token sampling masks,
    # attributed to the assistant node by the turn's `commit`, then dropped.
    sampling_mask: SamplingMask | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): the engine's weight-version spans over the completion,
    # attributed to the assistant node by the turn's `commit`, then dropped.
    weight_versions: list[WeightVersionSpan] | None = Field(default=None, exclude=True)

    # Transient carrier (excluded): per-completion-token top-k sampling heads,
    # attributed to the assistant node by the turn's `commit`, then dropped.
    top_logprobs: TopLogprobs | None = Field(default=None, exclude=True)

    @field_serializer("prompt_ids", "completion_ids", "completion_logprobs")
    def serialize_stream(self, values: list | np.ndarray) -> list:
        return as_list(values)


class Response(BaseModel):
    id: str
    created: int
    model: str
    message: AssistantMessage
    finish_reason: FinishReason
    usage: Usage | None = None
    tokens: TurnTokens | None = None
    raw: dict | None = Field(default=None, exclude=True, repr=False)
    """Full native response object returned to the program; excluded from traces."""
