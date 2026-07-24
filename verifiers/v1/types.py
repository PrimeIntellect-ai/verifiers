from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, AliasChoices, BaseModel, ConfigDict, Field
from renderers.base import MultiModalData
from typing_extensions import TypedDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextContentPart(StrictBaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrlSource(StrictBaseModel):
    url: str


class ImageUrlContentPart(StrictBaseModel):
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


class SystemMessage(StrictBaseModel):
    role: Literal["system"] = "system"
    content: MessageContent


class UserMessage(StrictBaseModel):
    role: Literal["user"] = "user"
    content: MessageContent


class ToolCall(StrictBaseModel):
    id: str
    name: str
    arguments: str
    """Raw JSON string of arguments, exactly as the model emitted it."""


class AssistantMessage(StrictBaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    provider_state: list[dict[str, Any]] | None = None
    """Opaque native items replayed to preserve signed or encrypted reasoning state."""


class ToolMessage(StrictBaseModel):
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


class Tool(StrictBaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool | None = None


FinishReason = Literal["stop", "length", "tool_calls"] | None


class Usage(StrictBaseModel):
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


@dataclass
class KeptTokens:
    """Kept-set sampling masks for sampling replay: `ids` (every kept set concatenated
    in position order) and `counts` (kept-set size per completion token; 0 = no usable
    mask). Base64 blobs straight off the `generate` response on the `TurnTokens`
    carrier; decoded to flat int32 arrays on `MessageNode` (`len(ids) == sum(counts)`,
    row boundaries recovered from `counts`)."""

    ids: Any
    counts: Any


class TurnTokens(StrictBaseModel):
    """Training tokens from renderer tokenization or provider-returned token IDs."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    prompt_ids: list[int] = Field(default_factory=list)
    completion_ids: list[int] = Field(default_factory=list)
    completion_logprobs: list[float] = Field(default_factory=list)

    # Transient carrier (excluded): per-message token spans into `prompt_ids` from the renderer,
    # consumed by the turn's `commit` to attribute tokens per message, then dropped.
    message_spans: list[tuple[int, int] | None] | None = Field(
        default=None, exclude=True
    )
    is_content: list[bool] | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): the renderer's multimodal sidecar (raw-image descriptors,
    # hashes, and placeholder offsets), attributed per node by the turn's `commit`, then dropped.
    multi_modal_data: MultiModalData | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): the MoE expert-routing data from `generate` (expert ids
    # per token), attributed per node by the turn's `commit` into `MessageNode.routed_experts`,
    # then dropped. None unless the engine ran with `enable_return_routed_experts`.
    routed_experts: RoutedExperts | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): the kept-set sampling masks from `generate` (token ids
    # surviving top-p/top-k truncation, per completion token), attributed to the assistant
    # node by the turn's `commit`, then dropped. None unless the engine ran with
    # `enable_return_kept_tokens`.
    kept_tokens: KeptTokens | None = Field(default=None, exclude=True)


class Response(StrictBaseModel):
    id: str
    created: int
    model: str
    message: AssistantMessage
    finish_reason: FinishReason
    usage: Usage | None = None
    tokens: TurnTokens | None = None
    raw: dict | None = Field(default=None, exclude=True, repr=False)
    """Full native response object returned to the program; excluded from traces."""


class SamplingConfig(BaseModel):
    """Typed sampling knobs; provider-specific keys pass through (extra='allow')."""

    model_config = ConfigDict(extra="allow")
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None
    max_tokens: int | None = Field(
        None, validation_alias=AliasChoices("max_tokens", "max_completion_tokens")
    )


Sampling = SamplingConfig


def _validate_id(plugin_id: str) -> str:
    from verifiers.utils.install_utils import is_hub_env, parse_env_id

    if is_hub_env(plugin_id):
        parse_env_id(plugin_id)  # raises ValueError on a malformed org/name[@version]
    return plugin_id


ID = Annotated[str, AfterValidator(_validate_id)]
"""Plugin id: `name`, `org/name`, or `org/name@version`."""
