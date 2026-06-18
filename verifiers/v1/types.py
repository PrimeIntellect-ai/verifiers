"""Provider-agnostic wire types: messages, tools, model responses.

These are the only types that cross the boundary between verifiers and a model
provider. They are pydantic models — never dicts with normalizers. The single
place raw provider dicts enter is the client implementation, which validates
them into these models explicitly.
"""

from collections.abc import Iterable
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, AliasChoices, BaseModel, ConfigDict, Field
from renderers.base import MultiModalData
from typing_extensions import TypedDict


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
    name: str | None = None
    """The originating tool/function name, recovered from the prompt's matching tool call.

    Most renderers key a tool result off `tool_call_id` alone, but some render the function name into
    the template (GPT-OSS Harmony emits `functions.<name>`, falling back to `functions.unknown`
    without it — which breaks token parity). The bridge makes this load-bearing: it renders only the
    new tail (e.g. `[tool, user]`), so the issuing assistant's tool call sits in the already-reused
    prefix and isn't re-sent — the name can't be recovered from the tail. So the dialect recovers it
    once while parsing the full prompt and attaches it here, where it rides along into later bridge tails."""


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
    """Token usage for one model response.

    ``prompt_tokens`` excludes cache-read tokens; ``cached_input_tokens`` carries that
    disjoint portion when the provider reports it. ``input_tokens`` and ``total_tokens``
    reconstruct the full logical sequence sizes. ``reasoning_tokens`` is an optional subset
    of ``completion_tokens`` and is not added again to totals. ``cost`` is the optional
    provider-reported cost for the response.
    """

    prompt_tokens: int
    completion_tokens: int
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = None

    @classmethod
    def aggregate(cls, usages: Iterable["Usage"]) -> "Usage | None":
        """Sum per-response usage while preserving whether cache usage was reported."""
        values = list(usages)
        if not values:
            return None
        cached = [
            usage.cached_input_tokens
            for usage in values
            if usage.cached_input_tokens is not None
        ]
        reasoning = [usage.reasoning_tokens for usage in values]
        costs = [usage.cost for usage in values]
        return cls(
            prompt_tokens=sum(usage.prompt_tokens for usage in values),
            completion_tokens=sum(usage.completion_tokens for usage in values),
            cached_input_tokens=sum(cached) if cached else None,
            reasoning_tokens=sum(reasoning)
            if all(v is not None for v in reasoning)
            else None,
            cost=sum(costs) if all(v is not None for v in costs) else None,
        )

    @property
    def input_tokens(self) -> int:
        return self.prompt_tokens + (self.cached_input_tokens or 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.completion_tokens


class RoutedExperts(TypedDict):
    """The raw MoE expert-routing data a `generate` response carries for router replay:
    base64 `data` (uint8 `[tokens, layers, top_k]`), its `shape`, and `start` — the prompt
    offset where the routing begins (0 = full prompt+completion). Kept opaque (`Any` data)
    so pydantic never validates the encoded blob."""

    data: Any
    shape: list[int]
    start: int


class TurnTokens(StrictBaseModel):
    """Token ids + sampling logprobs for one response, for training. Populated by the
    renderer client (client-side tokenization) or the chat client (parsed from vLLM's
    token ids); None when the provider returns neither."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    prompt_ids: list[int] = Field(default_factory=list)
    completion_ids: list[int] = Field(default_factory=list)
    completion_logprobs: list[float] = Field(default_factory=list)

    # Transient carrier (excluded): per-message token spans into `prompt_ids` from the renderer,
    # consumed by the turn's `commit` to attribute tokens per message, then dropped.
    message_spans: list[tuple[int, int] | None] | None = Field(
        default=None, exclude=True
    )
    # Transient carrier (excluded): the renderer's multimodal sidecar (raw-image descriptors,
    # hashes, and placeholder offsets), attributed per node by the turn's `commit`, then dropped.
    multi_modal_data: MultiModalData | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): the MoE expert-routing data from `generate` (expert ids
    # per token), attributed per node by the turn's `commit` into `MessageNode.routed_experts`,
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


Sampling = SamplingConfig
"""Alias for `SamplingConfig` — the terse name for a `sampling` field/arg."""


# --- ids ----------------------------------------------------------------------


def _validate_env_id(env_id: str) -> str:
    """Validate the id's shape — a hub id must be a well-formed ``org/name[@version]``; a
    local id is any module name. Returns it unchanged (the value stays a plain ``str``)."""
    from verifiers.utils.install_utils import is_hub_env, parse_env_id

    if is_hub_env(env_id):
        parse_env_id(env_id)  # raises ValueError on a malformed org/name[@version]
    return env_id


EnvId = Annotated[str, AfterValidator(_validate_env_id)]
"""A taskset / harness / environment id — ``name``, ``org/name``, or ``org/name@version``. A
plain validated ``str``; derive its package/module name with `env_name` / `env_module`
(`verifiers.v1.utils.install`)."""
