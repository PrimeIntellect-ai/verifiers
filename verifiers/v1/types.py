"""Provider-agnostic wire types: messages, tools, model responses.

These are the only types that cross the boundary between verifiers and a model
provider. They are pydantic models — never dicts with normalizers. The single
place raw provider dicts enter is the client implementation, which validates
them into these models explicitly.
"""

from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


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


class WireTensor(StrictBaseModel):
    """A tensor on the wire: dtype + shape + base64 of the raw buffer. JSON-native, so it
    survives `model_dump(mode="json")` + msgpack (a raw tensor would not)."""

    dtype: str
    shape: list[int]
    data: str  # base64 of arr.tobytes()


def _encode_wire_tensor(val: Any) -> WireTensor:
    """Encode a torch tensor / numpy array / already-encoded `{dtype,shape,data:bytes}`
    dict into a `WireTensor` (base64). Torch/numpy imported lazily — text-only callers
    never hit this."""
    import base64

    if isinstance(val, dict) and "data" in val and "dtype" in val:  # v0-wire encoded
        return WireTensor(
            dtype=str(val["dtype"]),
            shape=list(val["shape"]),
            data=base64.b64encode(bytes(val["data"])).decode(),
        )
    import numpy as np

    if hasattr(val, "detach"):  # torch tensor
        val = val.detach().cpu().contiguous().numpy()
    arr = np.ascontiguousarray(val)
    return WireTensor(
        dtype=str(arr.dtype),
        shape=list(arr.shape),
        data=base64.b64encode(arr.tobytes()).decode(),
    )


class MMData(StrictBaseModel):
    """Per-turn multimodal sidecar — the JSON-native mirror of `renderers.MultiModalData`.
    `mm_items[modality][i]` is the HF processor's per-image dict (e.g. `pixel_values`,
    `image_grid_thw`) with each tensor encoded as a `WireTensor`."""

    mm_items: dict[str, list[dict[str, WireTensor]]] = Field(default_factory=dict)
    mm_hashes: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def from_renderer(cls, mm: Any) -> "MMData | None":
        """Encode a `renderers.MultiModalData` (or an already-encoded dict) into a
        JSON-native `MMData`; None when there's no image/video payload."""
        if mm is None:
            return None
        items = getattr(mm, "mm_items", None)
        hashes = getattr(mm, "mm_hashes", None)
        if items is None and isinstance(mm, dict):
            items, hashes = mm.get("mm_items"), mm.get("mm_hashes")
        if not items:
            return None
        wire = {
            modality: [
                {key: _encode_wire_tensor(val) for key, val in item.items()}
                for item in per_modality
            ]
            for modality, per_modality in items.items()
        }
        return cls(mm_items=wire, mm_hashes=hashes or {})


class TurnTokens(StrictBaseModel):
    """Token ids + sampling logprobs for one response, for training. Populated by the
    renderer client (client-side tokenization) or the chat client (parsed from vLLM's
    token ids); None when the provider returns neither."""

    prompt_ids: list[int] = Field(default_factory=list)
    completion_ids: list[int] = Field(default_factory=list)
    completion_logprobs: list[float] = Field(default_factory=list)
    multi_modal_data: MMData | None = None
    """Per-turn image/video inputs (VLM training); None for text-only turns."""


class Response(StrictBaseModel):
    """One model completion, provider-agnostic."""

    id: str
    created: int
    model: str
    message: AssistantMessage
    finish_reason: FinishReason
    usage: Usage | None = None
    tokens: TurnTokens | None = None
    """Token ids + logprobs for training (renderer client, or chat client via vLLM)."""


# --- sampling -----------------------------------------------------------------


class SamplingConfig(BaseModel):
    """Typed sampling knobs; provider-specific keys pass through (extra='allow')."""

    model_config = ConfigDict(extra="allow")
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(
        None, validation_alias=AliasChoices("max_tokens", "max_completion_tokens")
    )
