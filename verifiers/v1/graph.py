"""Message-graph trajectory: store each message once, recover branches by walking.

A rollout is a graph of `MessageNode`s — one per distinct message, each linked to its
predecessor. A conversation is a path from a root to a leaf; prompt divergence from
compaction, subagents, or other history rewrites produces multiple leaves, so branching
falls out of the walk. A branch often corresponds to one harness context window, but it is
a physical, exact-prefix training view rather than a semantic context identity: a prefix
break can split one context and prefix reuse can preserve an ancestral path. Each node
stores only the tokens it *adds* to the cumulative sequence, keeping size linear in turns
and making a branch's training sample a cheap concat of node
`token_ids`/`mask`/`logprobs` along its path.

Token attribution (renderer client): the renderer reports, per prompt, each message's token
span (`RenderedTokens.message_token_spans()`, carried on `TurnTokens.message_spans`). A new
input message's node gets its span plus the leading template scaffold since the previous
message; the trailing scaffold (the generation prompt) goes on the assistant node, prefixed
to its sampled completion. By construction `concat(node.token_ids along a path)` reproduces
the exact `prompt_ids + completion_ids` the model saw.
"""

from __future__ import annotations

import binascii
import hashlib
import json
import time
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    ValidationInfo,
    field_serializer,
    field_validator,
)
from pydantic.json_schema import SkipJsonSchema
from renderers.base import RenderedTokens

from verifiers.v1.configs.harness import DialectName
from verifiers.v1.semantic import ParentLink
from verifiers.v1.types import (
    AssistantMessage,
    BoolStream,
    FloatStream,
    IntStream,
    Message,
    Response,
    SamplingMask,
    TextContentPart,
    Tool,
    ToolMessage,
    TopLogprobs,
    WeightVersionSpan,
    as_list,
    decode_ndarray,
    empty_stream,
    encode_ndarray,
)

if TYPE_CHECKING:
    from verifiers.v1.trace import Trace


RECORD_FLOAT_DECIMALS = 4
"""Default precision of per-token float streams in JSON records (`to_record`). Full-precision
digits are noise to every record reader and the least compressible bytes of a trace; four
decimals leave a logprob within 1e-4 of what the trainer saw. `to_record(float_decimals=...)`
overrides it per dump (`None` keeps every digit). The msgpack wire (`mode="python"`) keeps
full precision — training never reads the record."""


def same_tokens(a: list | np.ndarray, b: list | np.ndarray) -> bool:
    """Two token spans are equal (`==` on an array compares elementwise)."""
    return len(a) == len(b) and bool(np.array_equal(a, b))


ROUNDED_STREAMS = (
    "logprobs",
    "advantages",
    "reference_logprobs",
    "trainer_logprobs",
    "entropies",
)
"""Float streams a JSON record rounds to `float_decimals`. Loss weights stay exact: a 1/T
weight would round to zero."""


def _serialize_stream(
    values: list | np.ndarray, info: FieldSerializationInfo, rounded: bool
) -> list | dict:
    """JSON records: a list. Python dumps: a list too, unless the context sets `token_arrays`,
    when the msgpack wire carries a raw-bytes `__nd__` dict."""
    context = info.context or {}
    if info.mode_is_json():
        values = as_list(values)
        decimals = context.get("float_decimals", RECORD_FLOAT_DECIMALS)
        if not rounded or decimals is None:
            return values
        return [round(value, decimals) for value in values]
    if context.get("token_arrays"):
        return encode_ndarray(np.asarray(values))
    return as_list(values)


class MessageNode(BaseModel):
    """One message in the graph: a message plus the tokens it adds to the cumulative
    sequence. Concatenating a root→leaf path's nodes reconstructs that branch's full token
    sequence; the mask/logprobs make it a training sample."""

    parent: int | None = None
    """Index into `Trace.nodes` of the predecessor message; None for a root."""
    tools: list[Tool] = Field(default_factory=list, exclude_if=lambda tools: not tools)
    """Tools rendered into this branch's prompt. Populated only on root nodes."""
    semantic_parents: list[ParentLink] = Field(default_factory=list)
    """Typed incoming edges from other turns, declared by the harness (for example
    ``subagent_return`` or ``subagent_cancel``).

    They record causal links, not ancestry: unlike ``parent`` they imply no token prefix,
    do not affect physical branch construction, and may form loops. A list permits multiple parents of the same
    type, supports incremental appends, and preserves their advertised wire order; edge
    application prevents duplicate ``(node, type)`` links.
    """
    message: Message
    """The message this node carries (system / user / assistant / tool)."""
    sampled: bool = False
    """True iff a model call produced this message (the response passed to `commit`); False for
    every prompt-supplied message — including assistant/tool messages fabricated as context
    the model never generated, which role alone can't tell apart from real turns."""
    timestamp: float = Field(default_factory=time.time)
    """Wall-clock epoch seconds when this node was created. Nodes materialize at turn commit,
    so a turn's new input nodes and its assistant node carry (near-)identical stamps and the
    delta between consecutive sampled nodes is that turn's harness + inference wall-clock.
    Reused prefix nodes keep the stamp from the turn that first created them. Serialized, so
    a dump re-validated from wire/disk keeps the original times."""
    token_ids: IntStream = Field(default_factory=empty_stream(np.int32))
    """This message's delta contribution to the cumulative token sequence: its leading
    template scaffold + its own tokens — for an assistant, the generation-prompt scaffold
    followed by the sampled completion. Concatenated along a path, these reproduce the exact
    `prompt_ids + completion_ids` the model saw."""
    renderer_token_ids: IntStream | None = Field(default=None, exclude=True)
    """Logical renderer tokens retained only while extending a live rollout.
    None means they are identical to `token_ids`; an empty array is a real empty slice."""

    mask: BoolStream = Field(default_factory=empty_stream(np.bool_))
    """Per-token, parallel to `token_ids`: True for trainable, model-sampled tokens (only an
    assistant node's completion span); False for template scaffold and every input-message
    token."""
    is_content: BoolStream = Field(default_factory=empty_stream(np.bool_))
    """Per-token, parallel to `token_ids` (when populated): True for message-body tokens (the
    renderer's content), False for template scaffold (role-tag openers/closers, inter-turn
    separators, tool-response wraps, the generation prompt). Populated from the renderer's
    `RenderedTokens.is_content`; empty when the renderer doesn't attribute content (e.g. the
    default Jinja renderer) or for relay (eval) turns that carry no token ids. Distinct from
    `mask`: `mask` is "did the model sample this?" (assistant completion only); `is_content`
    is "is this caller/model body vs scaffold?" — meaningful on every role, so observation
    weighting (prime-rl `echo`) can train a tool/user message's *body* without its scaffold."""
    logprobs: FloatStream = Field(default_factory=empty_stream(np.float32))
    """Sampling logprobs for the sampled tokens — length equals the number of True entries in
    `mask`; empty for input messages."""
    advantages: FloatStream | None = None
    """Per-token credit over the sampled tokens, same layout as `logprobs`. `None` until a
    consumer's RL algorithm assigns it, which is not the same as a credit of zero: a group whose
    rewards were all equal is assigned zeros and carries no gradient, while an unassigned node was
    never scored at all."""
    reference_logprobs: FloatStream | None = None
    """Reference-model logprobs over the sampled tokens, in the same compact layout as
    `logprobs`. None means no reference model scored this node."""
    trainer_logprobs: FloatStream | None = None
    """Trainer-recomputed logprobs over the sampled tokens, in the same compact layout as
    `logprobs`. None means no trainer forward annotated this node."""
    entropies: FloatStream | None = None
    """Trainer policy entropies over the sampled tokens, in the same compact layout as
    `logprobs`. None means no trainer forward annotated this node."""
    loss_weights: dict[str, FloatStream] | None = None
    """Named loss-weight streams aligned to `token_ids`, consumer-stamped."""
    routed_experts: SkipJsonSchema[np.ndarray | None] = None
    """This node's slice of the MoE expert-routing array — uint8 `[len(token_ids), layers,
    top_k]`, the expert ids inference selected for exactly this node's tokens. Attributed from
    the turn's `generate` payload by `_attribute_routed_experts`; `Branch.routed_experts`
    concatenates these along the path into the trainer's router-replay input. Rides the wire as
    a raw-bytes `__nd__` dict; kept off disk by the dump-site `exclude` in prime-rl."""
    sampling_mask: SkipJsonSchema[SamplingMask | None] = None
    """Sampling masks for this node's sampled tokens.

    `ids` stores the flat token ids and `counts` stores each token's row size. Assistant
    nodes only. The arrays serialize as raw-byte `__nd__` dictionaries.
    """
    weight_versions: list[WeightVersionSpan] | None = None
    """Engine weight-version spans over this node's sampled tokens (indexes into the
    completion, as vLLM returns them). Assistant nodes only; None when the engine labels
    none."""

    top_logprobs: SkipJsonSchema[TopLogprobs | None] = None
    """Sampler top-k probabilities for sampled tokens; unnormalized over the head."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def logical_ids(self) -> np.ndarray:
        return (
            self.token_ids
            if self.renderer_token_ids is None
            else self.renderer_token_ids
        )

    @field_serializer("token_ids", "mask", "is_content", *ROUNDED_STREAMS)
    def serialize_token_stream(
        self, values: list | np.ndarray | None, info: FieldSerializationInfo
    ) -> list | dict | None:
        if values is None:
            return None
        return _serialize_stream(values, info, info.field_name in ROUNDED_STREAMS)

    @field_serializer("loss_weights")
    def serialize_loss_weights(
        self, streams: dict[str, list | np.ndarray] | None, info: FieldSerializationInfo
    ) -> dict | None:
        if streams is None:
            return None
        return {
            name: _serialize_stream(values, info, rounded=False)
            for name, values in streams.items()
        }

    @field_serializer("routed_experts")
    def serialize_ndarray_field(self, arr: np.ndarray | None) -> dict | None:
        """Integer array -> raw-bytes `__nd__` dict so it rides the wire (numpy can't JSON)."""
        return None if arr is None else encode_ndarray(arr)

    @field_validator("routed_experts", mode="before")
    @classmethod
    def deserialize_ndarray_field(cls, value: Any) -> np.ndarray | None:
        if value is None or isinstance(value, np.ndarray):
            return value
        if isinstance(value, dict) and value.get("__nd__"):
            return decode_ndarray(value)
        raise TypeError(f"cannot build ndarray field from {type(value).__name__}")

    @field_serializer("sampling_mask", "top_logprobs")
    def serialize_sampling_arrays(
        self, payload: SamplingMask | TopLogprobs | None
    ) -> dict | None:
        if payload is None:
            return None
        return {
            field.name: encode_ndarray(getattr(payload, field.name))
            for field in fields(payload)
        }

    @field_validator("sampling_mask", "top_logprobs", mode="before")
    @classmethod
    def deserialize_sampling_arrays(
        cls, value: Any, info: ValidationInfo
    ) -> SamplingMask | TopLogprobs | None:
        kind = SamplingMask if info.field_name == "sampling_mask" else TopLogprobs
        if value is None or isinstance(value, kind):
            return value
        if isinstance(value, dict):
            return kind(**{key: decode_ndarray(array) for key, array in value.items()})
        raise TypeError(f"cannot build {kind.__name__} from {type(value).__name__}")


def _canonical_tool_arguments(arguments: str) -> str:
    # Ignore JSON key order and whitespace when hashing equivalent tool calls.
    try:
        return json.dumps(json.loads(arguments), sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, ValueError):
        return arguments


# Provider-specific fields not represented by typed messages but required on replay.
_PROVIDER_STATE_FIELDS = frozenset({"encrypted_content", "signature", "data", "phase"})


def message_hash(message: Message) -> str:
    """Stable content hash on the fields that round-trip through a prompt — role, content
    (None and "" equal), assistant reasoning content when present, assistant tool calls,
    opaque continuation state, tool call id. Two messages hash equal iff they're the same
    conversational message, so a re-stated prefix message dedups to one node. The message part
    of the prefix key; salt-free so it is identical across processes and after deserialization."""
    digest = hashlib.blake2b(digest_size=16)

    def add(value: str) -> None:
        data = value.encode()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    add(type(message).__name__)
    if isinstance(message.content, list):
        add("content_parts")
        for part in message.content:
            add(part.type)
            if isinstance(part, TextContentPart):
                add(part.text)
            else:
                add(part.image_url.url)
    else:
        add("content_text")
        add(message.content or "")
    if isinstance(message, AssistantMessage):
        if message.reasoning_content is not None:
            add("reasoning_content")
            add(message.reasoning_content)
        for item in message.provider_state or []:
            kind = item.get("type") or (
                "message" if item.get("role") == "assistant" else ""
            )
            hashed_state = {
                key: item[key]
                for key in _PROVIDER_STATE_FIELDS
                if item.get(key) is not None
            }
            if kind == "message" and isinstance(item.get("content"), list):
                # Keep content parts the typed message does not expose, such as refusals.
                unparsed_content = [
                    part
                    for part in item.get("content") or []
                    if part.get("type") not in ("input_text", "output_text")
                ]
                if unparsed_content:
                    hashed_state["content"] = unparsed_content
            represented = kind in ("message", "reasoning") or (
                kind in ("function_call", "custom_tool_call")
                and any(
                    call.id == item.get("call_id") for call in message.tool_calls or []
                )
            )
            if represented and not hashed_state:
                continue
            # Unknown provider items still distinguish built-in calls and actions.
            state = hashed_state if represented else item
            add("provider_state")
            add(kind)
            add(json.dumps(state, sort_keys=True))
        for tc in message.tool_calls or []:
            add("tool_call")
            add(tc.type)
            add(tc.id)
            add(tc.name)
            add(tc.namespace or "")
            add(
                tc.arguments
                if tc.type == "custom"
                else _canonical_tool_arguments(tc.arguments)
            )
    elif isinstance(message, ToolMessage):
        add("tool_call_id")
        add(message.tool_call_id)
    return digest.hexdigest()


def _tools_hash(tools: list[Tool] | None) -> str:
    payload = [tool.model_dump(mode="json", exclude_none=True) for tool in tools or []]
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _message_key(trace: Trace, message: Message, dialect: DialectName | None) -> str:
    """`message_hash` of the message as the trace's harness replays it over `dialect`,
    so a turn the harness re-sends with cosmetic rewrites still matches the node it
    came from."""
    view = trace._replay_view
    return message_hash(view(message, dialect) if view is not None else message)


def _node_key(
    trace: Trace,
    parent: int | None,
    message: Message,
    tools: list[Tool] | None = None,
    *,
    dialect: DialectName | None,
) -> tuple[int | None, str | None, str]:
    return (
        parent,
        _tools_hash(tools) if parent is None else None,
        _message_key(trace, message, dialect),
    )


def _head_index(
    trace: Trace, dialect: DialectName | None
) -> dict[tuple[int | None, str | None, str], int]:
    """Physical node key -> id under `dialect`'s replay view, built lazily per dialect
    (and after deserialization)."""
    index = trace._head_index.get(dialect)
    if index is None:
        index = trace._head_index[dialect] = {
            _node_key(
                trace, node.parent, node.message, node.tools, dialect=dialect
            ): nid
            for nid, node in enumerate(trace.nodes)
        }
    return index


def _index_node(trace: Trace, node_id: int) -> None:
    """Point every built dialect index at a new node."""
    node = trace.nodes[node_id]
    for dialect, index in trace._head_index.items():
        key = _node_key(trace, node.parent, node.message, node.tools, dialect=dialect)
        index[key] = node_id


def message_prefix_len(
    trace: Trace, prompt: list[Message], dialect: DialectName | None = None
) -> int:
    """Length of the longest message-only graph prefix matching `prompt`."""
    children: dict[int | None, list[int]] = {}
    for node_id, node in enumerate(trace.nodes):
        children.setdefault(node.parent, []).append(node_id)

    parents: list[int | None] = [None]
    matched = 0
    for message in prompt:
        key = _message_key(trace, message, dialect)
        parents = [
            node_id
            for parent in parents
            for node_id in children.get(parent, [])
            if _message_key(trace, trace.nodes[node_id].message, dialect) == key
        ]
        if not parents:
            break
        matched += 1
    return matched


def _matching_node(
    trace: Trace,
    parent: int | None,
    message: Message,
    token_ids: list[int] | None = None,
    renderer_token_ids: list[int] | None = None,
    tools: list[Tool] | None = None,
    *,
    dialect: DialectName | None,
) -> int | None:
    """Find an existing child, optionally requiring its exact token spans.

    The head index deliberately points at only the latest content-equivalent child. Token-level
    prefix breaks can leave older physical variants under the same key, so a token mismatch falls
    back to a reverse scan rather than materializing a duplicate of an already-existing variant.
    """
    key = _node_key(trace, parent, message, tools, dialect=dialect)
    indexed = _head_index(trace, dialect).get(key)
    if (
        indexed is not None
        and (
            token_ids is None or same_tokens(trace.nodes[indexed].token_ids, token_ids)
        )
        and (
            renderer_token_ids is None
            or same_tokens(trace.nodes[indexed].logical_ids, renderer_token_ids)
        )
    ):
        return indexed
    if token_ids is None and renderer_token_ids is None:
        return indexed
    for node_id in range(len(trace.nodes) - 1, -1, -1):
        if node_id == indexed:
            continue
        node = trace.nodes[node_id]
        if (
            (token_ids is None or same_tokens(node.token_ids, token_ids))
            and (
                renderer_token_ids is None
                or same_tokens(node.logical_ids, renderer_token_ids)
            )
            and _node_key(trace, node.parent, node.message, node.tools, dialect=dialect)
            == key
        ):
            return node_id
    return None


def _matching_prefix_node(
    trace: Trace,
    parent: int | None,
    message: Message,
    prompt_ids: list[int],
    start: int,
    stop: int,
    renderer_prompt_ids: list[int],
    renderer_start: int,
    renderer_stop: int,
    tools: list[Tool] | None = None,
    *,
    dialect: DialectName | None,
) -> int | None:
    """Find the longest content-equivalent child matching inside `[start, stop]`.

    Renderers may leave a prompt-supplied assistant unattributed (`message_spans[i] is None`).
    Its existing sampled node still owns physical tokens, so use those tokens to recover the
    otherwise-missing message boundary. The next attributed message's start bounds the match:
    a sampled variant must never consume tokens that the renderer assigned to that message.
    """
    key = _node_key(trace, parent, message, tools, dialect=dialect)
    indexed = _head_index(trace, dialect).get(key)
    candidates: list[int] = []
    if indexed is not None:
        candidates.append(indexed)
    candidates.extend(
        node_id
        for node_id in range(len(trace.nodes) - 1, -1, -1)
        if node_id != indexed
        and _node_key(
            trace,
            trace.nodes[node_id].parent,
            trace.nodes[node_id].message,
            trace.nodes[node_id].tools,
            dialect=dialect,
        )
        == key
    )
    matches = [
        node_id
        for node_id in candidates
        if start + len(trace.nodes[node_id].token_ids) <= stop
        and same_tokens(
            prompt_ids[start : start + len(trace.nodes[node_id].token_ids)],
            trace.nodes[node_id].token_ids,
        )
        and renderer_start + len(trace.nodes[node_id].logical_ids) <= renderer_stop
        and same_tokens(
            renderer_prompt_ids[
                renderer_start : renderer_start + len(trace.nodes[node_id].logical_ids)
            ],
            trace.nodes[node_id].logical_ids,
        )
    ]
    return max(
        matches, key=lambda node_id: len(trace.nodes[node_id].token_ids), default=None
    )


@dataclass(frozen=True)
class PendingTurn:
    """A resolved prompt waiting on model inference.

    `prepare_turn` resolves the graph prefix used for renderer bridging before inference. `commit`
    preserves that inference-producing prefix, extends it with any matching nodes committed while
    inference was in flight, then adds only the remaining prompt tail and sampled response.
    """

    trace: Trace
    prompt: list[Message]
    tools: list[Tool]
    prefix_node_ids: list[int]
    path_len: int
    dialect: DialectName | None = None
    """The wire format of the request, whose replay view matches its prompt."""

    @property
    def tail_start(self) -> int:
        return len(self.prefix_node_ids)

    @property
    def tail(self) -> list[Message]:
        return self.prompt[self.tail_start :]

    def previous_renderer_token_ids(self) -> tuple[list[int], list[int]] | None:
        """Return the logical renderer prompt and completion for a bridge anchor.

        The anchor must end at a sampled assistant node. That node stores generation-prompt
        scaffold followed by sampled completion tokens, so split off the sampled suffix.
        """
        if not self.prefix_node_ids:
            return None
        last = self.trace.nodes[self.prefix_node_ids[-1]]
        if not last.sampled:
            return None
        num_sampled = int(np.count_nonzero(last.mask))
        if not num_sampled:
            return None

        # The renderer bridges over lists, so the stream converts once here.
        ids = np.concatenate(
            [self.trace.nodes[nid].logical_ids for nid in self.prefix_node_ids]
        )
        if len(ids) <= num_sampled:
            return None
        return ids[:-num_sampled].tolist(), ids[-num_sampled:].tolist()

    def prompt_message_spans(
        self, tail_attribution: RenderedTokens
    ) -> list[tuple[int, int] | None]:
        """Convert bridge-tail attribution into full-prompt message spans."""
        # Reused bridge tokens are unattributed, so scan only the newly rendered tail.
        tail_spans = RenderedTokens(
            message_indices=tail_attribution.message_indices[self.renderer_path_len :],
            message_roles=tail_attribution.message_roles,
        ).message_token_spans()
        # Tail spans are slice-relative; restore their full-prompt token offsets.
        return [None] * self.tail_start + [
            None
            if span is None
            else (span[0] + self.renderer_path_len, span[1] + self.renderer_path_len)
            for span in tail_spans
        ]

    @property
    def renderer_path_len(self) -> int:
        return sum(
            len(self.trace.nodes[nid].logical_ids) for nid in self.prefix_node_ids
        )

    def commit(self, response: Response) -> int:
        """Add this turn to the graph; returns the committed assistant node's id."""
        assistant_id = _commit_turn(self, response)
        if self.tools:
            self.trace.tools = list(
                {
                    (tool.namespace, tool.name, tool.type): tool
                    for tool in [*self.trace.tools, *self.tools]
                }.values()
            )
        self.trace.clear_preview(self)
        return assistant_id

    def abandon(self) -> None:
        """The request failed or was cancelled before a commit: its preview goes."""
        self.trace.clear_preview(self)

    def commit_prompt(self) -> None:
        """Record an input that terminated before model inference."""
        parent = self.prefix_node_ids[-1] if self.prefix_node_ids else None
        _head_index(self.trace, self.dialect)
        for message in self.tail:
            existing = _matching_node(
                self.trace, parent, message, tools=self.tools, dialect=self.dialect
            )
            if existing is not None:
                parent = existing
                continue
            self.trace.nodes.append(
                MessageNode(
                    parent=parent,
                    message=message,
                    tools=self.tools if parent is None else [],
                )
            )
            parent = len(self.trace.nodes) - 1
            _index_node(self.trace, parent)
        if self.tools:
            self.trace.tools = list(
                {
                    (tool.namespace, tool.name, tool.type): tool
                    for tool in [*self.trace.tools, *self.tools]
                }.values()
            )
        self.trace.clear_preview(self)


def prepare_turn(
    trace: Trace,
    prompt: list[Message],
    tools: list[Tool] | None = None,
    dialect: DialectName | None = None,
) -> PendingTurn:
    """Resolve a physical message-and-tools prefix without mutating the trace. `dialect`
    is the request's wire format, whose replay view matches the prompt to stored nodes."""
    idx = _head_index(trace, dialect)
    tools = list(tools or [])
    tools_hash = _tools_hash(tools)
    parent: int | None = None
    path_len = 0
    prefix_node_ids: list[int] = []
    for msg in prompt:
        existing = None
        if (
            isinstance(msg.content, list)
            and len(idx) <= 10
            and any(part.type == "image_url" for part in msg.content)
        ):
            children = [
                node_id
                for (node_parent, node_tools_hash, _), node_id in idx.items()
                if node_parent == parent
                and (parent is not None or node_tools_hash == tools_hash)
            ]
            # Repeated image URLs are cheaper to compare than to encode and hash again.
            # Only scan short, unambiguous parents; all other cases use the stable index.
            if len(children) == 1 and trace.nodes[children[0]].message == msg:
                existing = children[0]
        if existing is None:
            message_key = _message_key(trace, msg, dialect)
            existing = idx.get(
                (parent, tools_hash if parent is None else None, message_key)
            )
        if existing is None:
            break
        prefix_node_ids.append(existing)
        parent = existing
        path_len += len(trace.nodes[existing].token_ids)
    return PendingTurn(
        trace=trace,
        prompt=prompt,
        tools=tools,
        prefix_node_ids=prefix_node_ids,
        path_len=path_len,
        dialect=dialect,
    )


def _replace_placeholder_routing_row(
    trace: Trace, prefix_node_ids: list[int], arr: np.ndarray, off: int
) -> None:
    """Replace the prefix's placeholder routing row with the one this turn's prefill forwarded."""
    if not 1 <= off <= arr.shape[0]:
        return
    # Only assistant (`sampled`) nodes are affected: the model forward isn't run on  the final
    # generated token for such turns, meaning its routing decisions are fundamentally unavailable.
    # But, because routing needs one row per token, that row instead receives an inaccurate
    # placeholder, attempt to fix up below.
    node_with_placeholder = None
    for nid in reversed(prefix_node_ids):
        if len(trace.nodes[nid].token_ids):
            node_with_placeholder = trace.nodes[nid]
            break
    if node_with_placeholder is None or not node_with_placeholder.sampled:
        return
    node_rows = node_with_placeholder.routed_experts
    if (
        node_rows is None
        or node_rows.shape[0] == 0
        or node_rows.shape[1:] != arr.shape[1:]
    ):
        return
    # Row `i` of this turn's array is sequence position `start + i`, so the prefix's final position
    # is `arr[off - 1]`. Concatenating widens the node when this turn serialized `uint16`, where an
    # in-place write would truncate.
    node_with_placeholder.routed_experts = np.concatenate(
        [node_rows[:-1], arr[off - 1 : off]], axis=0
    )


def _attribute_routed_experts(
    trace: Trace,
    prefix_node_ids: list[int],
    new_node_ids: list[int],
    path_len: int,
    payload: Any,
) -> None:
    """Attach each new node's slice of this turn's MoE expert-routing array. The `generate`
    payload's array covers the turn's prompt+completion from `payload["start"]` (0 = from token
    0); the nodes created this turn tile sequence positions `[path_len:]` in creation order, so
    we hand each node `arr[off : off+len(node.token_ids)]` and advance. Reused-prefix nodes keep
    the routing attributed when they were first created, except for the one position this turn
    corrects (see `_replace_placeholder_routing_row`). A node whose slice falls outside the
    array (a `start` past `path_len`, e.g. an unexpected prefix-cache delta) is left unset — the
    branch then reports no routing rather than misaligning."""
    if payload is None:
        return
    raw = binascii.a2b_base64(payload["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(payload.get("dtype", "uint8"))).reshape(
        payload["shape"]
    )
    off = path_len - int(payload.get("start", 0) or 0)
    _replace_placeholder_routing_row(trace, prefix_node_ids, arr, off)
    needed = off + sum(len(trace.nodes[nid].token_ids) for nid in new_node_ids)
    for nid in new_node_ids:
        n = len(trace.nodes[nid].token_ids)
        end = off + n
        if n and 0 <= off and end <= arr.shape[0]:
            # Own only this node's rows; a view would retain the turn's full-context array.
            trace.nodes[nid].routed_experts = arr[off:end].copy()
        elif n and arr.shape[0] and 0 <= off and end == needed == arr.shape[0] + 1:
            # No forward pass follows the turn's final position, so it gets a placeholder: a
            # copy of the previous row, appended to this node's slice of the array.
            trace.nodes[nid].routed_experts = np.concatenate(
                [arr[off:], arr[-1:]], axis=0
            )
        off = end


def _attribute_sampling_mask(
    trace: Trace, assistant_id: int, payload: SamplingMask | None
) -> None:
    """Attach a completion-aligned sampling mask to the assistant node."""
    if payload is None:
        return
    node = trace.nodes[assistant_id]
    if len(payload.counts) != np.count_nonzero(node.mask) or int(
        payload.counts.sum()
    ) != len(payload.ids):
        return
    node.sampling_mask = payload


def _project_prompt_attribution(
    renderer_prompt_ids: np.ndarray,
    prompt_ids: np.ndarray,
    mm_token_type_id_map: dict[int, int],
    mm_placeholders: list[tuple[int, int]] | None,
    message_spans: list[tuple[int, int] | None] | None,
    is_content: np.ndarray | None,
) -> tuple[list[tuple[int, int] | None] | None, np.ndarray | None]:
    """Project logical attribution using vLLM's multimodal placeholder ranges."""
    if renderer_prompt_ids is prompt_ids or same_tokens(
        renderer_prompt_ids, prompt_ids
    ):
        return message_spans, is_content
    if not mm_token_type_id_map or mm_placeholders is None:
        raise ValueError(
            "cannot align renderer and vLLM prompt tokens without multimodal placeholders"
        )

    # Each logical token covers one prompt token, or one placeholder's whole range.
    is_mm = np.isin(renderer_prompt_ids, list(mm_token_type_id_map))
    placeholders = np.asarray(mm_placeholders, dtype=np.int64).reshape(-1, 2)
    if np.count_nonzero(is_mm) > len(placeholders):
        raise ValueError(
            "vLLM multimodal placeholders do not align with renderer prompt"
        )
    if np.count_nonzero(is_mm) < len(placeholders):
        raise ValueError("renderer prompt does not align with vLLM prompt tokens")
    lengths = np.ones(len(renderer_prompt_ids), dtype=np.int64)
    lengths[is_mm] = placeholders[:, 1]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    if (placeholders[:, 1] < 1).any() or not np.array_equal(
        placeholders[:, 0], offsets[:-1][is_mm]
    ):
        raise ValueError(
            "vLLM multimodal placeholders do not align with renderer prompt"
        )
    if offsets[-1] != len(prompt_ids) or not np.array_equal(
        prompt_ids[offsets[:-1][~is_mm]], renderer_prompt_ids[~is_mm]
    ):
        raise ValueError("renderer prompt does not align with vLLM prompt tokens")

    projected_spans = None
    if message_spans is not None:
        projected_spans = []
        for span in message_spans:
            if span is None:
                projected_spans.append(None)
                continue
            start, end = span
            if not 0 <= start <= end <= len(renderer_prompt_ids):
                raise ValueError("message span exceeds renderer prompt tokens")
            projected_spans.append((int(offsets[start]), int(offsets[end])))

    projected_is_content = is_content
    if is_content is not None and len(is_content):
        if len(is_content) != len(renderer_prompt_ids):
            raise ValueError(
                "content attribution does not match renderer prompt tokens"
            )
        projected_is_content = np.repeat(is_content, lengths)
    return projected_spans, projected_is_content


def _attribute_top_logprobs(
    trace: Trace, assistant_id: int, payload: TopLogprobs | None
) -> None:
    """Attach a completion-aligned top-k sampling head to the assistant node."""
    if payload is None:
        return
    node = trace.nodes[assistant_id]
    if (
        len(payload.counts) != np.count_nonzero(node.mask)
        or (payload.counts < 0).any()
        or int(payload.counts.sum()) != len(payload.ids)
        or len(payload.ids) != len(payload.logprobs)
    ):
        raise ValueError("Top logprobs must align with every sampled node token")
    node.top_logprobs = payload


def _commit_turn(turn: PendingTurn, response: Response) -> int:
    trace = turn.trace
    prompt = turn.prompt
    tokens = response.tokens
    # Constant per renderer, so re-stamping every turn is idempotent.
    if tokens is not None and tokens.mm_token_type_id_map:
        trace.mm_token_type_id_map = tokens.mm_token_type_id_map
    prompt_ids = np.asarray(tokens.prompt_ids if tokens else (), dtype=np.int32)
    # Logical renderer ids differ from `prompt_ids` only after multimodal expansion; otherwise
    # the nodes keep none and read `token_ids` (`MessageNode.logical_ids`).
    renderer_prompt_ids = (
        np.asarray(tokens.renderer_prompt_ids, dtype=np.int32)
        if tokens and tokens.renderer_prompt_ids is not None
        else prompt_ids
    )
    expanded = renderer_prompt_ids is not prompt_ids
    renderer_spans = tokens.message_spans if tokens else None
    renderer_is_content = (
        np.asarray(tokens.is_content, dtype=np.bool_)
        if tokens and tokens.is_content is not None
        else None
    )
    _head_index(trace, turn.dialect)

    prefix = turn.prefix_node_ids
    path_len = turn.path_len
    renderer_path_len = turn.renderer_path_len
    if tokens is not None and prefix:
        keep = 0
        off = 0
        renderer_off = 0
        for nid in prefix:
            node = trace.nodes[nid]
            node_renderer_ids = node.logical_ids
            if not same_tokens(
                prompt_ids[off : off + len(node.token_ids)], node.token_ids
            ) or not same_tokens(
                renderer_prompt_ids[
                    renderer_off : renderer_off + len(node_renderer_ids)
                ],
                node_renderer_ids,
            ):
                break
            off += len(node.token_ids)
            renderer_off += len(node_renderer_ids)
            keep += 1
        if tokens.bridged and keep != len(prefix):
            raise ValueError(
                "vLLM prompt tokens do not exactly extend the stored rollout prefix"
            )
        prefix = prefix[:keep]
        path_len = off
        renderer_path_len = renderer_off
    spans, is_content = _project_prompt_attribution(
        renderer_prompt_ids,
        prompt_ids,
        trace.mm_token_type_id_map,
        tokens.mm_placeholders if tokens else None,
        renderer_spans,
        renderer_is_content,
    )
    has_is_content = is_content is not None and len(is_content) == len(prompt_ids)

    # A parallel request may have committed more of this prompt after `prepare_turn` resolved
    # the inference prefix. Reconcile that still-uncommitted tail now, one whole message at a
    # time. Exact token-span equality preserves intentional renderer-level forks; tokenless relay
    # turns use message identity. Because commit is synchronous, once this loop stops the rest of
    # the tail can be appended without another request interleaving.
    prefix = list(prefix)
    while len(prefix) < len(prompt):
        i = len(prefix)
        span = spans[i] if spans and i < len(spans) else None
        end = span[1] if span else path_len
        renderer_span = (
            renderer_spans[i] if renderer_spans and i < len(renderer_spans) else None
        )
        renderer_end = renderer_span[1] if renderer_span else renderer_path_len
        parent = prefix[-1] if prefix else None
        if tokens is not None and span is None:
            next_start = next(
                (
                    later_span[0]
                    for later_span in (spans[i + 1 :] if spans else [])
                    if later_span is not None
                ),
                len(prompt_ids),
            )
            next_renderer_start = next(
                (
                    later_span[0]
                    for later_span in (
                        renderer_spans[i + 1 :] if renderer_spans else []
                    )
                    if later_span is not None
                ),
                len(renderer_prompt_ids),
            )
            existing = _matching_prefix_node(
                trace,
                parent,
                prompt[i],
                prompt_ids,
                path_len,
                next_start,
                renderer_prompt_ids,
                renderer_path_len,
                next_renderer_start,
                tools=turn.tools,
                dialect=turn.dialect,
            )
            if existing is not None:
                end += len(trace.nodes[existing].token_ids)
                renderer_end += len(trace.nodes[existing].logical_ids)
        else:
            node_tokens = prompt_ids[path_len:end]
            renderer_node_tokens = renderer_prompt_ids[renderer_path_len:renderer_end]
            existing = _matching_node(
                trace,
                parent,
                prompt[i],
                node_tokens if tokens is not None else None,
                renderer_node_tokens if tokens is not None else None,
                tools=turn.tools,
                dialect=turn.dialect,
            )
        if existing is None:
            break
        prefix.append(existing)
        path_len = end
        renderer_path_len = renderer_end

    num_reused = len(prefix)
    parent = prefix[-1] if prefix else None
    cursor: int | None = None
    renderer_cursor: int | None = None
    # Track new nodes separately so routed-expert attribution needs only node ids, not this path.
    new_node_ids: list[int] = []
    for i, msg in enumerate(prompt[num_reused:], start=num_reused):
        start = path_len if cursor is None else cursor
        span = spans[i] if spans and i < len(spans) else None
        end = span[1] if span else start
        # Each node copies its slice: a view would retain the turn's full-context array.
        node_tokens = prompt_ids[start:end].copy()
        renderer_start = (
            renderer_path_len if renderer_cursor is None else renderer_cursor
        )
        renderer_span = (
            renderer_spans[i] if renderer_spans and i < len(renderer_spans) else None
        )
        renderer_end = renderer_span[1] if renderer_span else renderer_start
        trace.nodes.append(
            MessageNode.model_construct(
                parent=parent,
                tools=turn.tools if parent is None else [],
                message=msg,
                token_ids=node_tokens,
                renderer_token_ids=renderer_prompt_ids[
                    renderer_start:renderer_end
                ].copy()
                if expanded
                else None,
                mask=np.zeros(len(node_tokens), dtype=np.bool_),
                is_content=is_content[start:end].copy()
                if has_is_content
                else np.zeros(0, dtype=np.bool_),
            )
        )
        parent = len(trace.nodes) - 1
        _index_node(trace, parent)
        new_node_ids.append(parent)
        cursor = end
        renderer_cursor = renderer_end

    comp_ids = np.asarray(tokens.completion_ids if tokens else (), dtype=np.int32)
    gen_start = path_len if cursor is None else cursor
    gen_prompt = prompt_ids[gen_start:]
    renderer_gen_start = (
        renderer_path_len if renderer_cursor is None else renderer_cursor
    )
    renderer_gen_prompt = renderer_prompt_ids[renderer_gen_start:]
    sampled = np.arange(len(gen_prompt) + len(comp_ids)) >= len(gen_prompt)
    trace.nodes.append(
        MessageNode.model_construct(
            parent=parent,
            tools=turn.tools if parent is None else [],
            message=response.message,
            sampled=True,
            token_ids=np.concatenate([gen_prompt, comp_ids]),
            renderer_token_ids=np.concatenate([renderer_gen_prompt, comp_ids])
            if expanded
            else None,
            mask=sampled,
            is_content=sampled.copy() if has_is_content else np.zeros(0, np.bool_),
            # TurnTokens is discarded after commit, so its completion-sized logprobs move
            # onto the node without a copy.
            logprobs=np.asarray(
                tokens.completion_logprobs if tokens else (), dtype=np.float32
            ),
            weight_versions=tokens.weight_versions if tokens else None,
        )
    )
    assistant_id = len(trace.nodes) - 1
    _index_node(trace, assistant_id)
    new_node_ids.append(assistant_id)

    # Attribute this turn's expert-routing array onto the nodes created this turn (new input
    # nodes in creation order, then the assistant node), each getting the routing for its tokens.
    # The prefix goes in too, so the position the previous turn could only pad can be corrected.
    _attribute_routed_experts(
        trace, prefix, new_node_ids, path_len, tokens.routed_experts if tokens else None
    )

    # Sampling masks are completion-aligned, so only the sampled node carries them.
    _attribute_sampling_mask(
        trace, assistant_id, tokens.sampling_mask if tokens else None
    )
    _attribute_top_logprobs(
        trace, assistant_id, tokens.top_logprobs if tokens else None
    )

    return assistant_id


# --- walking the graph (views) ---------------------------------------------------------


def leaves(trace: Trace) -> list[int]:
    """Node ids that are no node's parent — one per branch (the last node of each). The
    `Trace.branches` view walks each leaf's parents back to its root to build the branch."""
    has_child = {n.parent for n in trace.nodes if n.parent is not None}
    return [i for i in range(len(trace.nodes)) if i not in has_child]


def path(trace: Trace, node: int) -> list[Message]:
    """The conversation ending at `node`: its root-to-node messages."""
    messages: list[Message] = []
    current: int | None = node
    while current is not None:
        messages.append(trace.nodes[current].message)
        current = trace.nodes[current].parent
    messages.reverse()
    return messages
