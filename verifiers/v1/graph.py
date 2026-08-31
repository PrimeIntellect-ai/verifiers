"""Message-graph trajectory: store each message once, recover branches by walking.

A rollout is a graph of `MessageNode`s — one per distinct message, each linked to its
predecessor. The conversation is a path from a root to a leaf; branches (compaction,
subagents) are simply multiple leaves, so branching falls out of the walk. Each node stores
only the tokens it *adds* to the cumulative sequence, keeping size linear in turns and
making a branch's training sample a cheap concat of node `token_ids`/`mask`/`logprobs` along
its path.

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from pydantic.json_schema import SkipJsonSchema
from renderers.base import RenderedTokens

from verifiers.v1.types import (
    AssistantMessage,
    KeptTokens,
    Message,
    Response,
    TextContentPart,
    Tool,
    ToolMessage,
)

if TYPE_CHECKING:
    from verifiers.v1.trace import Trace


def _encode_ndarray(arr: np.ndarray) -> dict:
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


def _decode_ndarray(d: dict) -> np.ndarray:
    """Reverse :func:`_encode_ndarray`."""
    return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"])


class MessageNode(BaseModel):
    """One message in the graph: a message plus the tokens it adds to the cumulative
    sequence. Concatenating a root→leaf path's nodes reconstructs that branch's full token
    sequence; the mask/logprobs make it a training sample."""

    parent: int | None = None
    """Index into `Trace.nodes` of the predecessor message; None for a root."""
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
    token_ids: list[int] = Field(default_factory=list)
    """This message's delta contribution to the cumulative token sequence: its leading
    template scaffold + its own tokens — for an assistant, the generation-prompt scaffold
    followed by the sampled completion. Concatenated along a path, these reproduce the exact
    `prompt_ids + completion_ids` the model saw."""
    renderer_token_ids: list[int] | None = Field(default=None, exclude=True)
    """Logical renderer tokens retained only while extending a live rollout.
    None means they are identical to `token_ids`; an empty list is a real empty slice."""

    mask: list[bool] = Field(default_factory=list)
    """Per-token, parallel to `token_ids`: True for trainable, model-sampled tokens (only an
    assistant node's completion span); False for template scaffold and every input-message
    token."""
    is_content: list[bool] = Field(default_factory=list)
    """Per-token, parallel to `token_ids` (when populated): True for message-body tokens (the
    renderer's content), False for template scaffold (role-tag openers/closers, inter-turn
    separators, tool-response wraps, the generation prompt). Populated from the renderer's
    `RenderedTokens.is_content`; empty when the renderer doesn't attribute content (e.g. the
    default Jinja renderer) or for relay (eval) turns that carry no token ids. Distinct from
    `mask`: `mask` is "did the model sample this?" (assistant completion only); `is_content`
    is "is this caller/model body vs scaffold?" — meaningful on every role, so observation
    weighting (prime-rl `echo`) can train a tool/user message's *body* without its scaffold."""
    logprobs: list[float] = Field(default_factory=list)
    """Sampling logprobs for the sampled tokens — length equals the number of True entries in
    `mask`; empty for input messages."""
    advantages: list[float] | None = None
    """Per-token credit over the sampled tokens, same layout as `logprobs`. `None` until a
    consumer's RL algorithm assigns it, which is not the same as a credit of zero: a group whose
    rewards were all equal is assigned zeros and carries no gradient, while an unassigned node was
    never scored at all."""
    reference_logprobs: list[float] | None = None
    """Reference-model logprobs over the sampled tokens, in the same compact layout as
    `logprobs`. None means no reference model scored this node."""
    loss_weights: dict[str, list[float]] | None = None
    """Named loss-weight streams aligned to `token_ids`, consumer-stamped."""
    routed_experts: SkipJsonSchema[np.ndarray | None] = None
    """This node's slice of the MoE expert-routing array — uint8 `[len(token_ids), layers,
    top_k]`, the expert ids inference selected for exactly this node's tokens. Attributed from
    the turn's `generate` payload by `_attribute_routed_experts`; `Branch.routed_experts`
    concatenates these along the path into the trainer's router-replay input. Rides the wire as
    a raw-bytes `__nd__` dict; kept off disk by the dump-site `exclude` in prime-rl."""
    kept_tokens: SkipJsonSchema[KeptTokens | None] = None
    """Kept-set sampling masks for this node's sampled tokens, decoded: `ids` flat int32
    in position order, `counts` the per-token kept-set sizes (aligned with `logprobs`;
    0 = no mask). Assistant nodes only; consumed via `Branch.kept_tokens` for
    sampling-replay training. Rides the wire as raw-bytes `__nd__` dicts; kept off disk
    by the dump-site `exclude` in prime-rl."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def logical_ids(self) -> list[int]:
        return (
            self.token_ids
            if self.renderer_token_ids is None
            else self.renderer_token_ids
        )

    @field_serializer("routed_experts")
    def serialize_ndarray_field(self, arr: np.ndarray | None) -> dict | None:
        """Integer array -> raw-bytes `__nd__` dict so it rides the wire (numpy can't JSON)."""
        return None if arr is None else _encode_ndarray(arr)

    @field_validator("routed_experts", mode="before")
    @classmethod
    def deserialize_ndarray_field(cls, value: Any) -> np.ndarray | None:
        if value is None or isinstance(value, np.ndarray):
            return value
        if isinstance(value, dict) and value.get("__nd__"):
            return _decode_ndarray(value)
        raise TypeError(f"cannot build ndarray field from {type(value).__name__}")

    @field_serializer("kept_tokens")
    def serialize_kept_tokens(self, kept: KeptTokens | None) -> dict | None:
        """`KeptTokens` -> dict of raw-bytes `__nd__` entries so the arrays ride the wire."""
        if kept is None:
            return None
        return {
            "ids": _encode_ndarray(kept.ids),
            "counts": _encode_ndarray(kept.counts),
        }

    @field_validator("kept_tokens", mode="before")
    @classmethod
    def deserialize_kept_tokens(cls, value: Any) -> KeptTokens | None:
        if value is None or isinstance(value, KeptTokens):
            return value
        if isinstance(value, dict):
            return KeptTokens(
                ids=_decode_ndarray(value["ids"]),
                counts=_decode_ndarray(value["counts"]),
            )
        raise TypeError(f"cannot build KeptTokens from {type(value).__name__}")


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
    conversational message, so a re-stated prefix message dedups to one node. The dedup key
    for sharing a prefix across turns/branches; salt-free so it is identical across processes
    and after deserialization."""
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
            add(
                tc.arguments
                if tc.type == "custom"
                else _canonical_tool_arguments(tc.arguments)
            )
    elif isinstance(message, ToolMessage):
        add("tool_call_id")
        add(message.tool_call_id)
    return digest.hexdigest()


def _head_index(trace: Trace) -> dict[tuple[int | None, str], int]:
    """`(parent, msg_hash) -> node_id`, rebuilt lazily from `nodes` after deserialization."""
    if not trace._head_index and trace.nodes:
        trace._head_index = {
            (node.parent, message_hash(node.message)): nid
            for nid, node in enumerate(trace.nodes)
        }
    return trace._head_index


@dataclass(frozen=True)
class PendingTurn:
    """A resolved prompt waiting on model inference.

    `prepare_turn` does the one canonical graph prefix walk. Training clients use the resolved
    prefix for renderer bridging before inference, and `commit` uses the same prefix after
    inference to add only the prompt tail plus the sampled assistant response.
    """

    trace: Trace
    prompt: list[Message]
    prefix_node_ids: list[int]
    path_len: int

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
        num_sampled = sum(last.mask)
        if not num_sampled:
            return None

        renderer_prompt_ids: list[int] = []
        for nid in self.prefix_node_ids[:-1]:
            node = self.trace.nodes[nid]
            renderer_prompt_ids.extend(node.logical_ids)
        last_ids = last.logical_ids
        renderer_prompt_ids.extend(last_ids[:-num_sampled])
        completion_ids = last_ids[-num_sampled:]
        if not renderer_prompt_ids or not completion_ids:
            return None
        return renderer_prompt_ids, completion_ids

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

    def commit(self, response: Response, tools: list[Tool] | None = None) -> int:
        """Add this turn to the graph; returns the committed assistant node's id."""
        assistant_id = _commit_turn(self, response)
        if tools:
            self.trace.tools = tools
        return assistant_id

    def commit_prompt(self, tools: list[Tool] | None = None) -> None:
        """Record an input that terminated before model inference."""
        parent = self.prefix_node_ids[-1] if self.prefix_node_ids else None
        index = _head_index(self.trace)
        for message in self.tail:
            previous = parent
            self.trace.nodes.append(MessageNode(parent=parent, message=message))
            parent = len(self.trace.nodes) - 1
            index[(previous, message_hash(message))] = parent
        if tools:
            self.trace.tools = tools


def prepare_turn(trace: Trace, prompt: list[Message]) -> PendingTurn:
    """Resolve `prompt` against the trace graph without mutating it."""
    idx = _head_index(trace)
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
                for (node_parent, _), node_id in idx.items()
                if node_parent == parent
            ]
            # Repeated image URLs are cheaper to compare than to encode and hash again.
            # Only scan short, unambiguous parents; all other cases use the stable index.
            if len(children) == 1 and trace.nodes[children[0]].message == msg:
                existing = children[0]
        if existing is None:
            existing = idx.get((parent, message_hash(msg)))
        if existing is None:
            break
        prefix_node_ids.append(existing)
        parent = existing
        path_len += len(trace.nodes[existing].token_ids)
    return PendingTurn(
        trace=trace,
        prompt=prompt,
        prefix_node_ids=prefix_node_ids,
        path_len=path_len,
    )


def _attribute_routed_experts(
    trace: Trace,
    new_node_ids: list[int],
    path_len: int,
    payload: Any,
) -> None:
    """Attach each new node's slice of this turn's MoE expert-routing array. The `generate`
    payload's array covers the turn's prompt+completion from `payload["start"]` (0 = from token
    0); the nodes created this turn tile sequence positions `[path_len:]` in creation order, so
    we hand each node `arr[off : off+len(node.token_ids)]` and advance. Reused-prefix nodes keep
    the routing attributed when they were first created. A node whose slice falls outside the
    array (a `start` past `path_len`, e.g. an unexpected prefix-cache delta) is left unset — the
    branch then reports no routing rather than misaligning."""
    if payload is None:
        return
    raw = binascii.a2b_base64(payload["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(payload.get("dtype", "uint8"))).reshape(
        payload["shape"]
    )
    off = path_len - int(payload.get("start", 0) or 0)
    needed = off + sum(len(trace.nodes[nid].token_ids) for nid in new_node_ids)
    for nid in new_node_ids:
        n = len(trace.nodes[nid].token_ids)
        end = off + n
        if n and 0 <= off and end <= arr.shape[0]:
            # Own only this node's rows; a view would retain the turn's full-context array.
            trace.nodes[nid].routed_experts = arr[off:end].copy()
        elif n and arr.shape[0] and 0 <= off and end == needed == arr.shape[0] + 1:
            # The engine omits the turn's final position because no forward pass follows it.
            # Pad only the final node's suffix instead of copying the full-context array.
            trace.nodes[nid].routed_experts = np.concatenate(
                [arr[off:], arr[-1:]], axis=0
            )
        off = end


def _attribute_kept_tokens(
    trace: Trace, assistant_id: int, payload: KeptTokens | None
) -> None:
    """Attach this turn's kept-set sampling masks to the assistant node (the payload
    covers exactly the turn's completion tokens, so no path arithmetic). A payload
    that doesn't line up with the node's sampled tokens is dropped, not misaligned."""
    if payload is None:
        return
    counts = np.frombuffer(binascii.a2b_base64(payload.counts), dtype=np.int32)
    ids = np.frombuffer(binascii.a2b_base64(payload.ids), dtype=np.int32)
    node = trace.nodes[assistant_id]
    if len(counts) != sum(node.mask) or int(counts.sum()) != len(ids):
        return
    # Own the buffers — the payload views reference the turn's response bytes.
    node.kept_tokens = KeptTokens(ids=ids.copy(), counts=counts.copy())


def _project_prompt_attribution(
    renderer_prompt_ids: list[int],
    prompt_ids: list[int],
    mm_token_type_id_map: dict[int, int],
    mm_placeholders: list[tuple[int, int]] | None,
    message_spans: list[tuple[int, int] | None] | None,
    is_content: list[bool] | None,
) -> tuple[list[tuple[int, int] | None] | None, list[bool] | None]:
    """Project logical attribution using vLLM's multimodal placeholder ranges."""
    if renderer_prompt_ids == prompt_ids:
        return message_spans, is_content
    if not mm_token_type_id_map or mm_placeholders is None:
        raise ValueError(
            "cannot align renderer and vLLM prompt tokens without multimodal placeholders"
        )

    offsets = [0]
    prompt_offset = 0
    placeholder_index = 0
    for token_id in renderer_prompt_ids:
        if token_id in mm_token_type_id_map:
            if placeholder_index >= len(mm_placeholders):
                raise ValueError(
                    "vLLM multimodal placeholders do not align with renderer prompt"
                )
            offset, length = mm_placeholders[placeholder_index]
            if offset != prompt_offset or length < 1:
                raise ValueError(
                    "vLLM multimodal placeholders do not align with renderer prompt"
                )
            prompt_offset += length
            placeholder_index += 1
        else:
            if (
                prompt_offset >= len(prompt_ids)
                or prompt_ids[prompt_offset] != token_id
            ):
                raise ValueError(
                    "renderer prompt does not align with vLLM prompt tokens"
                )
            prompt_offset += 1
        offsets.append(prompt_offset)
    if prompt_offset != len(prompt_ids) or placeholder_index != len(mm_placeholders):
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
            projected_spans.append((offsets[start], offsets[end]))

    projected_is_content = is_content
    if is_content:
        if len(is_content) != len(renderer_prompt_ids):
            raise ValueError(
                "content attribution does not match renderer prompt tokens"
            )
        projected_is_content = []
        for index, value in enumerate(is_content):
            projected_is_content.extend([value] * (offsets[index + 1] - offsets[index]))
    return projected_spans, projected_is_content


def _commit_turn(turn: PendingTurn, response: Response) -> int:
    trace = turn.trace
    prompt = turn.prompt
    tokens = response.tokens
    # Constant per renderer, so re-stamping every turn is idempotent.
    if tokens is not None and tokens.mm_token_type_id_map:
        trace.mm_token_type_id_map = tokens.mm_token_type_id_map
    prompt_ids = tokens.prompt_ids if tokens else []
    renderer_prompt_ids = (
        tokens.renderer_prompt_ids
        if tokens and tokens.renderer_prompt_ids is not None
        else prompt_ids
    )
    renderer_spans = tokens.message_spans if tokens else None
    renderer_is_content = tokens.is_content if tokens else None
    idx = _head_index(trace)

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
            if (
                prompt_ids[off : off + len(node.token_ids)] != node.token_ids
                or renderer_prompt_ids[
                    renderer_off : renderer_off + len(node_renderer_ids)
                ]
                != node_renderer_ids
            ):
                break
            off += len(node.token_ids)
            renderer_off += len(node_renderer_ids)
            keep += 1
        if tokens.renderer_prompt_ids is not None and keep != len(prefix):
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
    num_reused = len(prefix)
    parent = prefix[-1] if prefix else None
    cursor: int | None = None
    renderer_cursor: int | None = None
    new_node_ids: list[int] = []
    for i, msg in enumerate(prompt[num_reused:], start=num_reused):
        key = (parent, message_hash(msg))
        start = path_len if cursor is None else cursor
        span = spans[i] if spans and i < len(spans) else None
        end = span[1] if span else start
        node_tokens = prompt_ids[start:end]
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
                message=msg,
                token_ids=node_tokens,
                renderer_token_ids=renderer_prompt_ids[renderer_start:renderer_end],
                mask=[False] * len(node_tokens),
                is_content=is_content[start:end] if has_is_content else [],
            )
        )
        parent = len(trace.nodes) - 1
        idx[key] = parent
        new_node_ids.append(parent)
        cursor = end
        renderer_cursor = renderer_end

    comp_ids = tokens.completion_ids if tokens else []
    gen_start = path_len if cursor is None else cursor
    gen_prompt = prompt_ids[gen_start:]
    renderer_gen_start = (
        renderer_path_len if renderer_cursor is None else renderer_cursor
    )
    renderer_gen_prompt = renderer_prompt_ids[renderer_gen_start:]
    trace.nodes.append(
        MessageNode.model_construct(
            parent=parent,
            message=response.message,
            sampled=True,
            token_ids=[*gen_prompt, *comp_ids],
            renderer_token_ids=[*renderer_gen_prompt, *comp_ids],
            mask=[False] * len(gen_prompt) + [True] * len(comp_ids),
            is_content=([False] * len(gen_prompt) + [True] * len(comp_ids))
            if has_is_content
            else [],
            # TurnTokens is discarded after commit, so transfer its logprobs without copying.
            logprobs=tokens.completion_logprobs if tokens else [],
        )
    )
    assistant_id = len(trace.nodes) - 1
    idx[(parent, message_hash(response.message))] = assistant_id
    new_node_ids.append(assistant_id)

    # Attribute this turn's expert-routing array onto the nodes created this turn (new input
    # nodes in creation order, then the assistant node), each getting the routing for its tokens.
    _attribute_routed_experts(
        trace, new_node_ids, path_len, tokens.routed_experts if tokens else None
    )

    # Attribute this turn's kept-set sampling masks onto the assistant node (they are
    # completion-aligned, so only the sampled node carries them).
    _attribute_kept_tokens(trace, assistant_id, tokens.kept_tokens if tokens else None)

    return assistant_id


# --- walking the graph (views) ---------------------------------------------------------


def leaves(trace: Trace) -> list[int]:
    """Node ids that are no node's parent — one per branch (the last node of each). The
    `Trace.branches` view walks each leaf's parents back to its root to build the branch."""
    has_child = {n.parent for n in trace.nodes if n.parent is not None}
    return [i for i in range(len(trace.nodes)) if i not in has_child]
