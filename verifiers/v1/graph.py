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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import ConfigDict, Field, field_serializer, field_validator
from renderers.base import MultiModalData, PlaceholderRange, RenderedTokens

from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Response,
    StrictBaseModel,
    TextContentPart,
    ToolMessage,
    Usage,
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


class MessageNode(StrictBaseModel):
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
    token_ids: list[int] = Field(default_factory=list)
    """This message's delta contribution to the cumulative token sequence: its leading
    template scaffold + its own tokens — for an assistant, the generation-prompt scaffold
    followed by the sampled completion. Concatenated along a path, these reproduce the exact
    `prompt_ids + completion_ids` the model saw."""
    mask: list[bool] = Field(default_factory=list)
    """Per-token, parallel to `token_ids`: True for trainable, model-sampled tokens (only an
    assistant node's completion span); False for template scaffold and every input-message
    token."""
    logprobs: list[float] = Field(default_factory=list)
    """Sampling logprobs for the sampled tokens — length equals the number of True entries in
    `mask`; empty for input messages."""
    finish_reason: FinishReason = None
    """The response's finish reason (assistant nodes only) — kept for truncation detection."""
    multi_modal_data: MultiModalData | None = None
    """The renderer items for the images this message's content introduces (pixel tensors,
    grids, hashes, placeholders) — the only carrier of the pixels from the env server to the
    trainer. `Branch.multi_modal_data` concatenates them along the path into the training
    `mm_kwargs`. Rides the wire as raw bytes (msgpack `bin`) since pydantic can't JSON the numpy;
    kept off disk by the dump-site `exclude` in prime-rl (the tensors bloat the rollout jsonl)."""
    usage: Usage | None = None
    """Provider-reported token usage for this message's response (assistant nodes). Preserved
    on the wire and on disk, including cache-read tokens when the provider reports them."""
    routed_experts: np.ndarray | None = None
    """This node's slice of the MoE expert-routing array — uint8 `[len(token_ids), layers,
    top_k]`, the expert ids inference selected for exactly this node's tokens. Attributed from
    the turn's `generate` payload by `_attribute_routed_experts`; `Branch.routed_experts`
    concatenates these along the path into the trainer's router-replay input. Rides the wire as
    a raw-bytes `__nd__` dict; kept off disk by the dump-site `exclude` in prime-rl."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_serializer("multi_modal_data")
    def serialize_multi_modal_data(self, mmd: MultiModalData | None) -> dict | None:
        """`MultiModalData` -> msgpack-safe dict so the pixel tensors ride the wire; numpy
        `mm_items` values become raw-bytes `__nd__` dicts (every renderer emits `return_tensors="np"`)."""
        if mmd is None:
            return None
        return {
            "mm_hashes": {k: list(v) for k, v in mmd.mm_hashes.items()},
            "mm_placeholders": {
                modality: [{"offset": p.offset, "length": p.length} for p in ranges]
                for modality, ranges in mmd.mm_placeholders.items()
            },
            "mm_items": {
                modality: [
                    {k: _encode_ndarray(v) for k, v in item.items()} for item in items
                ]
                for modality, items in mmd.mm_items.items()
            },
        }

    @field_validator("multi_modal_data", mode="before")
    @classmethod
    def deserialize_multi_modal_data(cls, value: Any) -> MultiModalData | None:
        if value is None or isinstance(value, MultiModalData):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"cannot build MultiModalData from {type(value).__name__}")
        return MultiModalData(
            mm_hashes={k: list(v) for k, v in (value.get("mm_hashes") or {}).items()},
            mm_placeholders={
                modality: [
                    PlaceholderRange(offset=p["offset"], length=p["length"])
                    for p in ranges
                ]
                for modality, ranges in (value.get("mm_placeholders") or {}).items()
            },
            mm_items={
                modality: [
                    {k: _decode_ndarray(v) for k, v in item.items()} for item in items
                ]
                for modality, items in (value.get("mm_items") or {}).items()
            },
        )

    @field_serializer("routed_experts")
    def serialize_routed_experts(self, re: np.ndarray | None) -> dict | None:
        """uint8 routing array -> raw-bytes `__nd__` dict so it rides the wire (numpy can't JSON)."""
        return None if re is None else _encode_ndarray(re)

    @field_validator("routed_experts", mode="before")
    @classmethod
    def deserialize_routed_experts(cls, value: Any) -> np.ndarray | None:
        if value is None or isinstance(value, np.ndarray):
            return value
        if isinstance(value, dict) and value.get("__nd__"):
            return _decode_ndarray(value)
        raise TypeError(f"cannot build routed_experts from {type(value).__name__}")


def _canonical_tool_arguments(arguments: str) -> str:
    try:
        return json.dumps(json.loads(arguments), sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, ValueError):
        return arguments


def message_hash(message: Message) -> str:
    """Stable content hash on the fields that round-trip through a prompt — role, content
    (None and "" equal), assistant reasoning content when present, assistant tool calls,
    tool call id. Two messages hash equal iff they're the same conversational message, so a
    re-stated prefix message dedups to one node. The dedup key for sharing a prefix across
    turns/branches; salt-free so it is identical across processes and after deserialization."""
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
        if message.provider_state:
            # Signed/encrypted continuation state distinguishes otherwise equal turns.
            add("provider_state")
            add(json.dumps(message.provider_state, sort_keys=True))
        for tc in message.tool_calls or []:
            add("tool_call")
            add(tc.id)
            add(tc.name)
            add(_canonical_tool_arguments(tc.arguments))
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

    @property
    def parent(self) -> int | None:
        return self.prefix_node_ids[-1] if self.prefix_node_ids else None

    def previous_token_ids(self) -> tuple[list[int], list[int]] | None:
        """Return `(previous_prompt_ids, previous_completion_ids)` for a bridge anchor.

        The anchor must end at a sampled assistant node. That node stores generation-prompt
        scaffold followed by sampled completion tokens, so split at the first sampled token.
        """
        if not self.prefix_node_ids:
            return None
        last = self.trace.nodes[self.prefix_node_ids[-1]]
        if not last.sampled:
            return None
        first_sampled = next(
            (i for i, sampled in enumerate(last.mask) if sampled), None
        )
        if first_sampled is None:
            return None
        if any(not sampled for sampled in last.mask[first_sampled:]):
            return None

        prompt_ids: list[int] = []
        for nid in self.prefix_node_ids[:-1]:
            prompt_ids.extend(self.trace.nodes[nid].token_ids)
        prompt_ids.extend(last.token_ids[:first_sampled])
        # Slicing already returns an independent list; avoid a second completion-sized copy.
        completion_ids = last.token_ids[first_sampled:]
        if not prompt_ids or not completion_ids:
            return None
        return prompt_ids, completion_ids

    def prompt_message_spans(
        self, tail_attribution: RenderedTokens
    ) -> list[tuple[int, int] | None]:
        """Convert bridge-tail attribution into full-prompt message spans."""
        # Reused bridge tokens are unattributed, so scan only the newly rendered tail.
        tail_spans = RenderedTokens(
            message_indices=tail_attribution.message_indices[self.path_len :],
            message_roles=tail_attribution.message_roles,
        ).message_token_spans()
        # Tail spans are slice-relative; restore their full-prompt token offsets.
        return [None] * self.tail_start + [
            None if span is None else (span[0] + self.path_len, span[1] + self.path_len)
            for span in tail_spans
        ]

    def commit(self, response: Response) -> None:
        _commit_turn(self, response)


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


def _part_modality(part) -> str | None:
    """The multimodal modality a content part introduces (currently only images), or None."""
    return "image" if getattr(part, "type", None) == "image_url" else None


def _attribute_mm(
    trace: Trace,
    path: list[tuple[int, Message]],
    num_reused: int,
    mmd: MultiModalData | None,
) -> None:
    """Attach each new image's renderer item to the node whose message introduced it. The
    renderer emits items per modality in prompt order (message order, then content-part order),
    so we walk the path advancing a per-modality cursor over every message's media but write
    only the nodes created this turn — `path[:num_reused]` is the reused prefix, already
    attributed when first created. Item order is all training needs; placeholder offsets aren't
    carried."""
    if mmd is None or mmd.is_empty():
        return
    cursors: dict[str, int] = {}
    for pos, (node_id, msg) in enumerate(path):
        content = msg.content
        if not isinstance(content, list):
            continue
        node_items: dict[str, list] = {}
        node_hashes: dict[str, list] = {}
        for part in content:
            modality = _part_modality(part)
            if modality is None:
                continue
            k = cursors.get(modality, 0)
            cursors[modality] = k + 1
            # Reused prefix: advance the cursor over its media, don't re-attribute.
            if pos < num_reused:
                continue
            items = mmd.mm_items.get(modality) or []
            hashes = mmd.mm_hashes.get(modality) or []
            if k < len(items):
                node_items.setdefault(modality, []).append(items[k])
            if k < len(hashes):
                node_hashes.setdefault(modality, []).append(hashes[k])
        if node_items:
            trace.nodes[node_id].multi_modal_data = MultiModalData(
                mm_items=node_items, mm_hashes=node_hashes
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
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(payload["shape"])
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


def _commit_turn(turn: PendingTurn, response: Response) -> None:
    """Insert one prepared model turn into the graph.

    Token attribution anchors new tokens to the cumulative *stored* length of the reused
    prefix (`path_len`), not message spans — the previous assistant's closing scaffold lives
    in its later input-form span but not its stored generation form, so anchoring on spans
    would drop it. The new tokens (`prompt_ids[path_len:]`) are split among the new input
    messages by span (leading template scaffold folds into the following message), and the
    trailing generation prompt goes on the assistant node before its sampled completion. By
    construction `concat(node.token_ids along the path) == prompt_ids + completion_ids`."""
    trace = turn.trace
    prompt = turn.prompt
    tokens = response.tokens
    multi_modal_data = tokens.multi_modal_data if tokens else None
    prompt_ids = tokens.prompt_ids if tokens else []
    spans = tokens.message_spans if tokens else None
    idx = _head_index(trace)

    # Token-based prefix reuse. `prepare_turn` matched the prefix by message hash (content); when
    # this turn carries token ids, tighten that to token identity — the stored prefix must be an
    # exact token prefix of what the model saw this turn (`prompt_ids`). Reuse whole nodes within
    # the longest common token prefix and fork at the first divergence, so a retokenized prior
    # (BPE drift, dropped `<think>`, rewritten tool calls) branches off with this turn's real
    # tokens instead of silently inheriting stale ones. Comparing the *concatenated* prefix (not
    # per-message spans) is what makes this correct: a prior assistant's stored generation form
    # and its re-rendered input form place the turn-close scaffold in different nodes but at the
    # same position, so only a genuine content/token change shifts the common prefix. The bridge
    # keeps the prior verbatim so it matches fully (stays linear); the eval relay carries no token
    # ids and keeps the message-hash prefix.
    prefix = turn.prefix_node_ids
    path_len = turn.path_len  # cumulative stored token length of the reused prefix
    if tokens is not None and prefix:
        # Compare node by node against the prompt_ids slice at the running offset (C-level list
        # ==, short-circuits at the first divergent node) — no full concatenation materialized.
        keep = 0
        off = 0
        for nid in prefix:
            node_tokens = trace.nodes[nid].token_ids
            if prompt_ids[off : off + len(node_tokens)] != node_tokens:
                break
            off += len(node_tokens)
            keep += 1
        prefix = prefix[:keep]
        path_len = off
    num_reused = len(prefix)
    parent = prefix[-1] if prefix else None
    # cursor: in prompt_ids, the end of the previous *new* message's tokens
    cursor: int | None = None
    # Track new nodes separately so routed-expert attribution does not need this full path.
    new_node_ids: list[int] = []
    # Materialize the reused message path only for multimodal cursor attribution.
    mm_path: list[tuple[int, Message]] | None = None
    if multi_modal_data is not None:
        mm_path = [(nid, prompt[i]) for i, nid in enumerate(prefix)]
    for i, msg in enumerate(prompt[num_reused:], start=num_reused):
        key = (parent, message_hash(msg))
        start = path_len if cursor is None else cursor
        span = spans[i] if spans and i < len(spans) else None
        end = span[1] if span else start
        node_tokens = prompt_ids[start:end]
        trace.nodes.append(
            # Every value is already typed framework data; avoid revalidating and copying
            # potentially huge token slices a second time.
            MessageNode.model_construct(
                parent=parent,
                message=msg,
                token_ids=node_tokens,
                mask=[False] * len(node_tokens),
            )
        )
        parent = len(trace.nodes) - 1
        idx[key] = parent
        new_node_ids.append(parent)
        if mm_path is not None:
            mm_path.append((parent, msg))
        cursor = end

    # Assistant node: trailing scaffold (the generation prompt) + the sampled completion.
    comp_ids = tokens.completion_ids if tokens else []
    gen_start = path_len if cursor is None else cursor
    gen_prompt = prompt_ids[gen_start:]
    trace.nodes.append(
        MessageNode.model_construct(
            parent=parent,
            message=response.message,
            sampled=True,
            token_ids=[*gen_prompt, *comp_ids],
            mask=[False] * len(gen_prompt) + [True] * len(comp_ids),
            # TurnTokens is discarded after commit, so transfer its logprobs without copying.
            logprobs=tokens.completion_logprobs if tokens else [],
            finish_reason=response.finish_reason,
            usage=response.usage,
        )
    )
    # Register the assistant so the next turn's prompt (which restates it) reuses this node.
    assistant_id = len(trace.nodes) - 1
    idx[(parent, message_hash(response.message))] = assistant_id
    new_node_ids.append(assistant_id)

    # Attribute this turn's images onto the input nodes that introduced them (by content part).
    if mm_path is not None:
        _attribute_mm(trace, mm_path, num_reused, multi_modal_data)

    # Attribute this turn's expert-routing array onto the nodes created this turn (new input
    # nodes in creation order, then the assistant node), each getting the routing for its tokens.
    _attribute_routed_experts(
        trace, new_node_ids, path_len, tokens.routed_experts if tokens else None
    )


# --- walking the graph (views) ---------------------------------------------------------


def leaves(trace: Trace) -> list[int]:
    """Node ids that are no node's parent — one per branch (the last node of each). The
    `Trace.branches` view walks each leaf's parents back to its root to build the branch."""
    has_child = {n.parent for n in trace.nodes if n.parent is not None}
    return [i for i in range(len(trace.nodes)) if i not in has_child]
