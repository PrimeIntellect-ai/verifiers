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

import base64
import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import ConfigDict, Field, field_serializer, field_validator
from renderers.base import MultiModalData, PlaceholderRange

from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Response,
    StrictBaseModel,
    ToolMessage,
    Usage,
)

if TYPE_CHECKING:
    from verifiers.v1.trace import Trace


def _encode_ndarray(arr: np.ndarray) -> dict:
    """A numpy array as a JSON/msgpack-safe dict (dtype + shape + base64 bytes)."""
    arr = np.ascontiguousarray(arr)
    return {
        "__nd__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _decode_ndarray(d: dict) -> np.ndarray:
    """Reverse :func:`_encode_ndarray`."""
    raw = base64.b64decode(d["data"])
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"])


def _encode_mm_value(val: Any) -> Any:
    """Encode one mm-item value: numpy arrays (or torch tensors) → the `__nd__` dict, anything
    else (already-JSON-safe scalars/lists) passed through."""
    if hasattr(val, "detach"):  # torch tensor
        val = val.detach().cpu().numpy()
    if isinstance(val, np.ndarray):
        return _encode_ndarray(val)
    return val


def _decode_mm_value(val: Any) -> Any:
    if isinstance(val, dict) and val.get("__nd__"):
        return _decode_ndarray(val)
    return val


def _encode_multi_modal_data(mmd: MultiModalData) -> dict:
    """`MultiModalData` → a JSON/msgpack-safe dict. `mm_hashes` rides as-is; `mm_placeholders`
    become `{offset,length}` dicts; each `mm_items` value is `_encode_mm_value`d (numpy → base64)."""
    return {
        "mm_hashes": {k: list(v) for k, v in mmd.mm_hashes.items()},
        "mm_placeholders": {
            modality: [{"offset": p.offset, "length": p.length} for p in ranges]
            for modality, ranges in mmd.mm_placeholders.items()
        },
        "mm_items": {
            modality: [
                {key: _encode_mm_value(val) for key, val in item.items()}
                for item in items
            ]
            for modality, items in mmd.mm_items.items()
        },
    }


def _decode_multi_modal_data(d: dict) -> MultiModalData:
    """Reverse :func:`_encode_multi_modal_data`."""
    return MultiModalData(
        mm_hashes={k: list(v) for k, v in (d.get("mm_hashes") or {}).items()},
        mm_placeholders={
            modality: [
                PlaceholderRange(offset=p["offset"], length=p["length"]) for p in ranges
            ]
            for modality, ranges in (d.get("mm_placeholders") or {}).items()
        },
        mm_items={
            modality: [
                {key: _decode_mm_value(val) for key, val in item.items()}
                for item in items
            ]
            for modality, items in (d.get("mm_items") or {}).items()
        },
    )


class MessageNode(StrictBaseModel):
    """One message in the graph: a message plus the tokens it adds to the cumulative
    sequence. Concatenating a root→leaf path's nodes reconstructs that branch's full token
    sequence; the mask/logprobs make it a training sample."""

    parent: int | None = None
    """Index into `Trace.nodes` of the predecessor message; None for a root."""
    message: Message
    """The message this node carries (system / user / assistant / tool)."""
    sampled: bool = False
    """True iff a model call produced this message (the response in `add_turn`); False for
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
    `mm_kwargs`. Rides the wire via a base64 (de)serializer since pydantic can't JSON the numpy;
    kept off disk by the dump-site `exclude` in prime-rl (the tensors bloat the rollout jsonl)."""
    usage: Usage | None = Field(default=None, exclude=True)
    """Provider-reported token usage for this message's response (assistant nodes). Transient
    (excluded from wire/disk); lets the live dashboard show token counts even when the endpoint
    returns no token ids (so `token_ids` is empty)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_serializer("multi_modal_data")
    def _ser_multi_modal_data(self, mmd: MultiModalData | None) -> dict | None:
        return _encode_multi_modal_data(mmd) if mmd is not None else None

    @field_validator("multi_modal_data", mode="before")
    @classmethod
    def _val_multi_modal_data(cls, value: Any) -> MultiModalData | None:
        if value is None or isinstance(value, MultiModalData):
            return value
        if isinstance(value, dict):
            return _decode_multi_modal_data(value)
        raise TypeError(f"cannot build MultiModalData from {type(value).__name__}")


def _content_str(content) -> str:
    """A message body as a stable string for hashing — plain text as-is, a content-part list as
    canonical JSON (so two messages carrying the same text/images hash equal), None as ""."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps([p.model_dump() for p in content], sort_keys=True)


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
    parts: list[str] = [type(message).__name__, _content_str(message.content)]
    if isinstance(message, AssistantMessage):
        if message.reasoning_content is not None:
            parts += ["reasoning_content", message.reasoning_content]
        for tc in message.tool_calls or []:
            parts += [tc.id, tc.name, _canonical_tool_arguments(tc.arguments)]
    elif isinstance(message, ToolMessage):
        parts.append(message.tool_call_id)
    return hashlib.blake2b("\x00".join(parts).encode(), digest_size=16).hexdigest()


def _head_index(trace: "Trace") -> dict[tuple[int | None, str], int]:
    """`(parent, msg_hash) -> node_id`, rebuilt lazily from `nodes` after deserialization."""
    if not trace._head_index and trace.nodes:
        trace._head_index = {
            (node.parent, message_hash(node.message)): nid
            for nid, node in enumerate(trace.nodes)
        }
    return trace._head_index


def _part_modality(part) -> str | None:
    """The multimodal modality a content part introduces (currently only images), or None."""
    return "image" if getattr(part, "type", None) == "image_url" else None


def _attribute_mm(
    trace: "Trace",
    path: "list[tuple[int, Message]]",
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


def add_turn(trace: "Trace", prompt: "list[Message]", response: Response) -> None:
    """Insert one model turn (its prompt messages + its response) into the graph. Reuses any
    existing prefix nodes (by `(parent, hash)`), creates a node per new message attributing
    its tokens, and appends a fresh assistant node holding the generation-prompt scaffold +
    the sampled completion.

    Token attribution anchors new tokens to the cumulative *stored* length of the reused
    prefix (`path_len`), not message spans — the previous assistant's closing scaffold lives
    in its later input-form span but not its stored generation form, so anchoring on spans
    would drop it. The new tokens (`prompt_ids[path_len:]`) are split among the new input
    messages by span (leading template scaffold folds into the following message), and the
    trailing generation prompt goes on the assistant node before its sampled completion. By
    construction `concat(node.token_ids along the path) == prompt_ids + completion_ids`."""
    tokens = response.tokens
    prompt_ids = list(tokens.prompt_ids) if tokens else []
    spans = tokens.message_spans if tokens else None
    idx = _head_index(trace)

    parent: int | None = None
    path_len = 0  # cumulative stored token length of the reused prefix
    # cursor: in prompt_ids, the end of the previous *new* message's tokens
    cursor: int | None = None
    # (node_id, message) per prompt message (reused prefix first, then new), for multimodal
    # attribution; `num_reused` marks where the newly-created nodes begin.
    path: list[tuple[int, Message]] = []
    num_reused = 0
    for i, msg in enumerate(prompt):
        key = (parent, message_hash(msg))
        existing = idx.get(key)
        if cursor is None and existing is not None:  # still extending the shared prefix
            parent = existing
            path_len += len(trace.nodes[existing].token_ids)
            path.append((existing, msg))
            num_reused += 1
            continue
        start = path_len if cursor is None else cursor
        span = spans[i] if spans and i < len(spans) else None
        end = span[1] if span else start
        node_tokens = prompt_ids[start:end]
        trace.nodes.append(
            MessageNode(
                parent=parent,
                message=msg,
                token_ids=node_tokens,
                mask=[False] * len(node_tokens),
            )
        )
        parent = len(trace.nodes) - 1
        idx[key] = parent
        path.append((parent, msg))
        cursor = end

    # Assistant node: trailing scaffold (the generation prompt) + the sampled completion.
    comp_ids = list(tokens.completion_ids) if tokens else []
    gen_start = path_len if cursor is None else cursor
    gen_prompt = prompt_ids[gen_start:]
    trace.nodes.append(
        MessageNode(
            parent=parent,
            message=response.message,
            sampled=True,
            token_ids=[*gen_prompt, *comp_ids],
            mask=[False] * len(gen_prompt) + [True] * len(comp_ids),
            logprobs=list(tokens.completion_logprobs) if tokens else [],
            finish_reason=response.finish_reason,
            usage=response.usage,
        )
    )
    # Register the assistant so the next turn's prompt (which restates it) reuses this node.
    idx[(parent, message_hash(response.message))] = len(trace.nodes) - 1

    # Attribute this turn's images onto the input nodes that introduced them (by content part).
    _attribute_mm(trace, path, num_reused, tokens.multi_modal_data if tokens else None)


# --- walking the graph (views) ---------------------------------------------------------


def leaves(trace: "Trace") -> list[int]:
    """Node ids that are no node's parent — one per branch (the last node of each). The
    `Trace.branches` view walks each leaf's parents back to its root to build the branch."""
    has_child = {n.parent for n in trace.nodes if n.parent is not None}
    return [i for i in range(len(trace.nodes)) if i not in has_child]
