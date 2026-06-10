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

import hashlib
from typing import TYPE_CHECKING

from pydantic import Field

from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Response,
    StrictBaseModel,
    ToolMessage,
)

if TYPE_CHECKING:
    from verifiers.v1.trace import Branch, Trace


class MessageNode(StrictBaseModel):
    """One message in the graph: a message plus the tokens it adds to the cumulative
    sequence. Concatenating a root→leaf path's nodes reconstructs that branch's full token
    sequence; the mask/logprobs make it a training sample."""

    parent: int | None = None
    """Index into `Trace.nodes` of the predecessor message; None for a root."""
    message: Message
    """The message this node carries (system / user / assistant / tool)."""
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


def message_hash(message: Message) -> str:
    """Stable content hash on the fields that round-trip through a prompt — role, content
    (None and "" equal), assistant tool calls, tool call id; `reasoning_content` ignored.
    Two messages hash equal iff they're the same conversational message, so a re-stated
    prefix message dedups to one node. The dedup key for sharing a prefix across
    turns/branches; salt-free so it is identical across processes and after deserialization."""
    parts: list[str] = [type(message).__name__, message.content or ""]
    if isinstance(message, AssistantMessage):
        for tc in message.tool_calls or []:
            parts += [tc.id, tc.name, tc.arguments]
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
    for i, msg in enumerate(prompt):
        key = (parent, message_hash(msg))
        existing = idx.get(key)
        if cursor is None and existing is not None:  # still extending the shared prefix
            parent = existing
            path_len += len(trace.nodes[existing].token_ids)
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
        cursor = end

    # Assistant node: trailing scaffold (the generation prompt) + the sampled completion.
    comp_ids = list(tokens.completion_ids) if tokens else []
    gen_start = path_len if cursor is None else cursor
    gen_prompt = prompt_ids[gen_start:]
    trace.nodes.append(
        MessageNode(
            parent=parent,
            message=response.message,
            token_ids=[*gen_prompt, *comp_ids],
            mask=[False] * len(gen_prompt) + [True] * len(comp_ids),
            logprobs=list(tokens.completion_logprobs) if tokens else [],
            finish_reason=response.finish_reason,
        )
    )
    # Register the assistant so the next turn's prompt (which restates it) reuses this node.
    idx[(parent, message_hash(response.message))] = len(trace.nodes) - 1


# --- walking the graph (views) ---------------------------------------------------------


def _path_to(trace: "Trace", leaf: int) -> list[int]:
    """Node ids from the root down to `leaf` (inclusive), in order."""
    path: list[int] = []
    nid: int | None = leaf
    while nid is not None:
        path.append(nid)
        nid = trace.nodes[nid].parent
    path.reverse()
    return path


def leaves(trace: "Trace") -> list[int]:
    """Node ids that are no node's parent — one per branch (the last node of each)."""
    has_child = {n.parent for n in trace.nodes if n.parent is not None}
    return [i for i in range(len(trace.nodes)) if i not in has_child]


def branches_from_nodes(trace: "Trace") -> list["Branch"]:
    """Each leaf's root→leaf node path becomes a `Branch` — one per leaf (one branch when
    linear, several under compaction or subagents)."""
    from verifiers.v1.trace import Branch

    return [
        Branch(index=i, nodes=[trace.nodes[nid] for nid in _path_to(trace, leaf)])
        for i, leaf in enumerate(leaves(trace))
    ]
