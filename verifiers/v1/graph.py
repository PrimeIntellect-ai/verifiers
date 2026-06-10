"""Message-graph trajectory: store each message once, recover branches by walking.

Replaces the flat `trajectory: list[Turn]` — where every turn restated the whole prompt,
so storage was quadratic in turns — with a graph of `MessageNode`s, one per distinct
message, each linked to its predecessor. A rollout's conversation is a path from a root to
a leaf; branches (compaction, subagents) are simply multiple leaves. Each node stores only
the tokens it *adds* to the cumulative sequence, so a branch's training sample is a cheap
concat of node `token_ids`/`mask`/`logprobs` along its path.

Token attribution (renderer client): the renderer reports, per prompt, each message's token
span (`RenderedTokens.message_token_spans()`, carried on `TurnTokens.message_spans`). A new
input message's node gets its span plus the leading template scaffold since the previous
message; the trailing scaffold (the generation prompt) goes on the assistant node, prefixed
to its sampled completion. By construction `concat(node.token_ids along a path)` reproduces
the exact `prompt_ids + completion_ids` the model saw (see `tests`).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from pydantic import Field

from verifiers.v1.types import (
    AssistantMessage,
    Message,
    Response,
    StrictBaseModel,
    ToolMessage,
    TurnTokens,
    Usage,
)

if TYPE_CHECKING:
    from verifiers.v1.trace import Branch, Trace, Turn


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


def message_hash(message: Message) -> str:
    """Stable content hash on the fields `branching.same_message` compares — role, content
    (None and "" equal), assistant tool calls, tool call id; `reasoning_content` ignored.
    The dedup key for sharing a prefix message across turns/branches; salt-free so it is
    identical across processes and after deserialization."""
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
    cursor: int | None = None  # in prompt_ids: end of the previous *new* message's tokens
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


def _path_to_turns(trace: "Trace", path: list[int]) -> list["Turn"]:
    """Reconstruct the per-turn `Turn` view along a node path: each assistant node closes a
    turn whose `prompt` is the messages seen so far and whose `tokens` are the running concat
    (prompt = ancestors' token_ids + this turn's generation scaffold; completion = sampled)."""
    from verifiers.v1.trace import Turn

    turns: list[Turn] = []
    prompt_msgs: list[Message] = []
    prompt_ids: list[int] = []
    for nid in path:
        node = trace.nodes[nid]
        if isinstance(node.message, AssistantMessage):
            comp_ids = [t for t, s in zip(node.token_ids, node.mask) if s]
            scaffold = [t for t, s in zip(node.token_ids, node.mask) if not s]
            full_prompt = [*prompt_ids, *scaffold]
            tokens = TurnTokens(
                prompt_ids=full_prompt,
                completion_ids=comp_ids,
                completion_logprobs=list(node.logprobs),
            )
            response = Response(
                id="",
                created=0,
                model="",
                message=node.message,
                finish_reason=None,
                usage=Usage(prompt_tokens=len(full_prompt), completion_tokens=len(comp_ids)),
                tokens=tokens,
            )
            turns.append(Turn(prompt=list(prompt_msgs), response=response, tokens=tokens))
        prompt_msgs.append(node.message)
        prompt_ids = [*prompt_ids, *node.token_ids]
    return turns


def branches_from_nodes(trace: "Trace") -> list["Branch"]:
    """Each leaf's root→leaf path becomes a `Branch` (turns = the assistant nodes on it)."""
    from verifiers.v1.trace import Branch

    return [
        Branch(index=i, turns=_path_to_turns(trace, _path_to(trace, leaf)))
        for i, leaf in enumerate(leaves(trace))
    ]


def branch_token_sequences(
    trace: "Trace",
) -> list[tuple[list[int], list[bool], list[float]]]:
    """Per branch (each leaf→root path), the flat training sequence: `(token_ids,
    mask, per_token_logprobs)` — a cheap concat of the path's nodes. `mask`
    marks the trainable (model-generated) tokens; logprobs are 0.0 on non-sampled tokens.
    The training consumer splits this into prompt (up to the first sampled token) +
    completion."""
    out: list[tuple[list[int], list[bool], list[float]]] = []
    for leaf in leaves(trace):
        ids: list[int] = []
        mask: list[bool] = []
        lps: list[float] = []
        for nid in _path_to(trace, leaf):
            node = trace.nodes[nid]
            li = 0
            for tok, sampled in zip(node.token_ids, node.mask):
                ids.append(tok)
                mask.append(sampled)
                if sampled:
                    lps.append(node.logprobs[li] if li < len(node.logprobs) else 0.0)
                    li += 1
                else:
                    lps.append(0.0)
        out.append((ids, mask, lps))
    return out


def trajectory_from_nodes(trace: "Trace") -> list["Turn"]:
    """Every turn in arrival order (one per assistant node, by node id) — the flat view that
    `Trace.trajectory` exposed."""
    turns_by_assistant: dict[int, "Turn"] = {}
    assistant_ids = [
        nid
        for nid, n in enumerate(trace.nodes)
        if isinstance(n.message, AssistantMessage)
    ]
    for leaf in leaves(trace):
        path = _path_to(trace, leaf)
        path_assistants = [n for n in path if isinstance(trace.nodes[n].message, AssistantMessage)]
        for turn, aid in zip(_path_to_turns(trace, path), path_assistants):
            turns_by_assistant[aid] = turn
    return [turns_by_assistant[aid] for aid in assistant_ids if aid in turns_by_assistant]
