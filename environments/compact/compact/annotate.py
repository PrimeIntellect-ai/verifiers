"""Replay resume-point annotation + selection (Option A: ``trace.info`` side-channel).

During generation the harness tags interesting nodes; the replay-buffer selector reads the
tags back. Tool *identity* (which tool) and "is this a tool call" come from the typed graph,
so only two things need a tag: branch provenance (compaction/subagent) and tool failure
status (``failed_tool_call``), which the trace otherwise drops.

Option A stores tags in ``trace.info["node_tags"]`` (a ``{str(node_id): tag}`` map). No
change to the core ``MessageNode``/``Trace`` schema — ``info`` already round-trips to disk and
the env-server wire. The trade-off vs. a typed node field (Option B) is an untyped dict joined
by ``node_id`` (stable across the dump→reload we control; fragile if the graph is re-derived).
"""

from __future__ import annotations

from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, ToolMessage


def branch_start_nodes(trace: Trace) -> list[int]:
    """First node of each forked branch. A node with >1 child is a fork point; ``children[0]``
    is the original line and ``children[1:]`` are the forks. The compacting harness rewrites
    context every turn, so each fork start is a compaction boundary."""
    children: dict[int | None, list[int]] = {}
    for nid, node in enumerate(trace.nodes):
        children.setdefault(node.parent, []).append(nid)
    starts: list[int] = []
    for kids in children.values():
        if len(kids) > 1:
            starts.extend(kids[1:])
    return starts


def tool_nodes(trace: Trace) -> dict[str, int]:
    """``tool_call_id -> node_id`` for every tool result — the robust key for joining the
    program's per-call failure log back onto the graph."""
    return {
        node.message.tool_call_id: nid
        for nid, node in enumerate(trace.nodes)
        if isinstance(node.message, ToolMessage)
    }


def tool_name(trace: Trace, node: MessageNode) -> str | None:
    """The tool/function name for a tool-result node: ``ToolMessage.name`` when the dialect
    recovered it, else the issuing assistant's matching ``tool_calls[].name``."""
    m = node.message
    if not isinstance(m, ToolMessage):
        return None
    if m.name:
        return m.name
    nid = node.parent
    while nid is not None:
        parent = trace.nodes[nid].message
        if isinstance(parent, AssistantMessage):
            for tc in parent.tool_calls or []:
                if tc.id == m.tool_call_id:
                    return tc.name
            break
        nid = trace.nodes[nid].parent
    return None


def get_tag(trace: Trace, node_id: int) -> str | None:
    """Read a node's replay tag (Option A: from ``trace.info``)."""
    return trace.info.get("node_tags", {}).get(str(node_id))


def resume_points(trace: Trace, *, kinds: set[str]) -> list[dict]:
    """Enumerate replay resume points, filtered to ``kinds``. Tool calls come from the typed
    graph; compaction comes from the harness tag. Each point carries its ``node`` id, ``kind``,
    and for tool calls the ``tool`` name and whether it ``failed``."""
    points: list[dict] = []
    for nid, node in enumerate(trace.nodes):
        if isinstance(node.message, ToolMessage):
            points.append(
                {
                    "node": nid,
                    "kind": "tool_call",
                    "tool": tool_name(trace, node),
                    "failed": get_tag(trace, nid) == "failed_tool_call",
                }
            )
        elif get_tag(trace, nid) == "compaction":
            points.append({"node": nid, "kind": "compaction"})
    return [p for p in points if p["kind"] in kinds]
