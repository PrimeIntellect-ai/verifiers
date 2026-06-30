"""Consumer-side resume-point selection for the replay buffer (Option B: ``MessageNode`` fields).

Reads the compaction tags the producing harness stamped on ``MessageNode.kind`` and turns them
into resume points the ReplayTaskset can sample. ``seed_messages`` builds the replay prefix (the
root->node path); ``snapshot_ref_of`` returns the durable sandbox handle to restore (None until
per-turn snapshot capture is wired).
"""

from __future__ import annotations

from verifiers.v1 import graph
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages, UserMessage


def get_tag(trace: Trace, node_id: int) -> str | None:
    """Read a node's replay tag (Option B: from the typed ``MessageNode.kind`` field)."""
    return trace.nodes[node_id].kind


def snapshot_ref_of(trace: Trace, node_id: int) -> str | None:
    """Durable sandbox snapshot ref for a node (Option B: from ``MessageNode.snapshot_ref``).
    None when snapshotting was off/unsupported (the skeleton never captures one yet)."""
    return trace.nodes[node_id].snapshot_ref


def seed_messages(trace: Trace, node_id: int) -> Messages:
    """The replay prefix: messages along root->node_id, in order."""
    path: Messages = []
    nid: int | None = node_id
    while nid is not None:
        path.append(trace.nodes[nid].message)
        nid = trace.nodes[nid].parent
    path.reverse()
    return path


def recheck_points(trace: Trace) -> list[dict]:
    """The 'try again' point: the rollout's final-answer leaf (the last branch's leaf). Purely
    structural — no producer tag needed — so it works for any rollout, compacting or linear."""
    leaves = graph.leaves(trace)
    return [{"node": max(leaves), "kind": "recheck"}] if leaves else []


def resume_points(trace: Trace, *, kinds: set[str]) -> list[dict]:
    """Resume points whose ``kind`` is in ``kinds``. ``compaction_before``/``compaction_after``
    come from the harness tags; ``recheck`` is the structural final-answer leaf. Each: ``node``
    id and ``kind``."""
    points: list[dict] = []
    for nid in range(len(trace.nodes)):
        tag = get_tag(trace, nid)
        if tag in kinds:
            points.append({"node": nid, "kind": tag})
    if "recheck" in kinds:
        points.extend(recheck_points(trace))
    return points


def build_seed(trace: Trace, point: dict, followup: str) -> Messages:
    """The seed conversation for a resume point: the ``root->node`` prefix, plus — for a
    ``recheck`` point — an appended user turn asking the model to check and fix its work."""
    msgs = seed_messages(trace, point["node"])
    if point["kind"] == "recheck":
        msgs = [*msgs, UserMessage(content=followup)]
    return msgs
