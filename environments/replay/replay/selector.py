"""Consumer-side resume-point selection for the replay buffer (Option B: ``MessageNode`` fields).

Reads the compaction tags the producing harness stamped on ``MessageNode.kind`` and turns them
into resume points the ReplayTaskset can sample. ``seed_messages`` builds the replay prefix (the
root->node path); ``snapshot_ref_of`` returns the durable sandbox handle to restore (None until
per-turn snapshot capture is wired).
"""

from __future__ import annotations

from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages


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


def resume_points(trace: Trace, *, kinds: set[str]) -> list[dict]:
    """Compaction resume points whose tag is in ``kinds``. Each: ``node`` id and ``kind``
    ("compaction_before" | "compaction_after")."""
    points: list[dict] = []
    for nid in range(len(trace.nodes)):
        tag = get_tag(trace, nid)
        if tag in kinds:
            points.append({"node": nid, "kind": tag})
    return points
