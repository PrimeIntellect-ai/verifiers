"""Producer-side helpers: locate the compaction resume points a harness should tag.

A compacting rollout splits into branches (one per context rewrite). Each compaction exposes
two replay resume points:

- ``compaction_after``  — the post-compaction branch start (the rewritten ``[system, user(notes)]``).
  Resuming here, the model continues solving *from* the compaction message.
- ``compaction_before`` — the leaf of the branch that compaction summarized (the prior turn's
  response). Resuming here, the model is back in the pre-compaction context and its continuation
  *writes* the compaction itself (then keeps solving).

This module only *finds* the nodes; where the tag is stored is the A/B decision (Option A:
``trace.info``; Option B: ``MessageNode.kind``) and lives at the harness write site.
"""

from __future__ import annotations

from verifiers.v1 import graph
from verifiers.v1.trace import Trace


def compaction_after_nodes(trace: Trace) -> list[int]:
    """Post-compaction branch starts: the first node of each forked branch. A node with >1
    child is a fork point; ``children[0]`` is the original line, ``children[1:]`` are the
    rewritten (post-compaction) branches. The compacting harness rewrites every turn, so each
    is a compaction boundary."""
    children: dict[int | None, list[int]] = {}
    for nid, node in enumerate(trace.nodes):
        children.setdefault(node.parent, []).append(nid)
    starts: list[int] = []
    for kids in children.values():
        if len(kids) > 1:
            starts.extend(kids[1:])
    return starts


def compaction_before_nodes(trace: Trace) -> list[int]:
    """Pre-compaction points: every branch leaf except the final-answer branch. Each such leaf
    is the turn whose output was summarized into the next branch's compaction message, so
    resuming there puts the model right before it writes a compaction."""
    leaves = sorted(graph.leaves(trace))
    return leaves[:-1] if len(leaves) > 1 else []
