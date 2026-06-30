"""Consumer-side buffer reading + resume-point selection (Option B: ``MessageNode`` fields).

Turns the compaction tags the producing harness stamped on ``MessageNode.kind`` into resume
points the tasksets/harnesses sample. ``build_seed`` produces the seed conversation per mode;
``snapshot_ref_of`` returns the durable sandbox handle to restore (None until per-turn snapshot
capture is wired). Only ``get_tag``/``snapshot_ref_of`` differ from Option A — everything else
(buffer reading, resume-point logic, seed building) is shared.
"""

from __future__ import annotations

import glob
import json

from verifiers.v1 import graph
from verifiers.v1.trace import Trace, WireTrace
from verifiers.v1.types import Messages, UserMessage

DEFAULT_FOLLOWUP = "Check your work. If anything is wrong, fix it and give the corrected final answer."


def get_tag(trace: Trace, node_id: int) -> str | None:
    """Read a node's replay tag (Option B: from the typed ``MessageNode.kind`` field)."""
    return trace.nodes[node_id].kind


def snapshot_ref_of(trace: Trace, node_id: int) -> str | None:
    """Durable sandbox snapshot ref for a node (Option B: from ``MessageNode.snapshot_ref``).
    None when snapshotting was off/unsupported (the skeleton never captures one yet)."""
    return trace.nodes[node_id].snapshot_ref


def iter_traces(buffer_glob: str):
    """Yield each stored rollout (``WireTrace``) from the buffer glob, in file then line order."""
    for path in sorted(glob.glob(buffer_glob)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield WireTrace.model_validate(json.loads(line))


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
    """Resume points whose ``kind`` is in ``kinds``. ``compaction_before``/``compaction_after``
    come from the harness tags; ``recheck`` and ``judge`` are the structural final-answer leaf
    (re-roll vs. judge-the-attempt). Each: ``node`` id and ``kind``."""
    points: list[dict] = []
    for nid in range(len(trace.nodes)):
        tag = get_tag(trace, nid)
        if tag in kinds:
            points.append({"node": nid, "kind": tag})
    leaves = graph.leaves(trace)
    if leaves:
        final = max(leaves)  # the rollout's final-answer leaf
        points += [{"node": final, "kind": k} for k in ("recheck", "judge") if k in kinds]
    return points


def render_transcript(trace: Trace, node_id: int) -> str:
    """The conversation along root->node_id as plain text, for a judge prompt."""
    lines = []
    for m in seed_messages(trace, node_id):
        content = m.content if isinstance(m.content, str) else (m.content or "")
        lines.append(f"{m.role}: {content}")
    return "\n".join(lines)


def judge_prompt(trace: Trace, node_id: int) -> str:
    """A 'was this attempt correct?' prompt presenting the rollout's transcript."""
    return (
        "You are judging whether a previous attempt solved its task correctly.\n\n"
        f"--- transcript ---\n{render_transcript(trace, node_id)}\n--- end transcript ---\n\n"
        "Was the final answer correct? Reply with exactly 'yes' or 'no'."
    )


def build_seed(trace: Trace, point: dict, followup: str) -> Messages:
    """The seed conversation for a resume point:
    - ``judge``   -> a single user turn presenting the rollout to be graded;
    - ``recheck`` -> the full rollout prefix + an appended check-your-work user turn;
    - ``compaction_*`` -> the plain ``root->node`` prefix.
    """
    if point["kind"] == "judge":
        return [UserMessage(content=judge_prompt(trace, point["node"]))]
    msgs = seed_messages(trace, point["node"])
    if point["kind"] == "recheck":
        msgs = [*msgs, UserMessage(content=followup)]
    return msgs
