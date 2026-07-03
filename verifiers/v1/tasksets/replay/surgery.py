"""Pure functions over saved rollout records (WireTrace jsonl lines, as raw dicts).

A saved trace's ``nodes`` list is a message *forest*: parent pointers form trees, the main
conversation is rooted at node 0, and subagent conversations (or bare probe calls) get their
own roots appended to the same list, interleaved by index. Compaction leaves no marker — it
is a fork in the main tree whose new child is a non-sampled user message (the summary the
harness restarted from) with sampled turns beneath it. Everything here works on the raw
dicts so the buffer scan never pays pydantic validation for megabyte-sized lines.
"""

from collections import defaultdict
from typing import Any

Node = dict[str, Any]


def usable(record: dict) -> bool:
    """Whether a saved rollout is replayable at all. Saved step files contain the full
    arrival window: errored rollouts and synthesized cancel markers (empty ``nodes``,
    ``errors`` set) ride along and must be screened out."""
    return bool(record.get("nodes")) and not record.get("errors") and any(n["sampled"] for n in record["nodes"])


def is_replay_derived(task: dict) -> bool:
    """Whether a saved task dict came from a replay taskset — it carries the replay
    lineage keys (`ReplayTask.source_task`/`source_id`), whatever the derivation."""
    return isinstance(task.get("source_task"), dict) and bool(task.get("source_id"))


def unwrap_source_task(task: dict) -> dict:
    """The innermost original task under any chain of replay derivations — what inner-taskset
    scoring, tools, and container provisioning must be keyed on, however deep the chain."""
    while is_replay_derived(task) and task["source_task"]:
        task = task["source_task"]
    return task


def build_children(nodes: list[Node]) -> tuple[dict[int, list[int]], list[int]]:
    """Child lists (in node-index order, i.e. creation order) and roots of the forest."""
    children: dict[int, list[int]] = defaultdict(list)
    roots: list[int] = []
    for i, node in enumerate(nodes):
        parent = node["parent"]
        if parent is None:
            roots.append(i)
        else:
            children[parent].append(i)
    return children, roots


def main_tree(children: dict[int, list[int]]) -> set[int]:
    """Node indices of the main conversation: the component rooted at node 0. Subagent
    trees have their own roots and must not leak into seeds or transcripts."""
    seen: set[int] = set()
    stack = [0]
    while stack:
        i = stack.pop()
        seen.add(i)
        stack.extend(children.get(i, []))
    return seen


def _subtree_has_sampled(nodes: list[Node], children: dict[int, list[int]], start: int) -> bool:
    stack = [start]
    while stack:
        i = stack.pop()
        if nodes[i]["sampled"]:
            return True
        stack.extend(children.get(i, []))
    return False


def compaction_forks(nodes: list[Node], children: dict[int, list[int]], tree: set[int]) -> list[int]:
    """Compaction points in the main tree, as the post-compaction child node indices.

    A fork child is a compaction iff it is a non-sampled user message whose subtree
    contains sampled turns — the harness rewrote its prompt to a summary and kept going.
    The detector is structural, not marker-based; other fork kinds (duplicated tool
    results re-appended after retries, retried assistant twins) don't match it."""
    forks: list[int] = []
    for parent, siblings in children.items():
        if len(siblings) < 2 or parent not in tree:
            continue
        for child in siblings[1:]:
            message = nodes[child]["message"]
            if (
                message["role"] == "user"
                and not nodes[child]["sampled"]
                and _subtree_has_sampled(nodes, children, child)
            ):
                forks.append(child)
    return sorted(forks)


def tool_call_anchors(nodes: list[Node], children: dict[int, list[int]], tree: set[int]) -> list[int]:
    """Resumable tool-result nodes in the main tree: points where the conversation up to
    and including the node is a well-formed prefix (every tool call issued so far has its
    result), so the model can continue right after the tool output. A result whose
    issuing assistant still has sibling calls pending is not resumable — the seed would
    carry a dangling call."""
    anchors: list[int] = []
    for i in sorted(tree):
        if nodes[i]["message"]["role"] != "tool":
            continue
        pending: set[str] = set()
        for j in path_to_root(nodes, i):
            message = nodes[j]["message"]
            if message["role"] == "assistant":
                pending.update(call["id"] for call in message.get("tool_calls") or [])
            elif message["role"] == "tool":
                pending.discard(message.get("tool_call_id"))
        if not pending:
            anchors.append(i)
    return anchors


def path_to_root(nodes: list[Node], leaf: int) -> list[int]:
    path = []
    i: int | None = leaf
    while i is not None:
        path.append(i)
        i = nodes[i]["parent"]
    return path[::-1]


def final_leaf(children: dict[int, list[int]], tree: set[int]) -> int:
    """The last-written leaf of the main tree — the conversation's final state. The
    global max-index leaf may sit in a subagent tree, so restrict to the main tree."""
    return max(i for i in tree if i not in children)


def path_messages(nodes: list[Node], path: list[int]) -> list[dict]:
    return [nodes[i]["message"] for i in path]


def continue_seed(nodes: list[Node], anchor: int) -> list[dict]:
    """CONTINUE seed: messages from the root down to (and including) the anchor node —
    the post-compaction user message (the exact prompt the original harness restarted
    from), or a tool result (the model resumes right after seeing it)."""
    return path_messages(nodes, path_to_root(nodes, anchor))


def recheck_seed(nodes: list[Node], children: dict[int, list[int]], tree: set[int], instruction: str) -> list[dict]:
    """RECHECK seed: the final branch's messages plus an appended check-your-work user turn.

    Rollouts routinely end truncated mid-tool-call (timeouts, context length); a trailing
    assistant's pending ``tool_calls`` must be stripped or the seed is API-malformed, and
    if that leaves it with no content it is dropped entirely."""
    path = path_to_root(nodes, final_leaf(children, tree))
    messages = [dict(m) for m in path_messages(nodes, path)]
    if messages and messages[-1]["role"] == "assistant" and messages[-1].get("tool_calls"):
        messages[-1]["tool_calls"] = None
        if not messages[-1].get("content"):
            messages.pop()
    messages.append({"role": "user", "content": instruction})
    return messages
