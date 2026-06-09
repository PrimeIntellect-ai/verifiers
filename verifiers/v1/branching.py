"""Branch detection: split a flat trajectory into the linear histories that made it.

A rollout records every model call in one flat `trajectory`, but the conversation
isn't always linear. An harness may **compact** its context (a turn's prompt drops
earlier history) or run **subagents** (several independent histories run concurrently
in one trajectory). Each maximal linear history is a **branch**.

We recover branches with one rule — *a turn extends a branch when that branch's last
prompt+response is a prefix of the turn's prompt* — applied with the more reliable
signal available:

  1. **tokens** — prefix match on token ids (prompt_ids/completion_ids), when every
     turn carries them (renderer client). Robust to message reformatting.
  2. **messages** — prefix match on the messages themselves. Assumes the harness echoes
     prior turns **byte-identically** (`reasoning_content` excepted — it never
     round-trips through a prompt; `None`/`""` content compare equal). Our built-in
     harnesses do; some external ones may not.

Every branch stays active and a turn extends the **longest** matching branch, so
compaction overlap and concurrent subagent branches resolve correctly. These functions
are pure (no Trace import at runtime) and test in isolation.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from verifiers.v1.types import AssistantMessage, Message, ToolMessage

if TYPE_CHECKING:
    from verifiers.v1.trace import Turn


def same_message(a: Message, b: Message) -> bool:
    """Equality on the fields that round-trip through a prompt: role, content, tool
    calls (assistant) / call id (tool). Ignores `reasoning_content`, and treats `None`
    and `""` content as equal — an external harness may echo a tool-call-only assistant
    message with either, and that shouldn't split a branch."""
    if type(a) is not type(b) or (a.content or "") != (b.content or ""):
        return False
    if isinstance(a, AssistantMessage):
        assert isinstance(b, AssistantMessage)
        calls_a, calls_b = a.tool_calls or [], b.tool_calls or []
        return len(calls_a) == len(calls_b) and all(
            x.id == y.id and x.name == y.name and x.arguments == y.arguments
            for x, y in zip(calls_a, calls_b)
        )
    if isinstance(a, ToolMessage):
        assert isinstance(b, ToolMessage)
        return a.tool_call_id == b.tool_call_id
    return True  # system / user: content already matched


def segment(turns: list[Turn]) -> list[list[int]]:
    """Group turn indices into branches — each a list of indices in order, possibly
    non-contiguous (concurrent subagent branches). Uses token-id prefixes when every turn
    carries them, else message prefixes. It builds, per turn, the `head` (the prompt that
    must extend a branch) and the `seq` (prompt+response, a branch's running prefix), then
    matches."""
    if turns and all(t.tokens and t.tokens.prompt_ids for t in turns):
        heads = [list(t.tokens.prompt_ids) for t in turns]
        seqs = [[*t.tokens.prompt_ids, *t.tokens.completion_ids] for t in turns]
        eq: Callable[[Any, Any], bool] = operator.eq
    else:
        heads = [list(t.prompt) for t in turns]
        seqs = [[*t.prompt, t.response.message] for t in turns]
        eq = same_message
    return _forest(heads, seqs, eq)


def _forest(
    heads: list[list], seqs: list[list], eq: Callable[[Any, Any], bool]
) -> list[list[int]]:
    """Longest-prefix multi-match: a turn joins the active branch whose running `seq` is
    the longest prefix of the turn's `head`, else starts a new branch. Keeping every
    branch active + longest-match is what resolves compaction overlap and concurrent
    subagent branches."""
    branches: list[list[int]] = []
    prefixes: list[list] = []  # parallel to branches: each branch's running seq
    for i, head in enumerate(heads):
        best, best_len = -1, -1
        for b, prefix in enumerate(prefixes):
            if (
                len(prefix) > best_len
                and len(head) >= len(prefix)
                and all(eq(p, h) for p, h in zip(prefix, head))
            ):
                best, best_len = b, len(prefix)
        if best >= 0:
            branches[best].append(i)
            prefixes[best] = seqs[i]
        else:
            branches.append([i])
            prefixes.append(seqs[i])
    return branches
