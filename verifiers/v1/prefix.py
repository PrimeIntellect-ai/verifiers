"""Prefix replay: start a rollout from the first model calls of a finished rollout.

A `Prefix` holds a source trace's first committed calls. The rollout runs normally from
the task start, but the train client answers those calls with the recorded completions
without sampling. The harness executes the recorded actions itself, so the sandbox, the
task's state and the harness's own process state are rebuilt; the model always sees the
real tool outputs of this rollout.

Replay is blind and per branch: each recorded call continues an earlier recorded call
(`PrefixCall.parent`) or starts a new branch (a sub-agent, say). A call of this rollout
is served the next unserved recorded call of its own branch, whatever the environment
returned; concurrent branches therefore get their own recordings regardless of arrival
order. A call whose branch has no recorded call left, or that continues a call that was
not replayed, is sampled. A rollout that ends before the cut just ends.

Replayed completions are committed as context (`MessageNode.replayed`, mask False).
`PrefixReplay.finish` records, for monitoring only, how many calls were replayed and how
many saw an observation (the prompt tokens added since the call they continue) that
differs from the recording.
"""

from __future__ import annotations

import hashlib
from array import array
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from verifiers.v1.types import FinishReason

if TYPE_CHECKING:
    from verifiers.v1.graph import PendingTurn
    from verifiers.v1.trace import Trace


def observation_hash(prompt_ids: list[int], previous: list[int]) -> str:
    """Hash of the prompt tokens added after `previous` (the continued call's prompt
    and completion), or of the whole prompt when it does not extend `previous`."""
    if previous and prompt_ids[: len(previous)] == previous:
        prompt_ids = prompt_ids[len(previous) :]
    return hashlib.sha256(array("i", prompt_ids).tobytes()).hexdigest()


class PrefixCall(BaseModel):
    completion_ids: list[int]
    finish_reason: FinishReason = None
    observation_hash: str
    """`observation_hash` of the source call's prompt, compared (not gated) on replay."""
    parent: int | None = None
    """Index of the recorded call this one continues; None starts a branch."""


class Prefix(BaseModel):
    calls: list[PrefixCall]
    """The source's first committed calls, in call order; the next call is sampled."""
    agent: str = "agent"
    """The episode agent whose rollout replays the prefix (the source trace's agent)."""
    source: dict[str, Any] = Field(default_factory=dict)
    """Caller metadata about the source rollout, copied to `trace.info["prefix"]`."""

    @classmethod
    def from_trace(
        cls, trace: Trace, cut: int | None = None, source: dict[str, Any] | None = None
    ) -> Prefix:
        """The prefix that replays `trace`'s first `cut` committed calls (all by default)."""
        if "prefix" in trace.info:
            raise ValueError("cannot build a prefix from a prefix-replayed trace")
        committed = [call for call in trace.calls if call.node is not None]
        if cut is not None and not 0 <= cut <= len(committed):
            raise ValueError(f"cut {cut} outside the trace's {len(committed)} calls")
        index_of = {call.node: index for index, call in enumerate(committed)}
        calls: list[PrefixCall] = []
        full_ids: list[list[int]] = []
        for call in committed[:cut]:
            node = call.node
            parts, parent, current = [], None, node
            while current is not None:
                parts.append(trace.nodes[current].token_ids)
                if parent is None and current != node and current in index_of:
                    parent = index_of[current]
                current = trace.nodes[current].parent
            ids = [token for part in reversed(parts) for token in part]
            prompt_len = len(ids) - sum(trace.nodes[node].mask)
            previous = full_ids[parent] if parent is not None else []
            calls.append(
                PrefixCall(
                    completion_ids=ids[prompt_len:],
                    finish_reason=call.finish_reason,
                    observation_hash=observation_hash(ids[:prompt_len], previous),
                    parent=parent,
                )
            )
            full_ids.append(ids)
        return cls(calls=calls, agent=trace.agent.name, source=source or {})


class PrefixReplay:
    """One rollout's replay state, consulted by the train client on every call."""

    def __init__(self, prefix: Prefix) -> None:
        self.prefix = prefix
        self.unserved = list(range(len(prefix.calls)))
        self.full_ids: dict[int, list[int]] = {}
        """Prompt and completion tokens of each served call, by recorded index."""
        self.obs_changed: list[bool] = []

    def take(
        self, prompt_ids: list[int], turn: PendingTurn | None
    ) -> PrefixCall | None:
        """The recorded call to serve next on this prompt's branch, or None to sample."""
        if not self.unserved:
            return None
        parent = None
        path = turn.prefix_node_ids if turn is not None else []
        for end in range(len(path), 0, -1):
            node = turn.trace.nodes[path[end - 1]]
            if node.sampled:
                if not node.replayed:
                    return None  # continues a live call
                # The served call that produced this node: the one whose tokens are its path.
                ids = [t for nid in path[:end] for t in turn.trace.nodes[nid].token_ids]
                parent = next((i for i, f in self.full_ids.items() if f == ids), None)
                if parent is None:
                    return None
                break
        calls = self.prefix.calls
        branch = [index for index in self.unserved if calls[index].parent == parent]
        if not branch:
            return None
        obs = observation_hash(prompt_ids, self.full_ids.get(parent, []))
        # Concurrent new branches (sub-agents) prefer the recording they reproduce exactly.
        index = next((i for i in branch if calls[i].observation_hash == obs), branch[0])
        call = calls[index]
        self.unserved.remove(index)
        self.full_ids[index] = [*prompt_ids, *call.completion_ids]
        self.obs_changed.append(obs != call.observation_hash)
        return call

    def finish(self, trace: Trace) -> None:
        """Record the replay's outcome on the trace."""
        realized = len(self.obs_changed)
        trace.info["prefix"] = {
            "cut": len(self.prefix.calls),
            "realized_cut": realized,
            "obs_changed": self.obs_changed,
            "source": self.prefix.source,
        }
        trace.record_metrics(
            {
                "prefix/realized_cut": realized,
                "prefix/obs_changed_frac": sum(self.obs_changed) / realized
                if realized
                else 0.0,
            }
        )
