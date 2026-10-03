"""Prefix replay: start a rollout from the first model calls of a finished rollout.

A `Prefix` holds a source trace's first committed calls. The rollout runs normally from
the task start, but the train client answers its first `len(calls)` model calls with the
recorded completions, in order, without sampling. The harness executes the recorded
actions itself, so the sandbox, the task's state and the harness's own process state are
rebuilt; the model always sees the real tool outputs of this rollout. Replay is blind:
calls are served by position, whatever the environment returned. Later calls, and every
call once the recording is exhausted, are sampled. A rollout that ends before the cut
just ends.

`PrefixReplay.finish` drops the replayed completions from the loss and records, for
monitoring only, how many calls were replayed and how many saw an observation (the
prompt tokens added since the previous call) that differs from the recording.
"""

from __future__ import annotations

import hashlib
from array import array
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from verifiers.v1.types import FinishReason

if TYPE_CHECKING:
    from verifiers.v1.trace import Trace


def observation_hash(prompt_ids: list[int], previous: list[int]) -> str:
    """Hash of the prompt tokens added after `previous` (the last call's prompt and
    completion), or of the whole prompt when it does not extend `previous`."""
    if previous and prompt_ids[: len(previous)] == previous:
        prompt_ids = prompt_ids[len(previous) :]
    return hashlib.sha256(array("i", prompt_ids).tobytes()).hexdigest()


class PrefixCall(BaseModel):
    completion_ids: list[int]
    finish_reason: FinishReason = None
    observation_hash: str
    """`observation_hash` of the source call's prompt, compared (not gated) on replay."""


class Prefix(BaseModel):
    calls: list[PrefixCall]
    """The source's first committed calls, in call order; the next call is sampled."""
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
        calls, previous = [], []
        for call in committed[:cut]:
            parts, current = [], call.node
            while current is not None:
                parts.append(trace.nodes[current].token_ids)
                current = trace.nodes[current].parent
            ids = [token for part in reversed(parts) for token in part]
            prompt_len = len(ids) - sum(trace.nodes[call.node].mask)
            calls.append(
                PrefixCall(
                    completion_ids=ids[prompt_len:],
                    finish_reason=call.finish_reason,
                    observation_hash=observation_hash(ids[:prompt_len], previous),
                )
            )
            previous = ids
        return cls(calls=calls, source=source or {})


class PrefixReplay:
    """One rollout's replay state, consulted by the train client on every call."""

    def __init__(self, prefix: Prefix) -> None:
        self.prefix = prefix
        self.obs_changed: list[bool] = []
        self.previous: list[int] = []

    def take(self, prompt_ids: list[int]) -> PrefixCall | None:
        """The recorded call to serve next, or None once the recording is exhausted."""
        if len(self.obs_changed) == len(self.prefix.calls):
            return None
        call = self.prefix.calls[len(self.obs_changed)]
        obs = observation_hash(prompt_ids, self.previous)
        self.obs_changed.append(obs != call.observation_hash)
        self.previous = [*prompt_ids, *call.completion_ids]
        return call

    def finish(self, trace: Trace) -> None:
        """Turn the replayed completions into context and record the outcome."""
        for call in trace.calls:
            if call.replayed and call.node is not None:
                node = trace.nodes[call.node]
                node.mask = [False] * len(node.mask)
                node.logprobs = []
                node.sampling_mask = None
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
