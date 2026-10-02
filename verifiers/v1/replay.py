"""Resume interrupted rollouts by replaying their recorded model replies.

A rollout whose context carries a `Replay` claims a free recorded trace of its agent on
its task and continues under that trace's id. A failed attempt frees its recording, so
the attempt that retries it replays the recording again, under its own id. Each model
request is matched to a path of the recording and answered with the recorded reply; the
harness still runs every tool call, so its runtime and program state are rebuilt.

A prompt matches its exact recorded path, or else (`structural`) a path with the same
tools and structure: every assistant message identical, every other message in the same
role (a tool result answering the same call) and differing only in volatile runs such as
an id, a hostname, a clock time, or a duration. A request no path answers goes to the
live model, and so does everything after it, since its prompt carries the live reply."""

import difflib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from verifiers.v1 import graph
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, Message, Response, Tool, ToolMessage

ReplayMatch = Literal["exact", "structural"]


@dataclass(eq=False)
class _Recording:
    trace: Trace
    children: dict[int | None, list[int]] = field(default_factory=dict)
    hashes: list[str] = field(default_factory=list)
    used: Counter[int] = field(default_factory=Counter)
    adopted: bool = False
    _masked: dict[int, str] = field(default_factory=dict)
    _toolsets: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for node_id, node in enumerate(self.trace.nodes):
            self.children.setdefault(node.parent, []).append(node_id)
            self.hashes.append(graph.message_hash(node.message))

    def toolset(self, root: int) -> str:
        if root not in self._toolsets:
            self._toolsets[root] = _toolset(self.trace.nodes[root].tools)
        return self._toolsets[root]

    def masked(self, node: int) -> str:
        if node not in self._masked:
            self._masked[node] = _volatile_masked(_text(self.trace.nodes[node].message))
        return self._masked[node]

    def next_reply(self, parent: int) -> int | None:
        """The parent's first recorded reply not yet replayed."""
        replies = [
            n for n in self.children.get(parent, []) if self.trace.nodes[n].sampled
        ]
        used = self.used[parent]
        return replies[used] if used < len(replies) else None


class Replay:
    def __init__(self, traces: list[Trace]) -> None:
        self._recordings = [_Recording(trace) for trace in traces]
        self._bound: dict[str, _Recording] = {}

    def claim(self, agent: str, task_key: str, trace_id: str) -> str:
        """Bind a newly minted trace of `agent` on `task_key` to a free recording and
        return the id it runs under: the recording's own id the first time the recording
        is claimed, else (a retry after a failed attempt) the trace's own."""
        bound = {id(recording) for recording in self._bound.values()}
        free = [
            recording
            for recording in self._recordings
            if id(recording) not in bound
            and recording.trace.agent.name == agent
            and recording.trace.task.key == task_key
        ]
        if not free:
            return trace_id
        recording = min(free, key=lambda recording: recording.adopted)
        recording.used.clear()
        if not recording.adopted:
            recording.adopted = True
            trace_id = recording.trace.id
        self._bound[trace_id] = recording
        return trace_id

    def release(self, trace_id: str) -> None:
        """Free a failed trace's recording for the attempt that retries it."""
        self._bound.pop(trace_id, None)

    def take(
        self, trace_id: str, prompt: list[Message], tools: list[Tool] | None
    ) -> tuple[Response, ReplayMatch] | None:
        """The recorded reply to `prompt` and how the prompt matched, or None to sample live."""
        recording = self._bound.get(trace_id)
        if not prompt or recording is None:
            return None
        source = recording.trace
        turn = graph.prepare_turn(source, prompt, tools)
        match: ReplayMatch = "exact"
        if len(turn.prefix_node_ids) == len(prompt):
            parent = turn.prefix_node_ids[-1]
        elif (parent := _structural_match(recording, prompt, tools)) is not None:
            match = "structural"
        else:
            return None
        node = recording.next_reply(parent)
        if node is None:
            return None
        recording.used[parent] += 1
        call = next((c for c in source.calls if c.node == node), None)
        response = Response(
            id=f"replay-{node}",
            created=0,
            model=call.model if call and call.model else "",
            message=source.nodes[node].message.model_copy(deep=True),
            finish_reason=call.finish_reason if call else None,
            usage=call.usage if call else None,
        )
        return response, match


def _structural_match(
    recording: _Recording, prompt: list[Message], tools: list[Tool] | None
) -> int | None:
    """The recorded node the prompt ends at, matched by structure. Every branch that fits
    the prompt so far stays a candidate, so siblings that agree on a long prefix resolve
    at the message that tells them apart. Among the full matches that still have a reply,
    the one with the most exact messages, then the most similar text, wins."""
    source = recording.trace
    toolset = _toolset(tools)
    frontier: dict[int | None, tuple[int, float]] = {None: (0, 0.0)}
    for message in prompt:
        key = graph.message_hash(message)
        exact: dict[int | None, tuple[int, float]] = {}
        alike: list[tuple[int, tuple[int, float]]] = []
        for parent, score in frontier.items():
            for n in recording.children.get(parent, []):
                if parent is None and recording.toolset(n) != toolset:
                    continue
                if recording.hashes[n] == key:
                    exact[n] = (score[0] + 1, score[1])
                elif _same_structure(source.nodes[n].message, message):
                    alike.append((n, score))
        frontier = exact
        if alike:
            text = _text(message)
            masked = _volatile_masked(text)
            alike = [(n, score) for n, score in alike if recording.masked(n) == masked]
            for n, (matched, similarity) in alike:
                if len(alike) > 1:
                    recorded = _text(source.nodes[n].message)
                    similarity += difflib.SequenceMatcher(None, recorded, text).ratio()
                frontier[n] = (matched, similarity)
        if not frontier:
            return None
    answered = [
        n for n in frontier if n is not None and recording.next_reply(n) is not None
    ]
    return max(answered, key=lambda n: frontier[n]) if answered else None


def _same_structure(recorded: Message, message: Message) -> bool:
    """Whether a recorded message may stand for `message` when only volatile text differs:
    never an assistant message (the model's own words must match exactly), else the same
    role, and for a tool result the same call."""
    if isinstance(message, AssistantMessage) or recorded.role != message.role:
        return False
    return not isinstance(message, ToolMessage) or (
        isinstance(recorded, ToolMessage)
        and recorded.tool_call_id == message.tool_call_id
    )


def _text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(getattr(part, "text", "") for part in message.content or [])


_VOLATILE = re.compile(
    r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"  # uuids
    r"|\b(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{8,}\b"  # hashes, hostnames
    r"|\b\d+(?:\.\d+)?\s?(?:ns|us|µs|ms|s|secs?|seconds?)\b"  # durations
    r"|\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b"  # clock times
    r"|\b\d{10,}\b"  # epoch timestamps
)
"""Runs that differ between two runs of one rollout without changing what it observed."""


def _toolset(tools: list[Tool] | None) -> str:
    """The tools a prompt declares, in any order, with volatile runs masked."""
    declared = sorted(
        json.dumps(tool.model_dump(mode="json", exclude_none=True), sort_keys=True)
        for tool in tools or []
    )
    return _volatile_masked("\n".join(declared))


def _volatile_masked(text: str) -> str:
    return _VOLATILE.sub("#", text)
