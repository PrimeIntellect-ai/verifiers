"""Replay recorded model replies: resume an interrupted rollout, or drive a seat verbatim.

A rollout whose context carries a `Replay` claims a free recorded trace of its agent and
continues under that trace's id; a failed attempt frees its recording, so the attempt that
retries it replays the recording again, under its own id. Each model request is answered
with the recorded reply that followed the prompt's last assistant message (or opened a
branch, when the prompt has none). The harness still runs every tool call, so its runtime
and program state are rebuilt.

Resume (the default) replays a reply only when the prompt is its recorded prompt: every
message identical (`exact`), or every assistant message identical and every other message
in the same role, answering the same call, and differing only in volatile runs such as an
id, a hostname, a clock time, or a duration (`structural`). Any other request samples live,
and so does everything after it, since its prompt carries the live reply.

Verbatim replays the recording whatever the tools returned or other agents said, never
samples live, and stops the rollout (`replay_ended`) once the recording has no reply left.
It claims a recording by agent alone, so a recorded seat can drive a seat of another task."""

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from verifiers.v1 import graph
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, Message, Response, Tool, ToolMessage

ReplayMatch = Literal["exact", "structural", "verbatim"]


@dataclass(eq=False)
class _Recording:
    trace: Trace
    adopted: bool = False
    used: set[int] = field(default_factory=set)
    hashes: list[str] = field(default_factory=list)
    replies: dict[str | None, list[int]] = field(default_factory=dict)
    """Recorded replies by the hash of the assistant message their prompt ends after."""

    def __post_init__(self) -> None:
        nodes = self.trace.nodes
        self.hashes = [graph.message_hash(node.message) for node in nodes]
        for reply, node in enumerate(nodes):
            if not node.sampled:
                continue
            last = node.parent
            while last is not None and not isinstance(
                nodes[last].message, AssistantMessage
            ):
                last = nodes[last].parent
            key = None if last is None else self.hashes[last]
            self.replies.setdefault(key, []).append(reply)

    def prompt(self, reply: int) -> list[int]:
        """The recorded prompt of `reply`, root first."""
        nodes: list[int] = []
        current = self.trace.nodes[reply].parent
        while current is not None:
            nodes.append(current)
            current = self.trace.nodes[current].parent
        return nodes[::-1]


class Replay:
    def __init__(self, traces: list[Trace], *, verbatim: bool = False) -> None:
        self.verbatim = verbatim
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
            and (self.verbatim or recording.trace.task.key == task_key)
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

    def drives(self, trace_id: str) -> bool:
        """Whether this trace follows its recording verbatim, never sampling live."""
        return self.verbatim and trace_id in self._bound

    def take(
        self, trace_id: str, prompt: list[Message], tools: list[Tool] | None
    ) -> tuple[Response, ReplayMatch] | None:
        """The recorded reply to `prompt` and how the prompt matched, or None."""
        recording = self._bound.get(trace_id)
        if not prompt or recording is None:
            return None
        last = next(
            (m for m in reversed(prompt) if isinstance(m, AssistantMessage)), None
        )
        key = None if last is None else graph.message_hash(last)
        candidates = [
            reply
            for reply in recording.replies.get(key, [])
            if reply not in recording.used
        ]
        matches = [
            (match, reply)
            for reply in candidates
            if (match := _compare(recording, reply, prompt, tools)) is not None
        ]
        if matches:
            match, reply = min(matches, key=lambda pair: pair[0] != "exact")
        elif self.verbatim and candidates:
            match, reply = "verbatim", candidates[0]
        else:
            return None
        recording.used.add(reply)
        source = recording.trace
        call = next((c for c in source.calls if c.node == reply), None)
        response = Response(
            id=f"replay-{reply}",
            created=0,
            model=call.model if call and call.model else "",
            message=source.nodes[reply].message.model_copy(deep=True),
            finish_reason=call.finish_reason if call else None,
            usage=call.usage if call else None,
        )
        return response, "verbatim" if self.verbatim else match


def _compare(
    recording: _Recording, reply: int, prompt: list[Message], tools: list[Tool] | None
) -> Literal["exact", "structural"] | None:
    """How `prompt` matches the recorded prompt of `reply`, or None when it does not."""
    nodes = recording.prompt(reply)
    if len(nodes) != len(prompt):
        return None
    recorded_tools = recording.trace.nodes[nodes[0]].tools
    exact = recorded_tools == list(tools or [])
    if not exact and _toolset(recorded_tools) != _toolset(tools):
        return None
    for node, message in zip(nodes, prompt):
        if recording.hashes[node] == graph.message_hash(message):
            continue
        exact = False
        recorded = recording.trace.nodes[node].message
        if (
            isinstance(message, AssistantMessage)
            or recorded.role != message.role
            or (
                isinstance(message, ToolMessage)
                and getattr(recorded, "tool_call_id", None) != message.tool_call_id
            )
            or _media(recorded) != _media(message)
            or _volatile_masked(_text(recorded)) != _volatile_masked(_text(message))
        ):
            return None
    return "exact" if exact else "structural"


def _text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(getattr(part, "text", "") for part in message.content or [])


def _media(message: Message) -> list[str]:
    """A message's non-text parts (images), which must match exactly."""
    if isinstance(message.content, str):
        return []
    return [part.model_dump_json() for part in message.content if part.type != "text"]


def _toolset(tools: list[Tool] | None) -> str:
    """The tools a prompt declares, in any order, with volatile runs masked."""
    declared = sorted(
        json.dumps(tool.model_dump(mode="json", exclude_none=True), sort_keys=True)
        for tool in tools or []
    )
    return _volatile_masked("\n".join(declared))


_VOLATILE = re.compile(
    r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"  # uuids
    r"|\b(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{8,}\b"  # hashes, hostnames
    r"|(?<=[-_/])[0-9a-fA-F]{8,}\b"  # ids in names and paths
    r"|\b\d+(?:\.\d+)?\s?(?:ns|us|µs|ms|s|secs?|seconds?)\b"  # durations
    r"|\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b"  # clock times
    r"|\b\d{10,}\b"  # epoch timestamps
)
"""Runs that differ between two runs of one rollout without changing what it observed."""


def _volatile_masked(text: str) -> str:
    return _VOLATILE.sub("#", text)
