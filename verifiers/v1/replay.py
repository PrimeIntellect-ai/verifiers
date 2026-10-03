"""Replay recorded model replies, in one of two modes.

`resume` (the default) continues an interrupted rollout: it replays recorded replies while
each prompt is exactly its recorded prompt (`exact`), then samples live from the first
prompt that differs.

`playback` plays a recording back, on any task: it replays every recorded reply whatever the
tools returned or other agents said, never samples live, and stops the rollout
(`replay_ended`) when the recording runs out. A reply played onto a prompt that differs from
its recorded prompt is `forced`: the recorded model never saw that prompt.

A rollout whose context carries a `Replay` claims a free recording of its agent (on its task,
in resume) and runs under the recording's trace id. A failed attempt frees its recording, so
the attempt that retries it replays the recording again, under its own id. Each model
request is answered with the recorded reply that followed the prompt's last assistant
message, or that opened a branch when the prompt has none. The harness still runs every tool
call, so its runtime and program state are rebuilt."""

from dataclasses import dataclass, field
from typing import Literal

from verifiers.v1 import graph
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, Message, Response, Tool

ReplayMode = Literal["resume", "playback"]
ReplayMatch = Literal["exact", "forced"]


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

    def is_prompt(self, reply: int, hashes: list[str], tools: list[Tool]) -> bool:
        """Whether a prompt with these message hashes and tools is `reply`'s recorded prompt."""
        nodes: list[int] = []
        current = self.trace.nodes[reply].parent
        while current is not None:
            nodes.append(current)
            current = self.trace.nodes[current].parent
        nodes.reverse()
        return [self.hashes[n] for n in nodes] == hashes and self.trace.nodes[
            nodes[0]
        ].tools == tools


class Replay:
    def __init__(self, traces: list[Trace], *, mode: ReplayMode = "resume") -> None:
        self.mode = mode
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
            and (self.mode == "playback" or recording.trace.task.key == task_key)
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

    def plays_back(self, trace_id: str) -> bool:
        """Whether this trace plays its recording back, never sampling live."""
        return self.mode == "playback" and trace_id in self._bound

    def take(
        self, trace_id: str, prompt: list[Message], tools: list[Tool] | None
    ) -> tuple[Response, ReplayMatch] | None:
        """The recorded reply for `prompt` and how it was matched, or None to sample live."""
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
        hashes = [graph.message_hash(message) for message in prompt]
        exact = [
            r for r in candidates if recording.is_prompt(r, hashes, list(tools or []))
        ]
        if exact:
            reply, match = exact[0], "exact"
        elif self.mode == "playback" and candidates:
            reply, match = candidates[0], "forced"
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
        return response, match
