"""Resume interrupted rollouts by replaying their recorded model replies.

A rollout whose context carries a `Replay` claims the first unclaimed recorded trace of
its agent on its task and continues under that trace's id. Each model request is matched to a path of
the recording and answered with the recorded reply; the harness still runs every tool call,
so its runtime and program state are rebuilt. A prompt matches its exact recorded path, or
else (`structural`) the path with the same tools and structure: every assistant message
identical, every other message in the same role (and for a tool result, the same call), its
text free to differ — a timestamp, a hostname, a session path. A request no path answers
goes to the live model, and so does everything after it, since its prompt carries the live
reply."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from verifiers.v1 import graph
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, Message, Response, Tool, ToolMessage

ReplayMatch = Literal["exact", "structural"]


@dataclass
class _Recording:
    trace: Trace
    children: dict[int | None, list[int]] = field(default_factory=dict)
    hashes: list[str] = field(default_factory=list)
    used: Counter[int] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        for node_id, node in enumerate(self.trace.nodes):
            self.children.setdefault(node.parent, []).append(node_id)
            self.hashes.append(graph.message_hash(node.message))


class Replay:
    def __init__(self, traces: list[Trace]) -> None:
        self._unclaimed = list(traces)
        self._claimed: dict[str, _Recording] = {}

    def claim(self, agent: str, task_key: str) -> str | None:
        """The recorded trace id a newly minted trace of `agent` on `task_key` continues."""
        for trace in self._unclaimed:
            if trace.agent.name == agent and trace.task.key == task_key:
                self._unclaimed.remove(trace)
                self._claimed[trace.id] = _Recording(trace)
                return trace.id
        return None

    def take(
        self, trace_id: str, prompt: list[Message], tools: list[Tool] | None
    ) -> tuple[Response, ReplayMatch] | None:
        """The recorded reply to `prompt` and how the prompt matched, or None to sample live."""
        recording = self._claimed.get(trace_id)
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
        replies = [
            n for n in recording.children.get(parent, []) if source.nodes[n].sampled
        ]
        if recording.used[parent] >= len(replies):
            return None
        node = replies[recording.used[parent]]
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
    source = recording.trace
    parent: int | None = None
    for message in prompt:
        options = recording.children.get(parent, [])
        if parent is None:
            options = [n for n in options if source.nodes[n].tools == list(tools or [])]
        key = graph.message_hash(message)
        exact = [n for n in options if recording.hashes[n] == key]
        if exact:
            parent = exact[0]
            continue
        if isinstance(message, AssistantMessage):
            return None
        alike = [
            n
            for n in options
            if source.nodes[n].message.role == message.role
            and (
                not isinstance(message, ToolMessage)
                or source.nodes[n].message.tool_call_id == message.tool_call_id
            )
        ]
        if not alike:
            return None
        text = _text(message)
        parent = max(
            alike, key=lambda n: _overlap(_text(source.nodes[n].message), text)
        )
    return parent


def _text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(getattr(part, "text", "") for part in message.content or [])


def _overlap(a: str, b: str) -> int:
    """Shared leading plus trailing characters: how alike two variants of one message are."""
    head = next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )
    tail = next(
        (i for i, (x, y) in enumerate(zip(reversed(a), reversed(b))) if x != y),
        min(len(a), len(b)),
    )
    return head + tail
