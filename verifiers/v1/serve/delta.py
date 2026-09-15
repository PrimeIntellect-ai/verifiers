"""Turn-by-turn episode deltas over the env-serve wire.

The worker streams a served episode as it grows. Each trace announces its own changes
(`Trace.notify`, fired by the rollout at every phase change and by the interception proxy
after every recorded turn); the `DeltaStreamer` then diffs the run's live traces against
what it has already sent and ships only the new part — the trace header once, then
appended nodes / calls / errors, semantic links landing on earlier nodes, the scalar
fields whose value changed (timing spans, stop condition, rewards, ...), and the
`pending` preview — the messages of the request in flight that no node holds yet, so a
watcher sees a tool result before the model has answered it. The `Trace` is
append-only at turn granularity (a turn's nodes are committed complete, with their
tokens), so every byte of the episode crosses the wire once and the stream costs about
what a single reply would; the reply that ends the run carries only the episode head and
per-trace counts the client checks its assembly against.

The client applies deltas into raw dicts (`EpisodeAssembly`) — cheap enough to hand a
caller after every turn — and validates the assembled record into a `WireEpisode` once,
when the run's reply lands.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Self

import msgpack
from pydantic import BaseModel

from verifiers.v1.serve.encoding import msgpack_encoder

logger = logging.getLogger(__name__)

HEADER_FIELDS = ("version", "id", "verifiers", "task")
"""Sent once, when a trace first appears."""

LIST_FIELDS = (
    "nodes",
    "calls",
    "errors",
    "extra_usage",
    "request_rewrites",
    "response_rewrites",
)
"""Append-only on the worker: each delta carries the items past the sent count."""

SCALAR_FIELDS = (
    "agent",
    "tools",
    "mm_token_type_id_map",
    "rewards",
    "metrics",
    "info",
    "root_reply",
    "is_completed",
    "ok",
    "stop_condition",
    "timing",
)
"""Re-sent whole whenever their dumped value changes."""


def pack(payload: Any) -> bytes:
    return msgpack.packb(payload, default=msgpack_encoder, use_bin_type=True)


def unpack(data: bytes) -> Any:
    # Trace dicts carry int keys (`mm_token_type_id_map`, node-indexed links).
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


def dump(model: BaseModel, **kwargs: Any) -> dict:
    # Task data and agent configs are subclasses of their declared types; dump the
    # runtime type so env-specific fields survive the wire.
    return model.model_dump(mode="python", serialize_as_any=True, **kwargs)


class TraceCursor:
    """What of one trace has been sent."""

    def __init__(self) -> None:
        self.sent = dict.fromkeys(LIST_FIELDS, 0)
        self.links: list[int] = []
        """Per sent node, how many of its semantic links went out with or after it."""
        self.scalars: dict[str, bytes] = {}
        self.pending: bytes = pack([])


class DeltaStreamer:
    """Sends a `RunSlot`'s trace changes as deltas, one flush per burst of changes.

    `watch` subscribes a minted trace (pass it as `run_slot`'s `on_trace`); each
    `Trace.notify` schedules a flush, and changes landing in the same loop iteration
    ride one flush. Use as an async context manager around the rollout: a clean exit
    flushes the final state (the slot then holds the finished episode's traces), a
    cancelled rollout flushes nothing — its client has already gone."""

    def __init__(self, slot: Any, send: Callable[[bytes], Awaitable[None]]) -> None:
        self.slot = slot
        self.send = send
        self.cursors: dict[str, TraceCursor] = {}
        self._scheduled = False
        self._flushes: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        for task in self._flushes:
            task.cancel()
        if exc_type is None:
            await self.flush()

    def watch(self, trace: Any) -> None:
        trace.watch(self._changed)
        self._changed(trace)

    def _changed(self, trace: Any) -> None:
        if self._scheduled:
            return
        self._scheduled = True
        asyncio.get_running_loop().call_soon(self._start_flush)

    def _start_flush(self) -> None:
        self._scheduled = False
        task = asyncio.create_task(self.flush())
        self._flushes.add(task)
        task.add_done_callback(self._flushes.discard)

    async def flush(self) -> None:
        # Serialized: deltas must leave in diff order, and the reply after the last.
        async with self._lock:
            for delta in self.diff():
                try:
                    await self.send(pack(delta))
                except Exception:  # a lost delta only delays the client's view
                    logger.warning(
                        "failed to send delta for %s", delta.get("trace"), exc_info=True
                    )

    def diff(self) -> list[dict]:
        traces = list(self.slot.traces)
        live = {trace.id for trace in traces}
        deltas: list[dict] = []
        for trace_id in [trace_id for trace_id in self.cursors if trace_id not in live]:
            # A retried attempt abandons its traces; the client drops them too.
            del self.cursors[trace_id]
            deltas.append({"trace": trace_id, "discard": True})
        for trace in traces:
            delta: dict[str, Any] = {"trace": trace.id}
            cursor = self.cursors.get(trace.id)
            if cursor is None:
                cursor = self.cursors[trace.id] = TraceCursor()
                delta["open"] = dump(trace, include=set(HEADER_FIELDS))
            links: dict[int, list[dict]] = {}
            for index in range(cursor.sent["nodes"]):
                node_links = trace.nodes[index].semantic_parents
                if len(node_links) > cursor.links[index]:
                    links[index] = [
                        dump(link) for link in node_links[cursor.links[index] :]
                    ]
                    cursor.links[index] = len(node_links)
            if links:
                delta["links"] = links
            for field in LIST_FIELDS:
                items = getattr(trace, field)
                sent = cursor.sent[field]
                if len(items) > sent:
                    delta[field] = [dump(item) for item in items[sent:]]
                    cursor.sent[field] = len(items)
                    if field == "nodes":
                        cursor.links.extend(
                            len(node.semantic_parents) for node in items[sent:]
                        )
            changed: dict[str, Any] = {}
            for field, value in dump(trace, include=set(SCALAR_FIELDS)).items():
                packed = pack(value)
                if cursor.scalars.get(field) != packed:
                    cursor.scalars[field] = packed
                    changed[field] = value
            if changed:
                delta["set"] = changed
            pending = [dump(message) for message in trace.pending]
            packed = pack(pending)
            if packed != cursor.pending:
                cursor.pending = packed
                # nodes landing clear the preview on the client by themselves
                if pending or "nodes" not in delta:
                    delta["pending"] = pending
            if len(delta) > 1:
                deltas.append(delta)
        return deltas


class EpisodeAssembly:
    """The client's growing picture of one served episode: raw trace dicts in arrival
    order, each shaped like a dumped `Trace`, plus the count of deltas applied."""

    def __init__(self) -> None:
        self.traces: dict[str, dict] = {}
        self.updates = 0

    def apply(self, delta: dict) -> None:
        trace_id = delta["trace"]
        if delta.get("discard"):
            self.traces.pop(trace_id, None)
            self.updates += 1
            return
        trace = self.traces.get(trace_id)
        if trace is None:
            trace = self.traces[trace_id] = {
                **delta["open"],
                **{field: [] for field in LIST_FIELDS},
                "pending": [],
            }
        for index, links in (delta.get("links") or {}).items():
            trace["nodes"][int(index)]["semantic_parents"].extend(links)
        for field in LIST_FIELDS:
            if field in delta:
                trace[field].extend(delta[field])
        trace.update(delta.get("set") or {})
        # a committed turn absorbs the preview; an explicit preview replaces it
        if "nodes" in delta:
            trace["pending"] = []
        if "pending" in delta:
            trace["pending"] = delta["pending"]
        self.updates += 1

    def finish(self, head: dict, summaries: list[TraceSummary]) -> dict:
        """The full episode record: `head` (the episode without its traces) joined with
        the assembled traces in the server's order. A count mismatch means a delta was
        lost, which the wire never does in a live connection — fail loudly."""
        traces = []
        for summary in summaries:
            trace = self.traces.get(summary.id)
            if trace is None:
                raise RuntimeError(f"served episode is missing trace {summary.id}")
            trace = {key: value for key, value in trace.items() if key != "pending"}
            if (
                len(trace["nodes"]) != summary.nodes
                or len(trace["calls"]) != summary.calls
            ):
                raise RuntimeError(
                    f"served trace {summary.id} assembled {len(trace['nodes'])} nodes / "
                    f"{len(trace['calls'])} calls, server reports {summary.nodes} / {summary.calls}"
                )
            traces.append(trace)
        return {**head, "traces": traces}


class TraceSummary(BaseModel):
    """A finished trace's identity and size, for the client's assembly check."""

    id: str
    nodes: int
    calls: int
