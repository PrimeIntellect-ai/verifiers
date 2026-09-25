"""Incremental trace deltas shared by served episodes and Flow.

The worker streams a served episode as it grows. Each trace announces its own changes
(`Trace.notify`, fired by the rollout at every phase change and by the interception proxy
after every recorded turn); the `DeltaStreamer` then diffs the run's live traces against
what it has already sent and ships only the new part — the trace header once, then
appended nodes / calls / errors, semantic links and routing-row repairs on earlier nodes, the scalar
fields whose value changed (timing spans, stop condition, rewards, ...), and the
`pending` preview — the messages of the request in flight that no node holds yet, so a
watcher sees a tool result before the model has answered it. The `Trace` is
append-only at turn granularity except for links and the last routing row of a node,
which the next prefill can repair. Apart from these and the preview, every byte of the
episode crosses the wire once and the stream costs about what a single reply would; the
reply that ends the run carries only the episode head and per-trace counts the client
checks its assembly against. A cursor advances only once its delta is on the wire, so a
send that fails is simply diffed again at the next flush.

The client applies deltas into raw dicts (`EpisodeAssembly`) — cheap enough to hand a
caller after every turn — and validates the assembled record into a `WireEpisode` once,
when the run's reply lands.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any, Self

import msgpack
import numpy as np
from pydantic import BaseModel

from verifiers.v1.graph import _decode_ndarray, _encode_ndarray
from verifiers.v1.serve.encoding import msgpack_encoder

if TYPE_CHECKING:
    from verifiers.v1.trace import Trace

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
    "num_input_tokens",
    "num_output_tokens",
    "num_total_tokens",
)
"""Re-sent whole whenever their dumped value changes."""


def pack(payload: Any) -> bytes:
    return msgpack.packb(payload, default=msgpack_encoder, use_bin_type=True)


def unpack(data: bytes) -> Any:
    # Node updates are keyed by node index (int).
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
        self.final_rows: dict[int, bytes] = {}
        """Per node that carries routing, its final row as packed when last sent."""
        self.scalars: dict[str, bytes] = {}
        self.pending: bytes = pack([])


class DeltaStreamer:
    """Sends the current traces' changes as deltas, one flush per burst of changes.

    `watch` subscribes a minted trace (pass it as the runner's `on_trace`); each
    `Trace.notify` schedules a flush, and changes landing in the same loop iteration
    ride one flush. Use as an async context manager around the rollout: a clean exit
    flushes the final state; a cancelled rollout settles pending sends without a final flush."""

    def __init__(
        self,
        traces: Callable[[], Iterable[Trace]],
        send: Callable[[dict], Awaitable[None]],
    ) -> None:
        self.traces = traces
        self.send = send
        self.cursors: dict[str, TraceCursor] = {}
        self._scheduled: asyncio.Handle | None = None
        self._flushes: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._scheduled is not None:
            self._scheduled.cancel()
            self._scheduled = None
        if exc_type is None:
            # let in-flight flushes finish, then send whatever the run left behind
            await asyncio.gather(*self._flushes, return_exceptions=True)
            await self.flush()
        else:
            for task in self._flushes:
                task.cancel()
            await asyncio.gather(*self._flushes, return_exceptions=True)

    def watch(self, trace: Trace) -> None:
        trace.watch(self._changed)
        self._changed(trace)

    def _changed(self, trace: Trace) -> None:
        if self._scheduled is not None:
            return
        self._scheduled = asyncio.get_running_loop().call_soon(self._start_flush)

    def _start_flush(self) -> None:
        self._scheduled = None
        task = asyncio.create_task(self.flush())
        self._flushes.add(task)
        task.add_done_callback(self._flushes.discard)

    async def flush(self) -> None:
        # Serialized: deltas must leave in diff order, and the reply after the last.
        async with self._lock:
            for trace_id, delta, cursor in self.diff():
                try:
                    await self.send(delta)
                except Exception:  # the cursor stays put: the next flush diffs it again
                    logger.warning(
                        "failed to send delta for %s", trace_id, exc_info=True
                    )
                    continue
                if cursor is None:
                    self.cursors.pop(trace_id, None)
                else:
                    self.cursors[trace_id] = cursor

    def _maybe_add_routing_repairs(
        self,
        delta: dict[str, Any],
        trace: Trace,
        cursor: TraceCursor,
        sent_nodes: int,
    ) -> None:
        """Add final rows repaired since they were sent, keyed by node index.

        Record every node's current row on the cursor. `sent_nodes` is the pre-flush
        count, so a node first sent in this delta is never reported as a repair.
        """
        repairs: dict[int, dict] = {}
        for index, node in enumerate(trace.nodes):
            if node.routed_experts is None:
                continue
            row = _encode_ndarray(node.routed_experts[-1:])
            packed = pack(row)
            if cursor.final_rows.get(index) != packed:
                cursor.final_rows[index] = packed
                if index < sent_nodes:
                    repairs[index] = row
        if repairs:
            delta["routing_repairs"] = repairs

    def diff(self) -> list[tuple[str, dict, TraceCursor | None]]:
        """Each trace's delta against its sent cursor, with the cursor as it stands once
        that delta is sent (None for a discard). Nothing here is committed: `flush`
        stores a cursor only after its delta left."""
        traces = list(self.traces())
        live = {trace.id for trace in traces}
        deltas: list[tuple[str, dict, TraceCursor | None]] = []
        for trace_id in [trace_id for trace_id in self.cursors if trace_id not in live]:
            # A retried attempt abandons its traces; the client drops them too.
            deltas.append((trace_id, {"trace": trace_id, "discard": True}, None))
        for trace in traces:
            delta: dict[str, Any] = {"trace": trace.id}
            sent = self.cursors.get(trace.id)
            cursor = copy.deepcopy(sent) if sent is not None else TraceCursor()
            if sent is None:
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
            self._maybe_add_routing_repairs(delta, trace, cursor, cursor.sent["nodes"])
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
                deltas.append((trace.id, delta, cursor))
        return deltas


class EpisodeAssembly:
    """The client's growing picture of one served episode: raw trace dicts in arrival
    order, each shaped like a dumped `Trace` plus its `pending` preview."""

    def __init__(self) -> None:
        self.traces: dict[str, dict] = {}

    def _maybe_apply_routing_repairs(self, delta: dict[str, Any], trace: dict) -> None:
        """Replace repaired final rows in nodes the client already holds.

        Rebuild each array: a repair can widen its dtype, and decoded rows alias
        the original delta's bytes, which must remain unchanged for consumers.
        """
        for index, row in (delta.get("routing_repairs") or {}).items():
            node = trace["nodes"][int(index)]
            node["routed_experts"] = _encode_ndarray(
                np.concatenate(
                    [_decode_ndarray(node["routed_experts"])[:-1], _decode_ndarray(row)]
                )
            )

    def apply(self, delta: dict) -> None:
        trace_id = delta["trace"]
        if delta.get("discard"):
            self.traces.pop(trace_id, None)
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
        self._maybe_apply_routing_repairs(delta, trace)
        # a later `links` delta grows a node's semantic_parents in place, so the node
        # is copied: the delta stays as it was when the caller received it
        if "nodes" in delta:
            trace["nodes"].extend(
                {**node, "semantic_parents": list(node.get("semantic_parents") or [])}
                for node in delta["nodes"]
            )
        for field in LIST_FIELDS:
            if field in delta and field != "nodes":
                trace[field].extend(delta[field])
        trace.update(delta.get("set") or {})
        # a committed turn absorbs the preview; an explicit preview replaces it
        if "nodes" in delta:
            trace["pending"] = []
        if "pending" in delta:
            trace["pending"] = delta["pending"]

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
