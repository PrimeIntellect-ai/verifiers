"""Incremental trace deltas, for anything that shows or records a trace while it runs.

A trace announces its own changes (`Trace.notify`, at every phase change and recorded turn).
`DeltaStreamer` diffs the watched traces against what it has already sent and hands `send` only
the new part: the header once, then appended nodes / calls / errors, semantic links added to
earlier nodes, scalar fields whose value changed, and the `pending` preview (messages of the
request in flight that no node holds yet). Nodes go without their per-token training arrays.
`EpisodeAssembly` applies those deltas back into raw trace dicts. The transport is the caller's:
a socket, a JSONL file, a queue. Both sides live here because the field lists track `Trace`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Self

from pydantic import BaseModel

from verifiers.v1.trace import Trace
from verifiers.v1.utils.logging import clip, get_logger

logger = get_logger(__name__)

HEADER_FIELDS = ("version", "id", "verifiers", "task")
"""Sent once, when a trace first appears."""

LIST_FIELDS = (
    "nodes",
    "calls",
    "request_rewrites",
    "response_rewrites",
)
"""Append-only: each delta carries the items past the sent count. Lists the library replaces
(`errors` on a retry, `extra_usage` on a rescore) are scalars instead, re-sent whole."""

SCALAR_FIELDS = tuple(
    sorted(
        {name for name, info in Trace.model_fields.items() if not info.exclude}.union(
            Trace.model_computed_fields
        ).difference(HEADER_FIELDS, LIST_FIELDS)
    )
)
"""Every other serialized field, derived values included: re-sent whole whenever its value changes."""

TRAINING_ARRAY_FIELDS = frozenset({"routed_experts", "sampling_mask", "top_logprobs"})
"""Node fields holding per-token training arrays; never streamed."""


def dump(model: BaseModel, **kwargs: Any) -> dict:
    # Dump the runtime type, so subclassed task data and agent configs keep their own fields.
    return model.model_dump(mode="python", serialize_as_any=True, **kwargs)


class TraceCursor:
    """What of one trace has been sent."""

    def __init__(self, attempt: int) -> None:
        self.attempt = attempt
        """The attempt this cursor diffs: a retry keeps the trace id but not the attempt."""
        self.sent = dict.fromkeys(LIST_FIELDS, 0)
        self.links: list[int] = []
        """Per sent node, how many of its semantic links have been sent."""
        self.scalars: dict[str, Any] = {}
        self.pending: list[dict] = []

    def copy(self) -> TraceCursor:
        """A copy safe to advance; the stored values themselves are never mutated."""
        cursor = TraceCursor(self.attempt)
        cursor.sent, cursor.links, cursor.scalars = (
            dict(self.sent),
            list(self.links),
            dict(self.scalars),
        )
        cursor.pending = self.pending
        return cursor


class DeltaStreamer:
    """Sends the current traces' changes as deltas, one flush per burst of changes.

    `watch` subscribes a trace; each `Trace.notify` schedules a flush, and changes landing in the
    same loop iteration share it. Use as an async context manager around the run: a clean exit
    flushes the final state, a failed one cancels pending sends. A cursor advances only once its
    delta is sent, so a failed send is diffed again at the next flush."""

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
        self._closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._closed = True  # a trace notifying after the run sends nothing
        if self._scheduled is not None:
            self._scheduled.cancel()
            self._scheduled = None
        if exc_type is not None:
            for task in self._flushes:
                task.cancel()
        await asyncio.gather(*self._flushes, return_exceptions=True)
        if exc_type is None:
            await self.flush()

    def watch(self, trace: Trace) -> None:
        # A plain function, not the bound method: grading deep-copies traces mid-run, and a
        # bound method would drag this streamer (and its asyncio tasks) into the copy.
        trace.watch(lambda changed: self._changed(changed))
        self._changed(trace)

    def _changed(self, trace: Trace) -> None:
        if self._scheduled is None and not self._closed:
            self._scheduled = asyncio.get_running_loop().call_soon(self._start_flush)

    def _start_flush(self) -> None:
        self._scheduled = None
        task = asyncio.create_task(self.flush())
        self._flushes.add(task)
        task.add_done_callback(self._flushes.discard)

    async def flush(self) -> None:
        async with self._lock:  # deltas must leave in diff order
            for trace_id, delta, cursor in self.diff():
                # The cursor stays put on failure: the next flush resends it.
                try:
                    await self.send(delta)
                except Exception as e:
                    logger.warning(
                        "Failed to send trace delta - %s: %s",
                        type(e).__name__,
                        clip(str(e)),
                        exc_info=logger.isEnabledFor(logging.DEBUG),
                    )
                    continue
                if cursor is None:
                    self.cursors.pop(trace_id, None)
                else:
                    self.cursors[trace_id] = cursor

    def diff(self) -> list[tuple[str, dict, TraceCursor | None]]:
        """Each trace's delta, with its cursor as it stands once that delta is sent (None for a
        discard). Nothing is committed here: `flush` stores a cursor only after its send."""
        traces = list(self.traces())
        live = {trace.id for trace in traces}
        # A trace that left the live set (a retried attempt) is discarded on the other side too.
        deltas = [
            (gone, {"trace": gone, "discard": True}, None)
            for gone in self.cursors
            if gone not in live
        ]
        for trace in traces:
            delta: dict[str, Any] = {"trace": trace.id}
            sent = self.cursors.get(trace.id)
            if sent is not None and sent.attempt != trace.attempt:
                sent = None  # a retry's fresh trace under the same id: reopen it from scratch
            cursor = sent.copy() if sent is not None else TraceCursor(trace.attempt)
            if sent is None:
                delta["open"] = dump(trace, include=set(HEADER_FIELDS))
            links = {}  # str keys: binary wire formats forbid int map keys
            for index in range(cursor.sent["nodes"]):
                node_links = trace.nodes[index].semantic_parents
                if len(node_links) > cursor.links[index]:
                    links[str(index)] = [
                        dump(link) for link in node_links[cursor.links[index] :]
                    ]
                    cursor.links[index] = len(node_links)
            if links:
                delta["links"] = links
            for field in LIST_FIELDS:
                items = getattr(trace, field)[cursor.sent[field] :]
                if not items:
                    continue
                if field == "nodes":
                    # Per-token streams ride as raw-bytes arrays; training arrays stay behind.
                    delta[field] = [
                        dump(
                            node,
                            context={"token_arrays": True},
                            exclude=TRAINING_ARRAY_FIELDS,
                        )
                        for node in items
                    ]
                    cursor.links.extend(len(node.semantic_parents) for node in items)
                else:
                    delta[field] = [dump(item) for item in items]
                cursor.sent[field] += len(items)
            changed = {}
            for field, value in dump(trace, include=set(SCALAR_FIELDS)).items():
                try:
                    same = cursor.scalars.get(field, ...) == value
                except Exception:  # noqa: BLE001 - arrays and tensors in `info` compare elementwise
                    same = False
                if not same:
                    changed[field] = value
            if changed:
                cursor.scalars.update(changed)
                delta["set"] = changed
            pending = [dump(message) for message in trace.pending]
            if pending != cursor.pending:
                cursor.pending = delta["pending"] = pending
            if len(delta) > 1:
                deltas.append((trace.id, delta, cursor))
        return deltas


class EpisodeAssembly:
    """The receiving side: raw trace dicts in arrival order, each shaped like a dumped `Trace`
    plus its `pending` preview. Validate one with `Trace.model_validate` once it is complete."""

    def __init__(self) -> None:
        self.traces: dict[str, dict] = {}

    def apply(self, delta: dict) -> None:
        trace_id = delta["trace"]
        if delta.get("discard"):
            self.traces.pop(trace_id, None)
            return
        trace = self.traces.get(trace_id)
        # An `open` for a known id is a retry: start over.
        if trace is None or "open" in delta:
            trace = self.traces[trace_id] = {
                **delta["open"],
                **{f: [] for f in LIST_FIELDS},
                "pending": [],
            }
        for index, links in (delta.get("links") or {}).items():
            trace["nodes"][int(index)]["semantic_parents"].extend(links)
        for field in LIST_FIELDS:
            if field in delta:
                # Nodes are copied, so a later `links` delta never mutates the delta the caller holds.
                items = delta[field]
                if field == "nodes":
                    items = [
                        {
                            **node,
                            "semantic_parents": list(
                                node.get("semantic_parents") or []
                            ),
                        }
                        for node in items
                    ]
                trace[field].extend(items)
        trace.update(delta.get("set") or {})
        if "pending" in delta:
            trace["pending"] = delta["pending"]


__all__ = ["DeltaStreamer", "EpisodeAssembly"]
