"""The ledger: one JSON record per step, written when the step completes, and one
progress event per line as the steps happen.

A resumed run attaches to every record whose key still matches and runs nothing
else. Traces go to the run's `traces.jsonl` in verifiers' format.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from verifiers.v1.cli.output import TRACES_FILE, append_trace, read_jsonl, type_adapter
from verifiers.v1.flow.work import WorkKind
from verifiers.v1.trace import Trace, WireTrace

logger = logging.getLogger("verifiers.flow")

EVENTS_FILE = "events.jsonl"
"""Filename the run's progress events are appended to (one JSON object per line)."""
INDEX_FILE = "traces.index.jsonl"
"""Where each trace sits in `traces.jsonl`: `{id, offset, length}` per line, so a step
attaching on resume reads its one trace instead of the whole file."""

SHORT = 16
"""Digest chars for a row key, a source hash and the run label."""
STEP_KEY = 24
"""Digest chars for a step key — what a resume matches a record on."""

Terminal = Literal["completed", "error"]
EventKind = Literal[
    "run",
    "drain",
    "row_started",
    "row_finished",
    "step_started",
    "step_attached",
    "step_retrying",
    "step_completed",
    "step_failed",
    "step_cancelled",
    "spread_started",
    "spread_finished",
    "rss",
]


class StepRecord(BaseModel):
    key: str
    row: str
    path: str
    """Scope path and name with its occurrence, e.g. `visit#1/build#0`."""
    index: int | None = None
    """The item index within a spread."""
    kind: WorkKind
    terminal: Terminal
    payload: Any = None
    trace_id: str | None = None
    error: str | None = None
    attempts: int = 1
    started_at: str
    finished_at: str


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical(value: Any) -> Any:
    """Sets as sorted lists, models as dicts: set order is hash-seeded, so a digest
    over raw sets would differ between processes."""
    if isinstance(value, (set, frozenset)):
        try:
            return sorted(_canonical(v) for v in value)
        except TypeError:
            return sorted(repr(v) for v in value)
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="python", by_alias=True))
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def digest(*parts: Any) -> str:
    flat = to_jsonable_python(_canonical(parts), fallback=repr)
    return hashlib.sha256(json.dumps(flat, sort_keys=True).encode()).hexdigest()


def trim_torn_tail(file: Path) -> None:
    """Drop the partial last line a kill mid-append left in a JSON-lines file, so the next
    append does not fuse with it into a corrupt line no reader can skip. Creates the file."""
    file.touch()
    data = file.read_bytes()
    if data and not data.endswith(b"\n"):
        file.write_bytes(data[: data.rfind(b"\n") + 1])


def row_key(row: Any) -> str:
    """A row's key: its `key` field or attribute when it has one, else a digest."""
    key = row.get("key") if isinstance(row, Mapping) else getattr(row, "key", None)
    return key if isinstance(key, str) and key else digest(row)[:SHORT]


class Ledger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.steps_dir = run_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.traces_file = run_dir / TRACES_FILE
        self.traces_file.touch()
        self.index_file = run_dir / INDEX_FILE
        self.events_file = run_dir / EVENTS_FILE
        for file in (self.events_file, self.index_file):
            trim_torn_tail(file)
        self.lock = asyncio.Lock()
        self._index: dict[str, tuple[int, int]] | None = (
            None  # trace id -> (offset, length)
        )

    def path(self, row: str, path: str, index: int | None) -> Path:
        name = (
            path.replace("/", "__")
            + (f".{index}" if index is not None else "")
            + ".json"
        )
        return self.steps_dir / row / name

    def get(self, row: str, path: str, index: int | None = None) -> StepRecord | None:
        file = self.path(row, path, index)
        if not file.exists():
            return None
        return StepRecord.model_validate_json(file.read_text())

    def put(self, record: StepRecord) -> None:
        file = self.path(record.row, record.path, record.index)
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp")
        tmp.write_text(record.model_dump_json(indent=1))
        os.replace(tmp, file)

    def event(self, kind: EventKind, **fields: Any) -> None:
        """One progress event as a JSON line: a single small append, so lines never
        interleave on one loop and a kill can tear at most the last one. Mirrored to
        the logger at INFO, as the same line."""
        event = {"type": kind, "at": now(), **fields}
        line = json.dumps(event)
        with self.events_file.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
        logger.info("%s", line)

    def events(self) -> list[dict[str, Any]]:
        """The run's events, oldest first."""
        return read_jsonl(self.events_file)

    # -- traces -----------------------------------------------------------------------

    async def append(self, trace: Trace) -> None:
        async with self.lock:
            start = self.traces_file.stat().st_size
            await append_trace(self.run_dir, trace, asyncio.Lock(), env="flow")
            length = self.traces_file.stat().st_size - start
            with self.index_file.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps({"id": trace.id, "offset": start, "length": length})
                    + "\n"
                )
            self.index()[trace.id] = (start, length)

    def index(self) -> dict[str, tuple[int, int]]:
        if self._index is None:
            self._index = {
                e["id"]: (e["offset"], e["length"])
                for e in (
                    read_jsonl(self.index_file) if self.index_file.exists() else []
                )
            }
        return self._index

    def trace(self, trace_id: str) -> WireTrace | None:
        """One trace by id, read in place; a trace the index lacks (a run written before
        the index, a torn index line) triggers one full rebuild of the index."""
        if trace_id not in self.index():
            self._reindex()
        if (where := self.index().get(trace_id)) is None:
            return None
        with self.traces_file.open("rb") as file:
            file.seek(where[0])
            episode = json.loads(file.read(where[1]))
        adapter = type_adapter(WireTrace)
        return next(
            (
                adapter.validate_python(t)
                for t in episode["traces"]
                if t["id"] == trace_id
            ),
            None,
        )

    def _reindex(self) -> None:
        index, offset = {}, 0
        with self.traces_file.open("rb") as file:
            for line in file:
                if line.strip():
                    try:
                        episode = json.loads(line)
                    except json.JSONDecodeError:  # the torn last line
                        break
                    for t in episode["traces"]:
                        index[t["id"]] = (offset, len(line))
                offset += len(line)
        with self.index_file.open("w", encoding="utf-8") as file:
            for tid, (start, length) in index.items():
                file.write(
                    json.dumps({"id": tid, "offset": start, "length": length}) + "\n"
                )
        self._index = index
