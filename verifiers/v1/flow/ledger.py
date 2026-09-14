"""The ledger: one JSON record per node instance, written at completion.

Resume reads the ledger and skips every instance whose key still matches.
Nothing is replayed. Traces go to the run's `traces.jsonl` in verifiers' format.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from verifiers.v1.cli.output import (
    TRACES_FILE,
    append_trace,
    read_episodes,
    type_adapter,
)
from verifiers.v1.episode import WireEpisode
from verifiers.v1.trace import Trace, WireTrace

logger = logging.getLogger("verifiers.flow")


class NodeRecord(BaseModel):
    key: str
    row: str
    node: str
    visit: int
    index: int | None = None
    kind: str
    terminal: str
    """`completed` | `error` | `exhausted`."""
    outcome: str | None = None
    summary: str = ""
    payload: Any = None
    trace_id: str | None = None
    trace_ids: list[str] = Field(default_factory=list)
    reward: float | None = None
    attempt: int = 1
    error: str | None = None
    started_at: str
    finished_at: str


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical(value: Any) -> Any:
    """Sets become sorted lists so digests are stable across processes (set
    iteration order is seeded); unsortable sets degrade to a sorted string form.
    Models flatten through `model_dump(mode="python", by_alias=True)` first: the
    dump is byte-identical to the base serialization for set-free models (so
    existing ledger keys never change) while `mode="python"` keeps set-valued
    fields AS SETS, letting the sorting above canonicalize them before
    serialization would convert them to seed-ordered lists."""
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
    """Sets are canonicalized to sorted lists BOTH before flattening (plain
    containers) and after (models serialize their inner sets), so digests are
    stable across processes regardless of hash seed."""
    flat = to_jsonable_python(_canonical(parts))
    return hashlib.sha256(
        json.dumps(_canonical(flat), sort_keys=True).encode()
    ).hexdigest()


def row_key(row: Any) -> str:
    key = getattr(row, "key", None)
    if isinstance(key, str) and key:
        return key
    return digest(row)[:16]


class Ledger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.nodes_dir = run_dir / "nodes"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / TRACES_FILE).touch()
        self.lock = asyncio.Lock()
        self._traces: dict[str, WireTrace] | None = None

    def path(self, row: str, node: str, visit: int, index: int | None) -> Path:
        name = f"{node}@{visit}" + (f".{index}" if index is not None else "") + ".json"
        return self.nodes_dir / row / name

    def item_records(self, row: str, node: str, visit: int) -> dict[int, NodeRecord]:
        """A fan-out node's per-item records for one visit, by original index."""
        out: dict[int, NodeRecord] = {}
        row_dir = self.nodes_dir / row
        if not row_dir.is_dir():
            return out
        for path in row_dir.glob(f"{node}@{visit}.*.json"):
            record = NodeRecord.model_validate_json(path.read_text())
            if record.index is not None:
                out[record.index] = record
        return out

    def get(
        self, row: str, node: str, visit: int, index: int | None = None
    ) -> NodeRecord | None:
        path = self.path(row, node, visit, index)
        if not path.exists():
            return None
        return NodeRecord.model_validate_json(path.read_text())

    def put(self, record: NodeRecord) -> None:
        path = self.path(record.row, record.node, record.visit, record.index)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(record.model_dump_json(indent=1))
        os.replace(tmp, path)

    async def append(self, trace: Trace, env: str) -> None:
        await append_trace(self.run_dir, trace, self.lock, env=env)
        if self._traces is not None:
            self._traces[trace.id] = trace  # type: ignore[assignment]

    def trace(self, trace_id: str) -> WireTrace | None:
        if self._traces is None:
            self._traces = {
                trace.id: trace
                for episode in self._episodes()
                for trace in episode.traces
            }
        return self._traces.get(trace_id)

    def _episodes(self) -> list[WireEpisode]:
        """The run's episodes; a torn FINAL line (a process killed mid-write) is
        skipped so one bad tail cannot block every resume attach."""
        try:
            return read_episodes(self.run_dir, WireTrace)
        except json.JSONDecodeError:
            lines = (self.run_dir / TRACES_FILE).read_text().splitlines()
            tail_torn = False
            episodes: list[WireEpisode] = []
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if i == len(lines) - 1:
                        tail_torn = True  # the process died mid-write
                        continue
                    raise
                record = WireEpisode.model_validate({**row, "traces": []})
                record.traces = [
                    type_adapter(WireTrace).validate_python(t)
                    for t in row["traces"]
                ]
                episodes.append(record)
            if not tail_torn:
                raise
            logger.warning("traces.jsonl ends in a torn line; skipping it")
            return episodes
