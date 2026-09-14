"""The ledger: one JSON record per node instance, written at completion.

Resume reads the ledger and skips every instance whose key still matches.
Nothing is replayed. Traces go to the run's `traces.jsonl` in verifiers' format.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from verifiers.v1.cli.output import TRACES_FILE, append_trace, read_episodes
from verifiers.v1.trace import Trace, WireTrace


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


def digest(*parts: Any) -> str:
    return hashlib.sha256(
        json.dumps(to_jsonable_python(parts), sort_keys=True).encode()
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
                for episode in read_episodes(self.run_dir, WireTrace)
                for trace in episode.traces
            }
        return self._traces.get(trace_id)
