"""Incremental lookup of standard saved traces, shared by runners and viewers."""

from __future__ import annotations

import asyncio
import json
from functools import cache
from pathlib import Path

from pydantic import TypeAdapter

from verifiers.v1.episode import EnvInfo, Episode, WireEpisode
from verifiers.v1.state import StateT
from verifiers.v1.task import DataT
from verifiers.v1.trace import AgentConfigT, Trace, WireTrace
from verifiers.v1.utils.aio import run_shielded

TRACES_FILE = "traces.jsonl"
type_adapter = cache(TypeAdapter)


def write_episode(
    results_dir: Path, episode: Episode[DataT, StateT, AgentConfigT]
) -> None:
    """Serialize and append one rollout episode in the worker thread."""
    # Preserve fields declared by typed Trace subclasses nested in the episode.
    data = type_adapter(type(episode)).dump_json(episode, exclude_none=True)
    with (results_dir / TRACES_FILE).open("ab") as f:
        f.write(data + b"\n")


def read_episodes(results_dir: Path, trace_type: type) -> list[WireEpisode]:
    """Load a run's saved rollouts from `traces.jsonl` with traces typed as
    `trace_type` (`Trace[WireTaskData, ...]` reads any taskset's file without
    importing it)."""
    trace_adapter = type_adapter(trace_type)
    episodes: list[WireEpisode] = []
    with (results_dir / TRACES_FILE).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            record = WireEpisode.model_validate({**row, "traces": []})
            record.traces = [
                trace_adapter.validate_python(trace) for trace in row["traces"]
            ]
            episodes.append(record)
    return episodes


async def append_episode(
    results_dir: Path,
    episode: Episode[DataT, StateT, AgentConfigT],
    lock: asyncio.Lock,
) -> None:
    """Append one finished rollout episode without blocking the event loop. The run's
    shared lock preserves whole-line ordering, and awaiting the worker preserves
    per-episode durability."""

    async def persist() -> None:
        async with lock:
            await asyncio.to_thread(write_episode, results_dir, episode)

    # Run lock acquisition and the worker to completion even under cancellation, so
    # finalized episodes are never lost mid-write (`run_shielded` re-raises the cancellation).
    await run_shielded(persist())


async def append_trace(
    results_dir: Path, trace: Trace, lock: asyncio.Lock, env: str = ""
) -> None:
    """Append one finished trace as a single-agent rollout episode — debug and replay,
    which complete trace-at-a-time, both go through here."""
    episode = Episode(
        env=EnvInfo(id=env),
        task=trace.task,
        traces=[trace],
        ok=trace.ok,
    )
    await append_episode(results_dir, episode, lock)


def trim_torn_tail(file: Path) -> None:
    """Drop the partial last line a kill mid-append left, so the next append does not fuse
    with it into a line no reader can skip. Creates the file."""
    file.touch()
    data = file.read_bytes()
    if data and not data.endswith(b"\n"):
        file.write_bytes(data[: data.rfind(b"\n") + 1])


class TraceStore:
    def __init__(self, root: Path, *, env: str = "") -> None:
        self.root, self.env = root, env
        self.file = root / TRACES_FILE
        self.lock = asyncio.Lock()
        self._index: dict[str, tuple[int, int]] = {}  # id -> (offset, length)
        self._offset = 0
        self.tokens: dict[str, int] = {}

    async def append(self, trace: Trace) -> None:
        await append_trace(self.root, trace, self.lock, env=self.env)

    def index(self) -> dict[str, tuple[int, int]]:
        """Index new complete records, retaining native token totals alongside their offsets."""
        if self.file.exists():
            with self.file.open("rb") as file:
                offset = self._offset
                file.seek(offset)
                for line in file:
                    if not line.endswith(b"\n"):
                        break
                    for t in json.loads(line)["traces"]:
                        self._index[t["id"]] = (offset, len(line))
                        self.tokens[t["id"]] = (
                            type_adapter(WireTrace).validate_python(t).num_total_tokens
                        )
                    offset += len(line)
                self._offset = offset
        return self._index

    def get(self, trace_id: str) -> WireTrace | None:
        if (where := self.index().get(trace_id)) is None:
            return None
        with self.file.open("rb") as file:
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
