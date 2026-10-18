"""Read and write standard saved traces."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import uuid
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path

from pydantic import TypeAdapter
from pydantic_core import to_json

from verifiers.v1.episode import EnvInfo, Episode, WireEpisode
from verifiers.v1.state import StateT
from verifiers.v1.task import DataT
from verifiers.v1.trace import AgentConfigT, Trace, WireTrace
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.artifacts import save_artifacts

TRACES_FILE = "traces.jsonl"
type_adapter = cache(TypeAdapter)


_SCHEMA = """
                CREATE TABLE IF NOT EXISTS meta(dev INTEGER, ino INTEGER, end INTEGER);
                CREATE TABLE IF NOT EXISTS episodes(
                    id TEXT PRIMARY KEY, offset INTEGER, length INTEGER, ordinal INTEGER);
                CREATE TABLE IF NOT EXISTS traces(
                    id TEXT PRIMARY KEY, episode TEXT, tokens INTEGER);
                CREATE INDEX IF NOT EXISTS trace_episode ON traces(episode);
            """


def _trace_record(trace: Trace) -> dict:
    # Validation can retain unknown fields on a wire trace whose base serializer
    # ignores extras. Keep those fields; typed fields and computed counts win.
    return {**(trace.model_extra or {}), **trace.to_record()}


def _episode_record(episode: Episode) -> dict:
    # Dump nested traces through their concrete types, including user subclass fields.
    row = episode.to_record()
    row["traces"] = [_trace_record(trace) for trace in episode.traces]
    return row


def _save_artifacts(results_dir: Path, trace: Trace) -> None:
    if not trace.state.artifacts:
        return
    save_artifacts(
        artifact_path(results_dir, trace.id),
        trace.state.artifacts,
        trace.task.data.artifacts,
        workdir=getattr(trace.agent.runtime, "workdir", None) or "/",
    )


def artifact_path(results_dir: Path, trace_id: str) -> Path:
    """The native artifact directory, shared by episode and trace writers."""
    if trace_id in {"", ".", ".."} or Path(trace_id).parts != (trace_id,):
        raise ValueError(f"trace ID must be a single path component: {trace_id!r}")
    return results_dir / "artifacts" / trace_id


@contextlib.contextmanager
def trace_lock(results_dir: Path) -> Iterator[None]:
    """Serialize writers across processes, including atomic stream replacements."""
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "traces.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _rows(file, offset: int = 0):
    file.seek(offset)
    while line := file.readline():
        start, end = offset, file.tell()
        if not line.endswith(b"\n"):
            break
        offset = end
        try:
            row = json.loads(line)
            if not isinstance(row.get("id"), str) or not isinstance(
                row.get("traces"), list
            ):
                continue
        except (ValueError, AttributeError):
            continue
        yield start, end - start, row, line


def episode_lines(results_dir: Path) -> list[bytes]:
    """Latest complete revision of each episode, in first-seen order.

    Reading never changes the stream or requires a writable results directory.
    Distinct episodes remain distinct even when their traces share an ID.
    """
    latest: dict[str, bytes] = {}
    try:
        with (results_dir / TRACES_FILE).open("rb") as file:
            for _, _, row, line in _rows(file):
                latest[row["id"]] = line
    except FileNotFoundError:
        pass
    return list(latest.values())


def replace_episodes(results_dir: Path, lines: Iterable[bytes]) -> None:
    """Atomically replace a stream under its writer lock; indexes rebuild lazily."""
    with trace_lock(results_dir):
        _replace(results_dir / TRACES_FILE, lines)


def _replace(path: Path, lines: Iterable[bytes]) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".traces-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            file.writelines(lines)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class _ReadIndex:
    """One thread's reusable cache for an unindexed, read-only stream."""

    def __init__(self, generation: tuple[int, ...]) -> None:
        self.generation = generation
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.executescript(_SCHEMA)

    def __del__(self) -> None:
        self.db.close()


class TraceStore:
    """Native episode JSONL with revision-aware random access and artifact persistence.

    ``write`` updates a trace inside its existing episode, preserving its envelope
    and sibling traces. ``write_episode`` appends an explicit rollout identity.
    ``traces.index.sqlite`` is a disposable offset cache: removing it loses no data.
    Live readers and resumes use the same latest-episode semantics.
    """

    def __init__(self, root: Path, *, env: str = "") -> None:
        self.root, self.env = Path(root), env
        self.file = self.root / TRACES_FILE
        self._reader = threading.local()

    @contextlib.contextmanager
    def _indexed(self):
        with (
            trace_lock(self.root),
            contextlib.closing(
                sqlite3.connect(self.root / "traces.index.sqlite", timeout=60)
            ) as db,
        ):
            db.executescript(_SCHEMA)
            with self.file.open("a+b") as file:
                stat = os.fstat(file.fileno())
                saved = db.execute("SELECT dev, ino, end FROM meta").fetchone()
                if (
                    saved is None
                    or saved[:2] != (stat.st_dev, stat.st_ino)
                    or saved[2] > stat.st_size
                ):
                    db.executescript(
                        "DELETE FROM meta; DELETE FROM episodes; DELETE FROM traces;"
                    )
                    saved = (stat.st_dev, stat.st_ino, 0)
                offset = saved[2]
                file.seek(offset)
                # Only parse the unindexed tail; recover a kill between JSONL and index commits.
                for start, length, row, _ in _rows(file, offset):
                    self._index_row(db, row, start, length)
                    offset = start + length
                # Invalid complete rows are skipped too. A torn final append is truncated
                # before the next writer can fuse another episode onto it.
                file.seek(offset)
                for line in file:
                    if not line.endswith(b"\n"):
                        break
                    offset += len(line)
                file.truncate(offset)
                db.execute("DELETE FROM meta")
                db.execute("INSERT INTO meta VALUES (?, ?, ?)", (*saved[:2], offset))
                db.commit()
                yield db, file
                db.commit()

    @contextlib.contextmanager
    def _read_index(self):
        # Writers publish offsets; readers never repair or truncate a shared run.
        # An older or interrupted run may have no current cache: derive a temporary
        # index from its complete canonical rows without needing filesystem writes.
        with self.file.open("rb") as file, contextlib.ExitStack() as stack:
            stat = os.fstat(file.fileno())
            index = self.root / "traces.index.sqlite"
            if index.exists():
                try:
                    db = stack.enter_context(
                        contextlib.closing(
                            sqlite3.connect(
                                index.resolve().as_uri() + "?mode=ro", uri=True
                            )
                        )
                    )
                    db.execute("BEGIN")
                    saved = db.execute("SELECT dev, ino, end FROM meta").fetchone()
                except sqlite3.DatabaseError:
                    # A cache may be copied incompletely or left before its schema
                    # was initialized. Canonical rows remain readable unchanged.
                    saved = None
                if saved == (stat.st_dev, stat.st_ino, stat.st_size):
                    yield db, file
                    return
            # A stale SQLite snapshot must not block an index commit while we
            # inspect a potentially large canonical stream.
            stack.close()
            generation = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            cached = getattr(self._reader, "index", None)
            if cached is None or cached.generation != generation:
                cached = _ReadIndex(generation)
                for start, length, row, _ in _rows(file):
                    self._index_row(cached.db, row, start, length)
                self._reader.index = cached
            yield cached.db, file

    @staticmethod
    def _index_row(db, row: dict, offset: int, length: int) -> None:
        episode = row["id"]
        db.execute("DELETE FROM traces WHERE episode = ?", (episode,))
        db.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET offset=excluded.offset, length=excluded.length",
            (episode, offset, length, offset),
        )
        for trace in row["traces"]:
            db.execute(
                "INSERT OR REPLACE INTO traces VALUES (?, ?, ?)",
                (trace["id"], episode, trace.get("num_total_tokens", 0)),
            )

    @staticmethod
    def _append(db, file, row: dict) -> None:
        data = to_json(row, inf_nan_mode="null") + b"\n"
        file.seek(0, os.SEEK_END)
        offset = file.tell()
        file.write(data)
        file.flush()
        TraceStore._index_row(db, row, offset, len(data))
        db.execute("UPDATE meta SET end = ?", (offset + len(data),))

    @staticmethod
    def _row(db, file, trace_id: str) -> dict | None:
        where = db.execute(
            "SELECT offset, length FROM episodes JOIN traces ON episodes.id=traces.episode WHERE traces.id=?",
            (trace_id,),
        ).fetchone()
        if where is None:
            return None
        file.seek(where[0])
        return json.loads(file.read(where[1]))

    def write(self, trace: Trace) -> None:
        """Save or annotate one trace. Existing multi-agent episode metadata survives."""
        record = _trace_record(trace)
        episode_id = uuid.uuid5(uuid.NAMESPACE_URL, f"verifiers:trace:{trace.id}").hex
        with self._indexed() as (db, file):
            _save_artifacts(self.root, trace)
            row = self._row(db, file, trace.id)
            if row is None:
                row = _episode_record(
                    Episode(
                        id=episode_id,
                        env=EnvInfo(id=self.env),
                        task=trace.task,
                        traces=[trace],
                        ok=trace.ok,
                    )
                )
            else:
                row["traces"] = [
                    record if t["id"] == trace.id else t for t in row["traces"]
                ]
                for key in (
                    "num_input_tokens",
                    "num_output_tokens",
                    "num_total_tokens",
                ):
                    row[key] = sum(t.get(key, 0) for t in row["traces"])
                # The environment owns arbitrary multi-agent episode verdicts.
                if row["id"] == episode_id and len(row["traces"]) == 1:
                    row["ok"] = trace.ok
            self._append(db, file, row)

    def write_episode(self, episode: Episode) -> None:
        row = _episode_record(episode)
        with self._indexed() as (db, file):
            for trace in episode.traces:
                _save_artifacts(self.root, trace)
            self._append(db, file, row)

    def get(self, trace_id: str, trace_type: type = WireTrace) -> Trace | None:
        if not self.file.exists():
            return None
        with self._read_index() as (db, file):
            row = self._row(db, file, trace_id)
            if row is None:
                return None
            return next(
                (
                    trace_type.model_validate(t, extra="allow")
                    for t in row["traces"]
                    if t["id"] == trace_id
                ),
                None,
            )

    def has(self, trace_id: str) -> bool:
        if not self.file.exists():
            return False
        with self._read_index() as (db, _):
            return (
                db.execute("SELECT 1 FROM traces WHERE id=?", (trace_id,)).fetchone()
                is not None
            )

    def index(self) -> dict[str, tuple[int, int]]:
        if not self.file.exists():
            return {}
        with self._read_index() as (db, _):
            return {
                row[0]: (row[1], row[2])
                for row in db.execute(
                    "SELECT traces.id, offset, length FROM traces JOIN episodes ON episodes.id=traces.episode"
                )
            }

    @property
    def tokens(self) -> dict[str, int]:
        if not self.file.exists():
            return {}
        with self._read_index() as (db, _):
            return dict(db.execute("SELECT id, tokens FROM traces"))

    async def append(self, trace: Trace) -> None:
        await run_shielded(asyncio.to_thread(self.write, trace))

    def compact(self, *, remove: Iterable[str] = ()) -> int:
        """Drop superseded revisions and explicitly removed traces atomically.

        Removal destroys payloads and their native usage; callers retaining billing
        history must keep that history separately. Other traces in an episode survive.
        """
        ids = set(remove)
        deleted: set[str] = set()
        if not self.file.exists():
            return 0
        with self._indexed() as (db, file):

            def lines():
                for offset, length in db.execute(
                    "SELECT offset, length FROM episodes ORDER BY ordinal"
                ):
                    file.seek(offset)
                    line = file.read(length)
                    row = json.loads(line)
                    removed = {t["id"] for t in row["traces"]} & ids
                    if not removed:
                        yield line
                        continue
                    deleted.update(removed)
                    row["traces"] = [t for t in row["traces"] if t["id"] not in ids]
                    if row["traces"]:
                        for key in (
                            "num_input_tokens",
                            "num_output_tokens",
                            "num_total_tokens",
                        ):
                            row[key] = sum(t.get(key, 0) for t in row["traces"])
                        yield to_json(row, inf_nan_mode="null") + b"\n"

            _replace(self.file, lines())
            for trace_id in deleted:
                shutil.rmtree(artifact_path(self.root, trace_id), ignore_errors=True)
        return len(deleted)


def write_episode(
    results_dir: Path, episode: Episode[DataT, StateT, AgentConfigT]
) -> None:
    """Serialize one explicitly identified rollout through native persistence."""
    TraceStore(results_dir).write_episode(episode)


def read_episodes(results_dir: Path, trace_type: type) -> list[WireEpisode]:
    """Load a run's saved rollouts from `traces.jsonl` with traces typed as
    `trace_type` (`Trace[WireTaskData, ...]` reads any taskset's file without
    importing it)."""
    trace_adapter = type_adapter(trace_type)
    episodes: list[WireEpisode] = []
    for line in episode_lines(results_dir):
        row = json.loads(line)
        record = WireEpisode.model_validate({**row, "traces": []})
        record.traces = [
            trace_adapter.validate_python(trace, extra="allow")
            for trace in row["traces"]
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
