"""Read and write standard saved traces."""

from __future__ import annotations

import asyncio
from functools import cache
from pathlib import Path

from pydantic import TypeAdapter

from verifiers.v1.episode import EnvInfo, Episode
from verifiers.v1.state import StateT
from verifiers.v1.task import DataT
from verifiers.v1.trace import EXCLUDE_FIELDS, AgentConfigT, Trace
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.artifacts import save_artifacts

TRACES_FILE = "traces.jsonl"
type_adapter = cache(TypeAdapter)


def write_episode(
    results_dir: Path, episode: Episode[DataT, StateT, AgentConfigT]
) -> None:
    """Serialize and append one rollout episode in the worker thread."""
    # Preserve fields declared by typed Trace subclasses nested in the episode. Raw tensors
    # stay on the msgpack wire only, as in `Episode.to_record`: they don't survive JSON.
    data = type_adapter(type(episode)).dump_json(
        episode,
        exclude={"traces": {"__all__": EXCLUDE_FIELDS}},
        exclude_none=True,
    )
    for trace in episode.traces:
        if trace.state.artifacts:
            if trace.id == ".." or Path(trace.id).parts != (trace.id,):
                raise ValueError(
                    f"trace ID must be a single path component: {trace.id!r}"
                )
            save_artifacts(
                results_dir / "artifacts" / trace.id,
                trace.state.artifacts,
                trace.task.data.artifacts,
                workdir=getattr(trace.agent.runtime, "workdir", None) or "/",
            )
    with (results_dir / TRACES_FILE).open("ab") as f:
        f.write(data + b"\n")


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
    """Append one finished trace as a single-agent rollout episode, for callers that
    complete trace-at-a-time (debug)."""
    episode = Episode(
        env=EnvInfo(id=env),
        task=trace.task,
        traces=[trace],
        ok=trace.ok,
    )
    await append_episode(results_dir, episode, lock)
