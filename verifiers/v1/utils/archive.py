"""Durable host-side copies of sandbox artifacts, independent of grading transport."""

from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from verifiers.v1.configs.archive import ArchiveConfig
from verifiers.v1.utils.artifacts import Artifact, collect

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

logger = logging.getLogger(__name__)


def _tar_filename(source: str) -> str:
    return f"{source.strip('/').replace('/', '__')}.tar"


def _resolved_source(workdir: PurePosixPath, source: str) -> str:
    return str(workdir / source)


def _entries(
    runtime: Runtime,
    artifacts: list[Artifact] | None,
    extra: list[str],
) -> list[Artifact]:
    """Task path list plus eval extras, deduped after workdir resolve."""
    workdir = PurePosixPath(getattr(runtime.config, "workdir", "") or "/")
    entries: list[Artifact] = []
    seen: set[str] = set()
    for artifact in artifacts or []:
        key = _resolved_source(workdir, artifact.source)
        if key in seen:
            continue
        seen.add(key)
        entries.append(artifact)
    for source in extra:
        key = _resolved_source(workdir, source)
        if key in seen:
            continue
        seen.add(key)
        entries.append(Artifact(source=source))
    return entries


async def archive(
    runtime: Runtime,
    dest: Path,
    artifacts: list[Artifact] | None = None,
    config: ArchiveConfig | None = None,
) -> None:
    """Copy declared (and convention) artifact roots from `runtime` onto `dest`.

    Default inventory is `/logs/artifacts` plus `artifacts` (the task path list).
    `config.extra` merges additional sources; `config.exclude` is applied to every
    root, including the convention dir. Best-effort: missing sources are recorded,
    not raised. Tar bytes are written as files; they are not stored on the trace.
    """
    dest.mkdir(parents=True, exist_ok=True)
    policy = config or ArchiveConfig()
    optional = _entries(runtime, artifacts, policy.extra)
    collected = await collect(
        runtime,
        optional,
        exclude=policy.exclude,
        max_bytes=policy.max_mb * 1024 * 1024,
    )
    entries: list[dict] = []
    for source, blob in collected.items():
        name = _tar_filename(source)
        present = blob is not None
        if present:
            (dest / name).write_bytes(blob)
        entries.append(
            {"source": source, "file": name, "present": present, "error": None}
        )
    (dest / "manifest.json").write_text(
        json.dumps({"entries": entries}, indent=2) + "\n"
    )
    logger.debug("archived %d artifact root(s) to %s", len(entries), dest)
