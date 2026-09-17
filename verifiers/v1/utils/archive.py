"""Durable host-side copies of sandbox artifacts, independent of grading transport."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from verifiers.v1.configs.archive import ArchiveConfig
from verifiers.v1.utils.artifacts import Artifact, collect

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

logger = logging.getLogger(__name__)


ManifestStatus = Literal["ok", "skipped"]


@dataclass
class ManifestEntry:
    source: str
    file: str | None
    harbor_destination: str | None
    status: ManifestStatus


def drop_archive(root: Path | None, trace_id: str) -> None:
    """Remove a rollout dump that will not join the episode (an abandoned retry).
    `root` is the run archive dir (`artifacts/`); dumps live at `<episode.id>/<trace.id>`."""
    if root is None:
        return
    for dump in root.glob(f"*/{trace_id}"):
        shutil.rmtree(dump, ignore_errors=True)


def _flatten(source: str) -> str:
    return f"{source.strip('/').replace('/', '__')}.tar"


def _host_file(source: str, destination: str | None) -> str:
    """On-disk path under the trace archive dir. Destination is Harbor's host name."""
    if destination is None:
        return _flatten(source)
    path = destination.rstrip("/")
    if not path.endswith(".tar"):
        path = f"{path}.tar"
    return path

def _entries(
    runtime: Runtime,
    artifacts: list[Artifact] | None,
    extra: list[str],
) -> list[Artifact]:
    """Task path list plus eval extras, deduped after workdir resolve."""
    workdir = PurePosixPath(getattr(runtime.config, "workdir", "") or "/")
    entries: list[Artifact] = []
    seen: set[str] = set[str]()
    for artifact in artifacts or []:
        key = str(workdir / artifact.source)
        if key in seen:
            continue
        seen.add(key)
        entries.append(artifact)
    for source in extra:
        key = str(workdir / source)
        if key in seen:
            continue
        seen.add(key)
        entries.append(Artifact(source=source))
    return entries

def _resolved_destinations(
    workdir: PurePosixPath, destinations: Mapping[str, str]
) -> dict[str, str]:
    return {
        str(workdir / source): dest
        for source, dest in destinations.items()
    }

async def archive(
    runtime: Runtime,
    dest: Path,
    artifacts: list[Artifact] | None = None,
    config: ArchiveConfig | None = None,
    destinations: Mapping[str, str] | None = None,
) -> None:
    """Copy declared (and convention) artifact roots from `runtime` onto `dest`.

    Default inventory is `/logs/artifacts` plus `artifacts` (the task path list).
    `config.extra` merges additional sources. Best-effort: each root is a
    `ManifestEntry` with `status` `ok` (tar written) or `skipped` (missing, over
    `max_mb`, or host-path collision). `file` is the host tar name on `ok`, else
    `null`. Tar bytes are written as files; they are not stored on the trace.

    Host names default to a flattened `source` (`/app/x` → `app__x.tar`).
    `destinations` maps sandbox source to a relative path under `dest` (Harbor
    `destination`); restore is unchanged. First writer wins on a colliding host
    path; later rows are `status: skipped`.
    """
    dest.mkdir(parents=True, exist_ok=True)
    policy = config or ArchiveConfig()
    names = _resolved_destinations(
        PurePosixPath(getattr(runtime.config, "workdir", "") or "/"),
        destinations or {},
    )
    sources = _entries(runtime, artifacts, policy.extra)

    collected = await collect(
        runtime,
        sources,
        max_bytes=policy.max_mb * 1024 * 1024,
        missing="omit",
        on_limit="omit",
    )

    entries: list[ManifestEntry] = []
    claimed: set[str] = set()
    for source, blob in collected.items():
        destination = names.get(source)
        name = _host_file(source, destination)
        file: str | None = None
        status: ManifestStatus
        if name in claimed:
            logger.warning(
                "archive: host path %s already claimed; skipping %s", name, source
            )
            status = "skipped"
        elif blob is not None:
            claimed.add(name)
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            file = name
            status = "ok"
        else:
            status = "skipped"
        entries.append(
            ManifestEntry(
                source=source,
                file=file,
                harbor_destination=destination,
                status=status,
            )
        )
    (dest / "manifest.json").write_text(
        json.dumps([asdict(entry) for entry in entries], indent=2) + "\n"
    )
    logger.debug("archived %d artifact root(s) to %s", len(entries), dest)
