"""Durable host-side copies of sandbox artifacts, independent of grading transport."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from verifiers.v1.configs.archive import ArchiveConfig
from verifiers.v1.utils.artifacts import Artifact, collect

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

logger = logging.getLogger(__name__)


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


def _resolved_destinations(
    workdir: PurePosixPath, destinations: Mapping[str, str]
) -> dict[str, str]:
    return {
        _resolved_source(workdir, source): dest
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
    `config.extra` merges additional sources; `config.exclude` is applied to every
    root, including the convention dir. Best-effort: missing sources are recorded,
    not raised. Tar bytes are written as files; they are not stored on the trace.

    Host names default to a flattened `source` (`/app/x` → `app__x.tar`).
    `destinations` maps sandbox source to a relative path under `dest` (Harbor
    `destination`); restore is unchanged. First writer wins on a colliding host path.
    """
    dest.mkdir(parents=True, exist_ok=True)
    policy = config or ArchiveConfig()
    names = _resolved_destinations(
        PurePosixPath(getattr(runtime.config, "workdir", "") or "/"),
        destinations or {},
    )
    optional = _entries(runtime, artifacts, policy.extra)
    collected = await collect(
        runtime,
        optional,
        exclude=policy.exclude,
        max_bytes=policy.max_mb * 1024 * 1024,
    )
    entries: list[dict] = []
    claimed: set[str] = set()
    for source, blob in collected.items():
        destination = names.get(source)
        name = _host_file(source, destination)
        if name in claimed:
            entries.append(
                {
                    "source": source,
                    "file": name,
                    "destination": destination,
                    "present": False,
                    "error": "collision",
                }
            )
            continue
        claimed.add(name)
        present = blob is not None
        if present:
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        entries.append(
            {
                "source": source,
                "file": name,
                "destination": destination,
                "present": present,
                "error": None,
            }
        )
    (dest / "manifest.json").write_text(
        json.dumps({"entries": entries}, indent=2) + "\n"
    )
    logger.debug("archived %d artifact root(s) to %s", len(entries), dest)
