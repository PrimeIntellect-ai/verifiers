"""Artifact collection and restoration across runtimes."""

from __future__ import annotations

import logging
import shlex
import uuid
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "/logs/artifacts"
"""Implicit artifact directory; tasks that write here need no declaration."""

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
"""Ceiling per collection. Sized for a delta, not a tree: the grading box boots from the
agent's image, so the repo is already there and only its output has to travel."""


class Artifact(BaseModel):
    """One path to restore at the same location in another runtime."""

    source: str
    exclude: list[str] = Field(default_factory=list)
    """`tar --exclude` patterns, applied when `source` is a directory."""
    required: bool = True


async def collect(
    runtime: Runtime, artifacts: list[Artifact] | None = None
) -> dict[str, bytes | None]:
    """Tar the convention dir and every declared path out of `runtime`.

    Keyed by source path; the values are tar archives. Insertion order is the order
    they were declared, and a path cannot be collected twice.

    A declared source that is missing raises: it was declared because grading needs it,
    and grading a partial state scores the rollout wrong rather than failing it. The
    implicit convention sweep is exempt — most tasks never write there.

    Each source is archived separately so its exclude patterns stay local.
    """
    # Resolve relative sources against the runtime workdir. Joining also normalises
    # `/work/` to `/work`, so one tree cannot key two entries (the source is both the
    # dict key and `restore`'s rm -rf target).
    workdir = PurePosixPath(getattr(runtime.config, "workdir", "") or "/")
    declared = [
        a.model_copy(update={"source": str(workdir / a.source)})
        for a in artifacts or []
    ]
    convention = PurePosixPath(ARTIFACTS_DIR)
    declared_paths = [PurePosixPath(artifact.source) for artifact in declared]
    if convention in declared_paths:
        entries = declared
    else:
        sweep_excludes = [
            str(path).lstrip("/")
            for path in declared_paths
            if path.is_relative_to(convention)
        ]
        entries = [
            Artifact(
                source=ARTIFACTS_DIR,
                exclude=sweep_excludes,
                required=False,
            )
        ]
        for artifact, path in zip(declared, declared_paths, strict=True):
            if convention.is_relative_to(path):
                artifact = artifact.model_copy(
                    update={
                        "exclude": [
                            *artifact.exclude,
                            str(convention).lstrip("/"),
                        ]
                    }
                )
            entries.append(artifact)

    seen: set[str] = set()
    for artifact in entries:
        if artifact.source in seen:
            raise RuntimeError(f"artifact {artifact.source!r} declared more than once")
        seen.add(artifact.source)

    collected: dict[str, bytes | None] = {}
    budget = MAX_ARTIFACT_BYTES
    for artifact in entries:
        source = artifact.source
        exists = f"test -e {shlex.quote(source)} || test -L {shlex.quote(source)}"
        if (await runtime.run(["sh", "-c", exists], {})).exit_code != 0:
            if not artifact.required:
                collected[source] = None
                continue
            raise RuntimeError(
                f"declared artifact {source!r} does not exist in the runtime"
            )
        archive = await _tar_out(runtime, artifact, budget)
        budget -= len(archive)
        collected[source] = archive

    logger.debug("collected artifact roots: %s", list(collected))
    return collected


async def restore(runtime: Runtime, collected: dict[str, bytes | None]) -> None:
    """Extract `collected` in `runtime` at the original absolute paths."""
    if not collected:
        return
    # Restoring into the subprocess runtime would extract absolute paths onto the
    # developer's filesystem, so refuse it before any archive reaches the host.
    if getattr(runtime.config, "type", None) == "subprocess":
        raise RuntimeError(
            "refusing to restore artifacts into the subprocess runtime: extraction "
            "writes to absolute paths on the host. Grade in a container."
        )
    # Clear every root up front, not per entry: a later nested root would otherwise
    # delete content an earlier one just restored. Clearing also drops any file or
    # symlink the image left at the target.
    roots = " ".join(shlex.quote(root) for root in collected)
    await _run(runtime, f"rm -rf -- {roots}", "clear artifact roots")
    for root, archive in collected.items():
        if archive is None:
            continue
        path = f"/tmp/vf-artifact-{uuid.uuid4().hex}.tar"
        await runtime.write(path, archive)
        await _run(
            runtime,
            f"tar -xf {shlex.quote(path)} -C / && rm -f {shlex.quote(path)}",
            f"restore artifact {root!r}",
        )


async def _tar_out(runtime: Runtime, artifact: Artifact, budget: int) -> bytes:
    path = f"/tmp/vf-artifact-{uuid.uuid4().hex}.tar"
    excludes = " ".join(f"--exclude={shlex.quote(p)}" for p in artifact.exclude)
    try:
        await _run(
            runtime,
            f"tar -cf {shlex.quote(path)} -C / {excludes} -- "
            f"{shlex.quote(artifact.source.lstrip('/'))}",
            f"collect artifact {artifact.source!r}",
        )
        # Size it in the box: an oversized collection is refused before it reaches host
        # memory, not after.
        sized = await runtime.run(["sh", "-c", f"wc -c < {shlex.quote(path)}"], {})
        if (raw := sized.stdout.strip()).isdigit() and int(raw) > budget:
            raise RuntimeError(
                f"artifact {artifact.source!r} takes the collection over the "
                f"{MAX_ARTIFACT_BYTES} byte limit. The grading box boots from the "
                "agent's image, so only the delta needs to travel — narrow the source "
                "or add `exclude` patterns."
            )
        return await runtime.read(path)
    finally:
        # Best-effort: the box is about to be destroyed and the name is unique per call.
        try:
            await runtime.run(["rm", "-f", path], {})
        except Exception:
            logger.debug("failed to remove %s", path, exc_info=True)


async def _run(runtime: Runtime, command: str, action: str) -> None:
    result = await runtime.run(["sh", "-c", command], {})
    if result.exit_code:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(f"failed to {action}: {detail}")
