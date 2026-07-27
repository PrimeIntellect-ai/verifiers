"""Carry a task's declared files out of the agent's box and into a grading box.

Grading in the box the agent worked in leaves a seam a policy under RL pressure will
find. Grading in a second box removes it — only what the task declares crosses over.

Two channels, non-overlapping: `Trace.info` is the durable record (`capture_patch` puts
the diff there, a judge puts its verdict there) and never travels; `/logs/artifacts/` is
transport, Harbor's in-sandbox convention, collected with no declaration and restored at
the same path in the grading box ("no translation", as in Harbor).

`collect` runs while the agent's box is alive, right after `Task.finalize` produced the
files. It is the barrier: once it returns, nothing downstream needs the agent's box and
it can be torn down.
"""

from __future__ import annotations

import logging
import shlex
import uuid
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from pydantic import Field

from verifiers.v1.errors import ArtifactError
from verifiers.v1.types import StrictBaseModel

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

logger = logging.getLogger(__name__)

CONVENTION_DIR = "/logs/artifacts"
"""Harbor's in-sandbox publish directory, swept implicitly so a task that writes here
needs no declaration."""

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
"""Ceiling per collection. Sized for a delta, not a tree: the grading box boots from the
agent's image, so the repo is already there and only its output has to travel."""


class Artifact(StrictBaseModel):
    """One path to carry into the grading box, where it lands at this same path.

    Harbor's `ArtifactConfig` minus `destination` (host trial-directory placement, which
    verifiers has no equivalent for) and `service` (compose sidecars, unsupported).
    """

    source: str
    exclude: list[str] = Field(default_factory=list)
    """`tar --exclude` patterns, applied when `source` is a directory."""


async def collect(
    runtime: Runtime, artifacts: list[Artifact] | None = None
) -> dict[str, bytes]:
    """Tar the convention dir and every declared path out of `runtime`.

    Keyed by source path; the values are tar archives. Insertion order is the order
    they were declared, and a path cannot be collected twice.

    A declared source that is missing raises: it was declared because grading needs it,
    and grading a partial state scores the rollout wrong rather than failing it. The
    implicit convention sweep is exempt — most tasks never write there.

    One archive per source: BusyBox `tar` (every alpine-based image) implements only
    `c`/`x`/`t` with no `-r` to append, and each source carries its own excludes anyway.
    """
    # Harbor permits a relative source, and the probe below resolves one against the
    # runtime's workdir — so the tar, which runs `-C /`, has to agree or it archives a
    # different file. Joining also normalises `/work/` to `/work`, so one tree cannot
    # key two entries (the source is both the dict key and `restore`'s rm -rf target).
    workdir = PurePosixPath(getattr(runtime.config, "workdir", "") or "/")
    declared = [
        a.model_copy(update={"source": str(workdir / a.source)})
        for a in artifacts or []
    ]
    convention = PurePosixPath(CONVENTION_DIR)
    sweep = not any(
        (p := PurePosixPath(a.source)) == convention
        or p.is_relative_to(convention)
        or convention.is_relative_to(p)
        for a in declared
    )
    entries = ([Artifact(source=CONVENTION_DIR)] if sweep else []) + declared

    collected: dict[str, bytes] = {}
    budget = MAX_ARTIFACT_BYTES
    for artifact in entries:
        source = artifact.source
        if (await runtime.run(["test", "-e", source], {})).exit_code != 0:
            if sweep and source == CONVENTION_DIR:
                continue
            raise ArtifactError(
                f"declared artifact {source!r} does not exist in the box; the task must "
                "produce it in finalize() (or a [[verifier.collect]] hook)"
            )
        archive = await _tar_out(runtime, artifact, budget)
        budget -= len(archive)
        collected[source] = archive

    logger.debug("collected artifact roots: %s", list(collected))
    return collected


async def restore(runtime: Runtime, collected: dict[str, bytes]) -> None:
    """Extract `collected` in `runtime` at the original absolute paths."""
    if not collected:
        return
    # Extraction writes to absolute paths. In a container that is the point; under the
    # subprocess runtime it is the developer's own filesystem.
    if getattr(runtime.config, "type", None) == "subprocess":
        raise ArtifactError(
            "refusing to restore artifacts into the subprocess runtime: extraction "
            "writes to absolute paths on the host. Grade in a container."
        )
    # Clear every root up front, not per entry: a later nested root would otherwise
    # delete content an earlier one just restored. Clearing also drops any file or
    # symlink the image left at the target.
    roots = " ".join(shlex.quote(root) for root in collected)
    await _run(runtime, f"rm -rf -- {roots}", "clear artifact roots")
    for root, archive in collected.items():
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
            raise ArtifactError(
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
        raise ArtifactError(f"failed to {action}: {detail}")
