"""Carry a rollout's grading inputs out of the agent's box and into a fresh one.

A task that is graded in the box the agent worked in has no defense against a policy
under RL pressure: a test file it can edit, grading state it can tamper with, an
artifact that leaks the expected answer. Grading in a *second* box removes the seam —
but only what the task declares crosses over, so the grader sees the agent's output
rather than the agent's environment.

Two channels, deliberately separate:

- ``Trace.info`` is the durable record. `capture_patch` puts the diff there, the agentic
  judge puts its verdict there, and both ride ``traces.jsonl``. It is never a transport
  mechanism.
- ``/logs/artifacts/`` is transport. Harbor's in-sandbox convention: anything written
  there is collected with no declaration at all. Content goes box -> host -> box and is
  discarded once the grading box has it.

`collect` runs while the agent's box is still alive (right after `Task.finalize`, which
is where a task snapshots state into files). It is the barrier: once it returns, the
agent's box can be torn down in the background because everything grading needs is on
the host. `restore` then places that content in the grading box at its original paths —
"no translation", matching Harbor, so a verifier script finds its inputs where the task
author put them.

Strict by design, unlike Harbor's best-effort collection: Harbor collects for
observability, where a dropped file costs a log line. Here a dropped file means grading
against an incomplete state and scoring a rollout wrong, which is worse than failing it.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shlex
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from verifiers.v1.errors import ArtifactError
from verifiers.v1.types import StrictBaseModel

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

logger = logging.getLogger(__name__)

CONVENTION_DIR = "/logs/artifacts"
"""Harbor's in-sandbox publish directory, collected implicitly. A task that writes here
needs no `artifacts` declaration at all; an explicit entry for this path replaces the
implicit one (and, being explicit, is then required to exist)."""

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
"""Ceiling on the collected archive. Sized for a delta, not a tree: the grading box boots
from the same image as the agent's, so the repo is already there and only what the agent
produced has to travel."""

MAX_ARTIFACT_FILES = 10_000

_SYSTEM_ROOTS = frozenset(
    {
        "/",
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/lib",
        "/lib64",
        "/proc",
        "/root",
        "/sbin",
        "/sys",
        "/usr",
        "/var",
    }
)
"""Refused as artifact sources: collecting one would sweep up the image rather than the
agent's work, and restoring it would overwrite the grading box's own system files."""


class Artifact(StrictBaseModel):
    """One path to carry from the agent's box into the grading box.

    Mirrors the subset of Harbor's `ArtifactConfig` that means something here. Harbor's
    `destination` and `service` are deliberately absent: `destination` positions a file
    in a host trial directory, which verifiers does not have (the trace is the record),
    and `service` addresses a compose sidecar, which no runtime supports yet.
    """

    source: str
    """Absolute path in the agent's box. Re-materializes at this same path in the
    grading box."""
    exclude: list[str] = []
    """`tar --exclude` patterns, applied when `source` is a directory."""


@dataclass(frozen=True)
class CollectedArtifact:
    """One declared source, tarred out of the agent's box."""

    root: str
    """Absolute path this archive covers. Cleared in the grading box before extraction,
    so a file baked into the image cannot survive underneath restored content and be
    mistaken for the agent's work."""
    archive: bytes


@dataclass(frozen=True)
class Collected:
    """Artifact content held on the host between the two boxes. Transient — it is not
    persisted and not part of the trace; only `Trace.info` is durable.

    One archive per source rather than one combined tar: BusyBox `tar` (every
    alpine-based image) implements only `c`/`x`/`t`, with no `-r` to append to an
    existing archive, and each source carries its own `exclude` patterns so they cannot
    share a single create either.
    """

    entries: list[CollectedArtifact]

    @property
    def roots(self) -> list[str]:
        return [entry.root for entry in self.entries]

    @property
    def is_empty(self) -> bool:
        return not self.entries

    @property
    def total_bytes(self) -> int:
        return sum(len(entry.archive) for entry in self.entries)


def _normalize(artifacts: list[Artifact]) -> list[Artifact]:
    """Resolve, validate and de-conflict declared sources.

    Overlapping entries raise rather than following Harbor's keep-the-first-and-warn:
    there, a skipped entry costs a log line; here it silently narrows what gets graded.
    """
    seen: list[Artifact] = []
    for artifact in artifacts:
        path = PurePosixPath(artifact.source)
        if not path.is_absolute():
            raise ArtifactError(
                f"artifact source {artifact.source!r} must be an absolute path in the box"
            )
        if ".." in path.parts:
            raise ArtifactError(
                f"artifact source {artifact.source!r} may not contain '..'"
            )
        resolved = path.as_posix().rstrip("/") or "/"
        if resolved in _SYSTEM_ROOTS:
            raise ArtifactError(
                f"artifact source {resolved!r} is a system directory; declare the "
                "specific paths the grader needs instead"
            )
        for other in seen:
            a, b = PurePosixPath(resolved), PurePosixPath(other.source)
            if a == b or a.is_relative_to(b) or b.is_relative_to(a):
                raise ArtifactError(
                    f"artifact sources {other.source!r} and {resolved!r} overlap; "
                    "one would be silently dropped"
                )
        seen.append(artifact.model_copy(update={"source": resolved}))
    return seen


def _vet(archive: bytes) -> None:
    """Reject an archive whose members could escape their roots on extraction.

    The agent chose this content, so it is untrusted even though we built the tar: an
    absolute or `..`-bearing member would write outside the declared roots when the
    grading box extracts at `/`, and a symlink or device node could redirect a later
    write. Vetting here means `restore` extracts something already checked.
    """
    total = files = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar:
                name = member.name
                path = PurePosixPath(name)
                if not name or path.is_absolute() or ".." in path.parts:
                    raise ArtifactError(
                        f"artifact archive contains unsafe path {name!r}"
                    )
                if not (member.isfile() or member.isdir()):
                    raise ArtifactError(
                        f"artifact {name!r} is neither a regular file nor a directory "
                        "(symlinks and special files are refused)"
                    )
                files += 1
                total += member.size if member.isfile() else 0
                if files > MAX_ARTIFACT_FILES:
                    raise ArtifactError(
                        f"artifacts exceed {MAX_ARTIFACT_FILES} files; narrow the "
                        "declared sources or add `exclude` patterns"
                    )
                if total > MAX_ARTIFACT_BYTES:
                    raise ArtifactError(_over_cap(total))
    except tarfile.TarError as exc:
        raise ArtifactError(f"unreadable artifact archive: {exc}") from exc


def _over_cap(size: int) -> str:
    return (
        f"artifacts total {size} bytes, over the {MAX_ARTIFACT_BYTES} byte limit. The "
        "grading box boots from the agent's image, so only the delta needs to travel — "
        "declare narrower sources, add `exclude` patterns, or raise MAX_ARTIFACT_BYTES."
    )


async def collect(
    runtime: Runtime, artifacts: list[Artifact] | None = None
) -> Collected:
    """Pull the convention dir and every declared path out of `runtime`, as one archive.

    Call once the agent has finished and `Task.finalize` has run, while the box is still
    alive. Returning is the barrier the box's teardown may proceed behind.

    A declared source that is missing raises — it was declared because grading needs it.
    The implicit convention dir is exempt: it is injected for every task, and most tasks
    never write to it.
    """
    declared = _normalize(list(artifacts or []))
    entries = list(declared)
    optional: set[str] = set()
    # Inject the convention sweep only when nothing declared touches it. A task that
    # names a path inside `/logs/artifacts/` has said precisely what it needs, and that
    # entry is required; sweeping the parent as well would overlap it and raise.
    convention = PurePosixPath(CONVENTION_DIR)
    if not any(
        (p := PurePosixPath(a.source)) == convention
        or p.is_relative_to(convention)
        or convention.is_relative_to(p)
        for a in declared
    ):
        entries.insert(0, Artifact(source=CONVENTION_DIR))
        optional = {CONVENTION_DIR}

    collected: list[CollectedArtifact] = []
    budget = MAX_ARTIFACT_BYTES
    for artifact in entries:
        source = artifact.source
        probe = await runtime.run(["test", "-e", source], {})
        if probe.exit_code != 0:
            if source in optional:
                continue
            raise ArtifactError(
                f"declared artifact {source!r} does not exist in the box; the task "
                "must produce it in finalize() (or a [[verifier.collect]] hook)"
            )
        archive = await _tar_out(runtime, artifact, budget)
        _vet(archive)
        budget -= len(archive)
        collected.append(CollectedArtifact(root=source, archive=archive))

    logger.debug(
        "collected %d artifact root(s): %s", len(collected), [c.root for c in collected]
    )
    return Collected(entries=collected)


async def _tar_out(runtime: Runtime, artifact: Artifact, budget: int) -> bytes:
    """Tar one source out of the box, refusing it in-box if it blows the budget."""
    path = f"/tmp/vf-artifact-{uuid.uuid4().hex}.tar"
    excludes = " ".join(f"--exclude={shlex.quote(p)}" for p in artifact.exclude)
    try:
        await _run(
            runtime,
            f"tar -cf {shlex.quote(path)} -C / {excludes} -- "
            f"{shlex.quote(artifact.source.lstrip('/'))}",
            f"collect artifact {artifact.source!r}",
        )
        # Size it in the box: an oversized collection must be refused before it is
        # pulled into host memory, not after.
        sized = await runtime.run(["sh", "-c", f"wc -c < {shlex.quote(path)}"], {})
        if sized.exit_code == 0 and (raw := sized.stdout.strip()).isdigit():
            if int(raw) > budget:
                raise ArtifactError(_over_cap(MAX_ARTIFACT_BYTES - budget + int(raw)))
        return await runtime.read(path)
    finally:
        # Best-effort: a leftover tar in a box about to be destroyed is harmless, and
        # the name is unique per call on a shared-filesystem runtime.
        try:
            await runtime.run(["rm", "-f", path], {})
        except Exception:  # noqa: BLE001 - cleanup must never mask a collection error.
            logger.debug("failed to remove %s", path, exc_info=True)


async def restore(runtime: Runtime, collected: Collected) -> None:
    """Place collected content in `runtime` at its original absolute paths.

    "No translation", matching Harbor: a verifier script finds its inputs where the task
    author wrote them. Each root is cleared first so a file baked into the image cannot
    survive beneath restored content and be mistaken for the agent's work.
    """
    if collected.is_empty:
        return
    # Restoring clears each root and extracts at `/`. In a container that is the point;
    # on the subprocess runtime `/` is the developer's own machine, so refuse rather
    # than rm -rf a host path that happens to match an artifact source.
    if getattr(runtime.config, "type", None) == "subprocess":
        raise ArtifactError(
            "refusing to restore artifacts into the subprocess runtime: extraction "
            "writes to absolute paths on the host. Grade in a container "
            "(--env.judge.runtime.type docker or prime)."
        )
    # Clear every root before extracting any of them: doing it per entry would let an
    # earlier entry's restored content be deleted by a later, nested root.
    roots = " ".join(shlex.quote(root) for root in collected.roots)
    await _run(runtime, f"rm -rf -- {roots}", "clear artifact roots")
    for entry in collected.entries:
        path = f"/tmp/vf-artifact-{uuid.uuid4().hex}.tar"
        await runtime.write(path, entry.archive)
        await _run(
            runtime,
            f"tar -xf {shlex.quote(path)} -C / && rm -f {shlex.quote(path)}",
            f"restore artifact {entry.root!r}",
        )


def release(runtime: Runtime) -> None:
    """Begin `runtime`'s teardown and return without waiting for it.

    Once `collect` has returned, nothing downstream needs the agent's box, and on a
    remote runtime its teardown is an API round trip the grading box should not wait
    behind. The runtime stays in the runtimes module's `_LIVE` weakset, so the atexit
    backstop still frees it if the loop dies first.

    Safe to call inside a `provision()` block only if the caller detaches that context
    first (`AsyncExitStack.pop_all()`); otherwise the context manager's own `stop()`
    awaits a teardown anyway and nothing is gained.
    """
    if runtime.stopped:
        return
    # Mark it synchronously, mirroring `Runtime.stop`'s own "before the await" rule.
    # `create_task` only schedules, so without this a second `release()` — or a
    # `provision()` exit that beats the loop to it — would start a second teardown.
    runtime.stopped = True
    task = asyncio.create_task(runtime.stop())
    # asyncio keeps only a weak reference to a running task, so a fire-and-forget
    # teardown can be garbage collected mid-flight. Hold it until it finishes.
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)


_PENDING: set[asyncio.Task] = set()


async def _run(runtime: Runtime, command: str, action: str) -> None:
    result = await runtime.run(["sh", "-c", command], {})
    if result.exit_code:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise ArtifactError(f"failed to {action}: {detail}")
