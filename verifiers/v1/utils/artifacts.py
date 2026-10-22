"""Bounded, validated artifact collection and restoration across runtimes."""

from __future__ import annotations

import io
import json
import logging
import posixpath
import shlex
import shutil
import tarfile
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "/logs/artifacts"
"""Implicit artifact directory; tasks that write here need no declaration."""

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
"""Default ceiling per collection. Sized for a delta, not a tree: the grading box boots from the
agent's image, so the repo is already there and only its output has to travel."""


class Artifact(BaseModel):
    """One path to restore at the same location in another runtime."""

    source: str
    destination: str | None = None
    """Saved path relative to the trace's artifact directory; grading uses `source`."""
    exclude: list[str] = Field(default_factory=list)
    """`tar --exclude` patterns, applied when `source` is a directory."""
    required: bool = True

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str | None) -> str | None:
        if not value:
            return None
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or "\\" in value
            or path.parts[0] == "manifest.json"
        ):
            raise ValueError(
                "artifact destination must be a relative path without '..' or "
                "backslashes, outside the reserved manifest.json path"
            )
        return str(path)


async def collect(
    runtime: Runtime,
    artifacts: list[Artifact] | None = None,
    *,
    max_bytes: int = MAX_ARTIFACT_BYTES,
    sweep: bool = True,
) -> dict[str, bytes | None]:
    """Tar the convention dir (when `sweep`) and every declared path out of `runtime`.

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
    if not sweep or convention in declared_paths:
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

    # Batch roots to save remote round trips. Leave room below the shell argument
    # limit for the command itself and further quoting by runtime transports.
    batches: list[list[str]] = [[]]
    batch_bytes = 0
    for artifact in entries:
        source_bytes = len(shlex.quote(artifact.source).encode()) + 1
        if batches[-1] and batch_bytes + source_bytes > 8 * 1024:
            batches.append([])
            batch_bytes = 0
        batches[-1].append(artifact.source)
        batch_bytes += source_bytes

    existence: list[str] = []
    for sources in batches:
        output = await _run(
            runtime,
            f"for source in {shlex.join(sources)}; do "
            'if test -e "$source" || test -L "$source"; then echo 1; else echo 0; fi; '
            "done",
            "check artifact roots",
        )
        existence.extend(output.splitlines())
    collected: dict[str, bytes | None] = {}
    budget = max_bytes
    for artifact, exists in zip(entries, existence, strict=True):
        source = artifact.source
        if exists != "1":
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


def save_artifacts(
    directory: Path,
    collected: dict[str, bytes | None],
    artifacts: list[Artifact],
    *,
    workdir: str = "/",
) -> None:
    """Save collected files at their declared destinations, with a manifest.

    Unpack each validated archive in isolation before moving its files into the
    saved tree. Failed or conflicting entries are recorded without losing the
    completed trace. Source-mirrored directories may merge: collection splits
    nested declarations out of the implicit convention sweep.
    """
    declared = {str(PurePosixPath(workdir) / a.source): a for a in artifacts}
    manifest: list[dict] = []
    claims: list[tuple[PurePosixPath, bool]] = []
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        saved = Path(temporary) / "artifacts"
        saved.mkdir()
        (saved / "manifest.json").touch()
        for source, archive in collected.items():
            artifact = declared.get(source, Artifact(source=source))
            destination = artifact.destination or artifact.source.lstrip("/")
            entry = {
                "source": source,
                "service": getattr(artifact, "service", "main"),
                "destination": destination,
                "status": "empty",
            }
            manifest.append(entry)
            try:
                _validate_restore(source, archive)
                Artifact.validate_destination(artifact.destination)
                target = PurePosixPath(destination)
                entry["destination"] = str(target)
                if archive is None:
                    continue
                with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
                    members = tar.getmembers()
                    if not members:
                        continue
                    root = PurePosixPath(source).relative_to("/")
                    paths = [
                        saved / target / PurePosixPath(member.name).relative_to(root)
                        for member in members
                    ]
                    # Explicit renames claim a whole subtree. Source-mirrored
                    # entries may share directories, but never overwrite files.
                    if any(
                        (artifact.destination or renamed)
                        and (
                            target.is_relative_to(prior) or prior.is_relative_to(target)
                        )
                        for prior, renamed in claims
                    ) or any(
                        path.is_symlink()
                        or (path.exists() and not (member.isdir() and path.is_dir()))
                        or any(
                            parent.is_symlink()
                            or (parent.exists() and not parent.is_dir())
                            for parent in path.parents
                            if parent.is_relative_to(saved)
                        )
                        for member, path in zip(members, paths, strict=True)
                    ):
                        entry["status"] = "skipped"
                        entry["error"] = (
                            "destination overlaps a previously saved artifact"
                        )
                        logger.warning("artifact %r: %s", source, entry["error"])
                        continue
                    with tempfile.TemporaryDirectory(dir=temporary) as unpacked:
                        tar.extractall(unpacked, filter=_saved_member)
                        for member, path in zip(members, paths, strict=True):
                            if member.isdir():
                                path.mkdir(parents=True, exist_ok=True)
                            else:
                                path.parent.mkdir(parents=True, exist_ok=True)
                                (Path(unpacked) / member.name).replace(path)
                    claims.append((target, artifact.destination is not None))
                    entry["status"] = "ok"
            except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
                entry.update(status="failed", error=str(exc))
                logger.warning("failed to save artifact %r: %s", source, exc)
        (saved / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if directory.is_symlink():
            raise ValueError(f"artifact directory must not be a symlink: {directory}")
        if directory.exists():
            shutil.rmtree(directory)
        saved.replace(directory)


async def restore(
    runtime: Runtime,
    collected: dict[str, bytes | None],
    *,
    protected: Sequence[str] = (),
) -> None:
    """Extract `collected` in `runtime` at the original absolute paths.

    Archive bytes are untrusted: the agent controls both their source files and the
    runtime tooling that creates them. Every archive is therefore validated on the host
    before the grading runtime is changed. Regular files, directories, and links can
    travel; extraction through a symlink (one any archive restores) cannot.

    A link may point outside its root (a virtualenv's `bin/python` names the image's
    interpreter): extraction never follows it, and the box it lands in is the grader's
    own. `protected` names the paths the grader stages after restoring (its tests, its
    report): no restored link may land in one or on an ancestor of one, so a
    submission cannot read or replace the grader's files through its own paths.
    """
    if not collected:
        return
    # Restoring into the subprocess runtime would extract absolute paths onto the
    # developer's filesystem, so refuse it before any archive reaches the host.
    if getattr(runtime.config, "type", None) == "subprocess":
        raise RuntimeError(
            "refusing to restore artifacts into the subprocess runtime: extraction "
            "writes to absolute paths on the host. Grade in a container."
        )
    check_restore(collected, protected=protected)
    # Clear every root that restores up front, not per entry: a later nested root
    # would otherwise delete content an earlier one just restored. Clearing also drops
    # any file or symlink the image left at the target. A root the run left nothing
    # in (the convention dir of a run that never wrote there) is not cleared: the
    # grader keeps what its image has there.
    roots = " ".join(
        shlex.quote(root) for root, archive in collected.items() if archive is not None
    )
    # Container exec requires its configured cwd to exist, including while a
    # submission replaces the entire working directory (or one of its parents). The
    # convention dir exists for the grader as under Harbor, whose verifier always has it.
    workdir = shlex.quote(getattr(runtime.config, "workdir", None) or "/")
    clear = f"rm -rf -- {roots} && " if roots else ""
    await _run(
        runtime,
        f"{clear}mkdir -p -- {workdir} {shlex.quote(ARTIFACTS_DIR)}",
        "clear artifact roots",
    )
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


def check_restore(
    collected: dict[str, bytes | None], *, protected: Sequence[str] = ()
) -> None:
    """Raise `RuntimeError` for any archive `restore` would refuse, on the host and
    without a runtime: a member outside its root, a special file, an unsafe link, a
    path that traverses another archive's link, or a link that reaches `protected`.
    A caller can check a collected submission before it provisions a grader."""
    links: dict[PurePosixPath, str] = {}
    owners: dict[PurePosixPath, str] = {}
    for root, archive in collected.items():
        for member, target in _validate_restore(root, archive).items():
            owners[member] = root
            if target is not None:
                links[member] = target
    # Each archive is checked alone above. A link one archive restores must not
    # redirect a path another archive extracts beneath it.
    for member, root in owners.items():
        for parent in member.parents:
            if parent in links and owners[parent] != root:
                raise RuntimeError(
                    f"artifact member {str(member)!r} of {root!r} traverses the "
                    f"symlink {str(parent)!r} restored by {owners[parent]!r}"
                )
    guarded = [PurePosixPath(posixpath.normpath(path)) for path in protected]
    for link in links if guarded else ():
        landing = _resolve(link, links)
        # A link that lands on or inside the grader's files would let the submission
        # stand in for them. One onto an ancestor (`/`, a fixture tree's link to the
        # root) names none of them and restores, as Harbor restores every link;
        # nothing extracts through any link either way.
        if any(landing.is_relative_to(path) for path in guarded):
            raise RuntimeError(
                f"artifact symlink {str(link)!r} -> {links[link]!r} reaches the "
                "grader's own files"
            )


async def _tar_out(runtime: Runtime, artifact: Artifact, budget: int) -> bytes:
    path = f"/tmp/vf-artifact-{uuid.uuid4().hex}.tar"
    excludes = " ".join(f"--exclude={shlex.quote(p)}" for p in artifact.exclude)
    try:
        # macOS tar otherwise adds AppleDouble sidecars next to a directory root.
        size = await _run(
            runtime,
            f"COPYFILE_DISABLE=1 tar -cf {shlex.quote(path)} -C / {excludes} -- "
            f"{shlex.quote(artifact.source.lstrip('/'))} && wc -c < {shlex.quote(path)}",
            f"collect artifact {artifact.source!r}",
        )
        # An archive over the budget is the submission's size, not a transfer fault:
        # say so as the other submission defects do, before moving any bytes. The
        # bounded read below still enforces the budget whatever the box reports.
        reported = size.split()
        if reported and reported[-1].isdigit() and int(reported[-1]) > budget:
            raise RuntimeError(
                f"artifact {artifact.source!r} is {reported[-1]} bytes, over the "
                f"{budget}-byte collection budget"
            )
        # The runtime enforces the remaining collection budget while transferring the
        # bytes, so replacing or growing the archive cannot race a separate size probe.
        archive, dropped = _drop_special_files(
            await runtime.read(path, max_bytes=budget)
        )
        if dropped:
            logger.warning(
                "artifact %r: dropped %d FIFO or device node(s), e.g. %s",
                artifact.source,
                len(dropped),
                dropped[:3],
            )
        return archive
    finally:
        # Best-effort: the box is about to be destroyed and the name is unique per call.
        try:
            await runtime.run(["rm", "-f", path], {})
        except Exception:
            logger.debug("failed to remove %s", path, exc_info=True)


def _drop_special_files(archive: bytes) -> tuple[bytes, list[str]]:
    """Remove FIFOs and device nodes, plus hard links to them, from a collected archive.

    They carry no content, and `restore` refuses them, so one an agent leaves in its
    workspace (e.g. a named pipe it made to test streaming input) would otherwise fail
    the whole restore. Returns the archive and the dropped member names."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        members = source.getmembers()
        special = {m.name for m in members if m.isfifo() or m.ischr() or m.isblk()}
        if not special:
            return archive, []
        if any(m.issparse() for m in members):
            raise RuntimeError("cannot filter special files from sparse artifacts")
        dropped = [
            m.name
            for m in members
            if m.name in special or (m.islnk() and m.linkname in special)
        ]
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode="w:", format=source.format) as target:
            for member in members:
                if member.name in dropped:
                    continue
                content = source.extractfile(member) if member.isfile() else None
                target.addfile(member, content)
    return out.getvalue(), dropped


def _validate_restore(
    root: str, archive: bytes | None
) -> dict[PurePosixPath, str | None]:
    """Reject archive content that could restore outside its declared root. Returns
    every member's absolute path and, for a symlink, its target."""
    root_path = PurePosixPath(root)
    if (
        root_path.anchor != "/"
        or ".." in root_path.parts
        or root_path == PurePosixPath("/")
    ):
        raise RuntimeError(
            f"artifact root {root!r} must be an absolute path below '/' with no '..'"
        )
    if archive is None:
        return {}

    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            regular_files: set[PurePosixPath] = set()
            members: set[PurePosixPath] = set()
            symlinks: dict[PurePosixPath, str] = {}
            for member in tar:
                member_path = PurePosixPath(member.name)
                destination = PurePosixPath("/") / member_path
                if (
                    not member.name
                    or member_path.is_absolute()
                    or ".." in member_path.parts
                    or not destination.is_relative_to(root_path)
                ):
                    raise RuntimeError(
                        f"artifact member {member.name!r} is outside declared root "
                        f"{root!r}"
                    )
                if member_path in members:
                    raise RuntimeError(f"duplicate artifact member {member.name!r}")
                members.add(member_path)
                if member.issym():
                    symlinks[member_path] = member.linkname
                elif member.islnk():
                    # Tar deduplicates hard-linked build outputs (e.g. Cargo's build
                    # scripts). A target must already be a regular member of this
                    # archive, never a path supplied by the verifier image.
                    target = PurePosixPath(member.linkname)
                    if (
                        not member.linkname
                        or target.is_absolute()
                        or ".." in target.parts
                        or target not in regular_files
                        or target == member_path
                    ):
                        raise RuntimeError(
                            f"artifact hard link {member.name!r} has unsafe target "
                            f"{member.linkname!r}"
                        )
                elif not (member.isfile() or member.isdir()):
                    raise RuntimeError(
                        f"artifact member {member.name!r} is a link or special file"
                    )
                if member.isfile():
                    regular_files.add(member_path)
            # Check after reading the whole archive, so a later symlink cannot
            # turn an earlier member's parent or a link target into a redirection.
            for member_path in members:
                if any(parent in symlinks for parent in member_path.parents):
                    raise RuntimeError(
                        f"artifact member {str(member_path)!r} traverses a symlink"
                    )
            for member_path, linkname in symlinks.items():
                if not linkname or PurePosixPath("/") / member_path == root_path:
                    raise RuntimeError(
                        f"artifact symlink {str(member_path)!r} has unsafe target "
                        f"{linkname!r}"
                    )
    except tarfile.TarError as exc:
        raise RuntimeError(f"unreadable artifact archive for {root!r}: {exc}") from exc
    return {PurePosixPath("/") / path: symlinks.get(path) for path in sorted(members)}


def _saved_member(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
    """`save_artifacts`' extraction filter: the `data` filter, except that a symlink is
    saved as written, wherever it points (a virtualenv's `bin/python` names the image's
    interpreter), as `restore` keeps it. Writing one never follows it, and no member is
    written through one: `_validate_restore` refuses an archive with a member below a
    symlink."""
    if member.issym():
        return tarfile.tar_filter(member, path)
    return tarfile.data_filter(member, path)


MAX_LINK_DEPTH = 40
"""Links followed while resolving one path, as Linux's ELOOP limit."""


def _resolve(
    path: PurePosixPath, links: dict[PurePosixPath, str], depth: int = 0
) -> PurePosixPath:
    """Where `path` lands once the restored `links` are followed, the way the kernel
    walks it: each component in turn, `..` after the link before it resolved. Paths
    outside the restored trees are the grading image's own and resolve as written."""
    resolved = PurePosixPath("/")
    for part in path.parts[1:]:
        if part == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / part
        if candidate not in links:
            resolved = candidate
            continue
        if depth >= MAX_LINK_DEPTH:
            raise RuntimeError(f"artifact symlink {str(candidate)!r} loops")
        target = PurePosixPath(links[candidate])
        resolved = _resolve(
            target if target.is_absolute() else candidate.parent / target,
            links,
            depth + 1,
        )
    return resolved


async def _run(runtime: Runtime, command: str, action: str) -> str:
    result = await runtime.run(["sh", "-c", command], {})
    if result.exit_code:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(f"failed to {action}: {detail}")
    return result.stdout
