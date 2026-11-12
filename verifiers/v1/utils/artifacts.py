"""Bounded, validated artifact collection and restoration across runtimes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import re
import shlex
import shutil
import stat
import tarfile
import tempfile
import time
import uuid
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from verifiers.v1.utils.format import format_time
from verifiers.v1.utils.logging import get_logger

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime, RuntimeConfig

logger = get_logger(__name__)

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
            or path.parts[0] in ("manifest.json", ".archives")
        ):
            raise ValueError(
                "artifact destination must be a relative path without '..' or "
                "backslashes, outside reserved manifest.json and .archives paths"
            )
        return str(path)


def validate_artifact_mounts(config: RuntimeConfig, sources: Iterable[str]) -> None:
    """Keep mounted storage separate from artifact copying and destructive restoration."""
    mounts = getattr(config, "mounts", {})
    if not mounts:
        return
    workdir = getattr(config, "workdir", None) or "/app"
    for source in [ARTIFACTS_DIR, *sources]:
        path = posixpath.join(workdir, source)
        path = posixpath.normpath("/" + path.lstrip("/"))
        for target in mounts:
            common = posixpath.commonpath((path, target))
            if common in (target, path):
                raise ValueError(
                    f"artifact root {path!r} overlaps mount {target!r}; place artifacts "
                    "outside all mounts"
                )


async def validate_runtime_mounts(runtime: Runtime, sources: Iterable[str]) -> None:
    """Reject relocated mounts and aliases that bypass lexical overlap checks."""
    mounts = getattr(runtime.config, "mounts", {})
    if not mounts:
        return
    sources = list(sources)
    validate_artifact_mounts(runtime.config, sources)
    workdir = PurePosixPath(getattr(runtime.config, "workdir", None) or "/")
    paths: set[str] = set()
    for source in [*mounts, ARTIFACTS_DIR, *sources]:
        path = workdir / source
        # Keep '..' until after checking its preceding components for symlinks.
        paths.update(str(parent) for parent in (path, *path.parents))
    await _run(
        runtime,
        f"for mount_path in {shlex.join(sorted(paths))}; do "
        'if [ -L "$mount_path" ]; then '
        "printf 'mount or artifact path traverses symlink: %s\\n' "
        '"$mount_path" >&2; exit 1; fi; done',
        "validate mount and artifact paths",
    )
    # Read procfs through exec: file downloads may require seeking to EOF.
    mountinfo = await _run(runtime, "cat /proc/self/mountinfo", "read mount table")
    records = [line.split() for line in mountinfo.splitlines()]
    mounted = {record[4] for record in records}
    # Renaming a parent relocates a bind mount without introducing any symlinks.
    for target in mounts:
        escaped = re.sub(r"[ \t\n\\]", lambda m: f"\\{ord(m[0]):03o}", target)
        if escaped not in mounted:
            raise RuntimeError(
                f"mount target {target!r} is no longer mounted; "
                "do not move mount targets or their parent directories"
            )


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
    started = time.perf_counter()
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
        await validate_runtime_mounts(runtime, sources)
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

    logger.debug(
        "Collected artifacts in %s roots=%s",
        format_time(time.perf_counter() - started),
        list(collected),
    )
    return collected


def save_artifacts(
    directory: Path,
    collected: dict[str, bytes | None],
    artifacts: list[Artifact],
    *,
    workdir: str = "/",
) -> str:
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
        (saved / ".archives").mkdir()
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
                Artifact.validate_destination(destination)
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
                        logger.warning(
                            "Skipped artifact %s - %s", source, entry["error"]
                        )
                        continue
                    with tempfile.TemporaryDirectory(dir=temporary) as unpacked:
                        tar.extractall(unpacked, filter="data")
                        for member, path in zip(members, paths, strict=True):
                            if member.isdir():
                                path.mkdir(parents=True, exist_ok=True)
                            else:
                                path.parent.mkdir(parents=True, exist_ok=True)
                                (Path(unpacked) / member.name).replace(path)
                    claims.append((target, artifact.destination is not None))
                    entry["status"] = "ok"
                    canonical = _rebuild(archive)
                    digest = hashlib.sha256(canonical).hexdigest()
                    archive_path = f".archives/{digest}.tar"
                    (saved / archive_path).write_bytes(canonical)
                    entry.update(
                        archive=archive_path, sha256=digest, size=len(canonical)
                    )
            except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
                entry.update(status="failed", error=str(exc))
                logger.warning("Failed to save artifact %s - %s", source, exc)
        manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
        (saved / "manifest.json").write_bytes(manifest_bytes)
        if directory.is_symlink():
            raise ValueError(f"artifact directory must not be a symlink: {directory}")
        if directory.exists():
            shutil.rmtree(directory)
        saved.replace(directory)
    return hashlib.sha256(manifest_bytes).hexdigest()


def _read_saved_file(directory: Path, relative: str, limit: int) -> bytes:
    """Open a saved file through directory descriptors, rejecting links and FIFOs."""
    if limit < 0:
        raise ValueError("artifact byte budget must be non-negative")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    parent = os.open(directory, flags | os.O_DIRECTORY)
    try:
        parts = PurePosixPath(relative).parts
        for name in parts[:-1]:
            child = os.open(name, flags | os.O_DIRECTORY, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError(
                    f"invalid or oversized saved artifact file: {relative}"
                )
            content = stream.read(limit + 1)
            if len(content) > limit:
                raise ValueError(
                    f"saved artifact file exceeds its byte budget: {relative}"
                )
            return content
    finally:
        os.close(parent)


def load_artifacts(
    directory: Path,
    *,
    max_bytes: int = MAX_ARTIFACT_BYTES,
    expected_manifest_sha256: str | None = None,
) -> dict[str, bytes | None]:
    """Load complete, hash-checked submissions saved by `save_artifacts`.

    Extracted files are a convenient inspection view; regrading uses only the
    original canonical archives. Older manifests without archives cannot establish
    a complete submission and are rejected rather than silently reconstructed.
    """
    manifest_bytes = _read_saved_file(directory, "manifest.json", 1024 * 1024)
    if (
        expected_manifest_sha256 is not None
        and hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256
    ):
        raise ValueError("artifact manifest failed its integrity check")
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, list):
        raise TypeError("artifact manifest must be a list")
    collected: dict[str, bytes | None] = {}
    remaining = max_bytes
    for entry in manifest:
        if not isinstance(entry, dict) or not isinstance(entry.get("source"), str):
            raise TypeError("invalid artifact manifest entry")
        source = entry["source"]
        if source in collected:
            raise ValueError(f"duplicate artifact root {source!r}")
        _validate_restore(source, None)
        if entry.get("status") == "empty":
            collected[source] = None
            continue
        digest = entry.get("sha256")
        if (
            entry.get("status") != "ok"
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"artifact {source!r} has no complete saved archive")
        relative = f".archives/{digest}.tar"
        if entry.get("archive") != relative:
            raise ValueError(f"invalid archive path for {source!r}")
        archive = _read_saved_file(directory, relative, remaining)
        if len(archive) != entry.get("size"):
            raise ValueError(
                f"artifact {source!r} exceeds its recorded size or byte budget"
            )
        if hashlib.sha256(archive).hexdigest() != digest:
            raise ValueError(f"artifact {source!r} failed its integrity check")
        _validate_restore(source, archive)
        collected[source] = archive
        remaining -= len(archive)
    return collected


async def restore(runtime: Runtime, collected: dict[str, bytes | None]) -> None:
    """Extract `collected` in `runtime` at the original absolute paths.

    Archive bytes are untrusted: the agent controls both their source files and the
    runtime tooling that creates them. Every archive is therefore validated on the host
    before the grading runtime is changed. Regular files, directories, and confined links
    can travel; link chains and extraction through symlinks cannot.
    """
    if not collected:
        return
    await validate_runtime_mounts(runtime, collected)
    # Restoring into the subprocess runtime would extract absolute paths onto the
    # developer's filesystem, so refuse it before any archive reaches the host.
    if getattr(runtime.config, "type", None) == "subprocess":
        raise RuntimeError(
            "refusing to restore artifacts into the subprocess runtime: extraction "
            "writes to absolute paths on the host. Grade in a container."
        )
    for root, archive in collected.items():
        _validate_restore(root, archive)
    collected = {
        root: None if archive is None else _rebuild(archive)
        for root, archive in collected.items()
    }
    # Clear every root up front, not per entry: a later nested root would otherwise
    # delete content an earlier one just restored. Clearing also drops any file or
    # symlink the image left at the target.
    roots = " ".join(shlex.quote(root) for root in collected)
    # Container exec requires its configured cwd to exist, including while a
    # submission replaces the entire working directory (or one of its parents).
    workdir = shlex.quote(getattr(runtime.config, "workdir", None) or "/")
    await _run(
        runtime,
        f"rm -rf -- {roots} && mkdir -p -- {workdir}",
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


async def _tar_out(runtime: Runtime, artifact: Artifact, budget: int) -> bytes:
    path = f"/tmp/vf-artifact-{uuid.uuid4().hex}.tar"
    excludes = " ".join(f"--exclude={shlex.quote(p)}" for p in artifact.exclude)
    try:
        # macOS tar otherwise adds AppleDouble sidecars next to a directory root.
        await _run(
            runtime,
            f"COPYFILE_DISABLE=1 tar -cf {shlex.quote(path)} -C / {excludes} -- "
            f"{shlex.quote(artifact.source.lstrip('/'))}",
            f"collect artifact {artifact.source!r}",
        )
        # The runtime enforces the remaining collection budget while transferring the
        # bytes, so replacing or growing the archive cannot race a separate size probe.
        archive, dropped = _drop_special_files(
            await runtime.read(path, max_bytes=budget)
        )
        if dropped:
            logger.warning(
                "Dropped %d FIFO or device node(s) from artifact %s, e.g. %s",
                len(dropped),
                artifact.source,
                dropped[:3],
            )
        return archive
    finally:
        # Best-effort: the box is about to be destroyed and the name is unique per call.
        try:
            await runtime.run(["rm", "-f", path], {})
        except Exception:
            logger.debug("Failed to remove %s", path, exc_info=True)


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


def _rebuild(archive: bytes) -> bytes:
    """Write a fresh archive holding exactly the members Python's tarfile reads.

    Validation reads archives with Python's tarfile, but the grading box extracts
    them with its own tar. An agent controls the archive bytes, and two tar parsers
    can disagree about the same bytes, so the box only ever extracts an archive
    written here from the members validation saw. Owners and permission bits are
    kept (tasks rely on ownership, e.g. git's safe.directory check); setuid,
    setgid and sticky bits are not.
    """
    out = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source,
        tarfile.open(fileobj=out, mode="w:", format=tarfile.PAX_FORMAT) as target,
    ):
        for member in source:
            info = tarfile.TarInfo(member.name)
            info.mode = member.mode & 0o777
            info.uid, info.gid = member.uid, member.gid
            info.uname, info.gname = member.uname, member.gname
            info.mtime = member.mtime
            content = None
            if member.isfile():
                info.type = tarfile.REGTYPE
                info.size = member.size
                content = source.extractfile(member)
            elif member.isdir():
                info.type = tarfile.DIRTYPE
            elif member.issym() or member.islnk():
                info.type = tarfile.SYMTYPE if member.issym() else tarfile.LNKTYPE
                info.linkname = member.linkname
            else:
                raise RuntimeError(
                    f"artifact member {member.name!r} is a link or special file"
                )
            target.addfile(info, content)
    return out.getvalue()


def _validate_restore(root: str, archive: bytes | None) -> None:
    """Reject archive content that could restore outside its declared root."""
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
        return

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
                if member.issparse():
                    # A sparse member declares its full size in a few header bytes,
                    # and rebuilding the archive on the host would write every hole
                    # out as zeros. GNU tar only writes these with --sparse; bsdtar
                    # writes them for any file with holes.
                    raise RuntimeError(
                        f"artifact member {member.name!r} is a sparse file, which "
                        "cannot be restored"
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
            relative_root = root_path.relative_to("/")
            for member_path, linkname in symlinks.items():
                target = PurePosixPath(linkname)
                if not linkname or target.is_absolute():
                    raise RuntimeError(
                        f"artifact symlink {str(member_path)!r} has unsafe target "
                        f"{linkname!r}"
                    )
                resolved = member_path.parent
                if not resolved.is_relative_to(relative_root):
                    raise RuntimeError(
                        f"artifact symlink {str(member_path)!r} has unsafe target "
                        f"{linkname!r}"
                    )
                # Walk before normalizing '..': a/../b is unsafe if a is itself
                # a symlink. Missing internal targets are safe (e.g. build outputs).
                for part in target.parts:
                    resolved = resolved.parent if part == ".." else resolved / part
                    if not resolved.is_relative_to(relative_root) or any(
                        path in symlinks for path in (resolved, *resolved.parents)
                    ):
                        raise RuntimeError(
                            f"artifact symlink {str(member_path)!r} has unsafe target "
                            f"{linkname!r}"
                        )
    except tarfile.TarError as exc:
        raise RuntimeError(f"unreadable artifact archive for {root!r}: {exc}") from exc


async def _run(runtime: Runtime, command: str, action: str) -> str:
    result = await runtime.run(["sh", "-c", command], {})
    if result.exit_code:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(f"failed to {action}: {detail}")
    return result.stdout
