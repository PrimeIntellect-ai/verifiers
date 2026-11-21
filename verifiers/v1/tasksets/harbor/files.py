"""Harbor input selection and content identities.

Harbor 0.24 supplies pathspec but predates its task file-filter helper. Ignore
rules belong to tests/ and solution/; image build contexts use .dockerignore.
"""

import hashlib
import io
import json
import os
import stat
import tarfile
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathspec import GitIgnoreSpec


class TaskFiles:
    def __init__(self, task_dir: Path):
        self.directory = Path(os.path.abspath(task_dir))
        self.root = task_dir.resolve()
        self.rules: dict[Path, GitIgnoreSpec | None] = {}

    def _excluded(self, path: Path) -> bool:
        relative = path.relative_to(self.root)
        if path.name == ".harborignore":
            return True
        if not relative.parts or relative.parts[0] not in {"tests", "solution"}:
            return False
        scopes = []
        candidate = self.root / relative.parts[0]
        for part in relative.parts[1:]:
            if candidate not in self.rules:
                ignore = candidate / ".harborignore"
                if not ignore.resolve().is_relative_to(self.root):
                    raise ValueError(f"Ignore file escapes task directory: {ignore}")
                if ignore.exists() and not ignore.is_file():
                    raise ValueError(f"Ignore rules must be a regular file: {ignore}")
                rules = None
                if ignore.is_file():
                    # Plain Harbor tasks work without the optional Harbor extra.
                    # Only parsing authored ignore rules needs its dependency.
                    from pathspec import GitIgnoreSpec

                    rules = GitIgnoreSpec.from_lines(ignore.read_text().splitlines())
                self.rules[candidate] = rules
            if (rules := self.rules[candidate]) is not None:
                scopes.append((candidate, rules))
            candidate /= part
            suffix = "/" if candidate != path or path.is_dir() else ""
            excluded = False
            for directory, rules in scopes:
                match = rules.check_file(
                    candidate.relative_to(directory).as_posix() + suffix
                )
                if match.include is not None:
                    excluded = match.include
            if excluded:
                return True
        return False

    def excluded(self, path: Path) -> bool:
        path = Path(os.path.abspath(path))
        if path.is_relative_to(self.directory):
            path = self.root / path.relative_to(self.directory)
        if not path.is_relative_to(self.root):
            raise ValueError(f"Task input escapes task directory: {path}")
        if self._excluded(path):
            return True
        resolved = path.resolve()
        return (
            resolved != path
            and resolved.is_relative_to(self.root)
            and self._excluded(resolved)
        )

    def require(self, path: Path) -> None:
        if self.excluded(path):
            raise ValueError(
                f"Required task input is excluded by .harborignore: {path}"
            )

    def inputs(self, directory: Path, *, include_rules: bool = False) -> Iterator[Path]:
        """Walk selected inputs in archive order, pruning excluded directories."""
        for path in sorted(directory.iterdir()):
            if self.excluded(path) and not (
                include_rules and path.name == ".harborignore"
            ):
                continue
            yield path
            if path.is_dir() and not path.is_symlink():
                yield from self.inputs(path, include_rules=include_rules)


def source_digest(directory: Path, *, task_dir: Path | None = None) -> str:
    """Hash selected names, types, permissions, bytes, links and ignore rules."""
    selected = TaskFiles(task_dir if task_dir is not None else directory)
    digest = hashlib.sha256()
    if not directory.is_dir():
        return digest.hexdigest()
    hardlinks: dict[tuple[int, int], str] = {}
    for path in selected.inputs(directory, include_rules=True):
        metadata = path.lstat()
        mode = metadata.st_mode
        relative = path.relative_to(directory).as_posix()
        link = None
        if stat.S_ISLNK(mode):
            content = os.fsencode(os.readlink(path))
        elif stat.S_ISREG(mode):
            link = hardlinks.setdefault((metadata.st_dev, metadata.st_ino), relative)
            with path.open("rb") as stream:
                content = hashlib.file_digest(stream, "sha256").digest()
        elif stat.S_ISDIR(mode):
            content = b""
        else:
            raise ValueError(
                f"Task input is not a regular file, directory or link: {path}"
            )
        digest.update(
            json.dumps([relative, mode & 0o177777, link]).encode()
            + b"\0"
            + content
            + b"\0"
        )
    return digest.hexdigest()


def make_tar(directory: Path) -> bytes:
    # Include rule changes in the cache key, and never serve stale task content.
    directory = directory.parent.resolve() / directory.name
    return _make_tar(directory, source_digest(directory, task_dir=directory.parent))


@lru_cache(maxsize=2)
def _make_tar(directory: Path, digest: str) -> bytes:
    selected = TaskFiles(directory.parent)
    if directory.name == "tests":
        selected.require(directory / "test.sh")
    buffer = io.BytesIO()

    def include(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        path = directory / member.name
        if not path.resolve().is_relative_to(selected.root):
            raise ValueError(f"Task input escapes task directory: {path}")
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise ValueError(f"Unsupported task input: {path}")
        return member

    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        # TarFile.gettarinfo remembers hard-link inodes before calling its
        # filter, so only selected paths enter the archive.
        for path in selected.inputs(directory):
            archive.add(
                path,
                arcname=str(path.relative_to(directory)),
                recursive=False,
                filter=include,
            )
    return buffer.getvalue()
