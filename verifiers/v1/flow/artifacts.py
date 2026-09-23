"""Optional immutable file revisions, independent of workflow state."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from pydantic import Field

from verifiers.v1.flow.unit import Unit

ArtifactRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
"""An immutable Git commit, addressed within its GitArtifacts repository."""
KEEP = "refs/flow/artifacts"
_IDENTITY = {
    "GIT_AUTHOR_NAME": "flow",
    "GIT_AUTHOR_EMAIL": "flow@local",
    "GIT_COMMITTER_NAME": "flow",
    "GIT_COMMITTER_EMAIL": "flow@local",
    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
}


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class GitArtifacts:
    """Writes produce retained revisions; selecting one is a separate workflow transition."""

    def __init__(self, unit: Unit | Path | str) -> None:
        self.path = (unit.path if isinstance(unit, Unit) else Path(unit)).resolve()
        if not (self.path / ".git").is_dir():
            self.path.mkdir(parents=True, exist_ok=True)
            git(self.path, "init", "-q")

    def read_bytes(self, revision: ArtifactRevision, path: str) -> bytes | None:
        self._relative(path)
        git(self.path, "cat-file", "-e", f"{revision}^{{commit}}")
        out = subprocess.run(
            ["git", "-C", str(self.path), "show", f"{revision}:{path}"],
            capture_output=True,
            check=False,
        )
        return out.stdout if out.returncode == 0 else None

    def read(self, revision: ArtifactRevision, path: str) -> str | None:
        value = self.read_bytes(revision, path)
        return value.decode() if value is not None else None

    def listing(self, revision: ArtifactRevision, prefix: str = "") -> list[str]:
        return git(
            self.path,
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "--",
            *([prefix] if prefix else []),
        ).splitlines()

    def archive(
        self,
        revision: ArtifactRevision,
        *,
        prefix: str = "",
        only: tuple[str, ...] = (),
    ) -> bytes:
        return subprocess.run(
            [
                "git",
                "-C",
                str(self.path),
                "archive",
                "--format=tar",
                f"--prefix={prefix.rstrip('/') + '/' if prefix else ''}",
                revision,
                *only,
            ],
            capture_output=True,
            check=True,
        ).stdout

    @staticmethod
    def _relative(path: str) -> Path:
        rel = Path(path)
        if (
            not rel.parts
            or rel.is_absolute()
            or any(p in ("..", ".git") for p in rel.parts)
        ):
            raise ValueError(f"unsafe artifact path: {path!r}")
        return rel

    @classmethod
    def _target(cls, root: Path, path: str) -> Path:
        relative = cls._relative(path)
        if any((root / parent).is_symlink() for parent in relative.parents):
            raise ValueError(f"artifact path traverses a symlink: {path!r}")
        return root / relative

    @staticmethod
    def _remove(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    def _snapshot(
        self,
        base: ArtifactRevision | None,
        change: Callable[[Path], None],
        message: str,
    ) -> ArtifactRevision:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tree = root / "tree"
            tree.mkdir()
            if base is not None:
                with tarfile.open(fileobj=io.BytesIO(self.archive(base))) as archive:
                    archive.extractall(tree, filter="data")
            change(tree)
            env = {**os.environ, **_IDENTITY, "GIT_INDEX_FILE": str(root / "index")}

            def run(*args: str) -> str:
                return subprocess.run(
                    [
                        "git",
                        "--git-dir",
                        str(self.path / ".git"),
                        "--work-tree",
                        str(tree),
                        *args,
                    ],
                    cwd=tree,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()

            run("read-tree", base if base else "--empty")
            run("add", "--all", "--force", "--", ".")
            tree_sha = run("write-tree")
            if base and tree_sha == git(self.path, "rev-parse", f"{base}^{{tree}}"):
                return base
            revision = run(
                "commit-tree", tree_sha, *(["-p", base] if base else []), "-m", message
            )
            git(self.path, "update-ref", f"{KEEP}/{revision}", revision)
            return revision

    def write(
        self,
        *,
        base: ArtifactRevision | None,
        files: dict[str, str | bytes | None],
        message: str = "artifacts",
    ) -> ArtifactRevision:
        """Write or delete (None) paths. Identical inputs produce the same revision."""

        def change(tree: Path) -> None:
            for path, value in files.items():
                target = self._target(tree, path)
                self._remove(target)
                if value is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(
                        value.encode() if isinstance(value, str) else value
                    )

        return self._snapshot(base, change, message)

    def capture(
        self,
        *,
        base: ArtifactRevision,
        archive: bytes,
        prefix: str,
        only: tuple[str, ...],
        message: str = "capture",
    ) -> ArtifactRevision:
        """Replace owned paths from a runtime tar archive, preserving modes and safe symlinks."""

        def change(tree: Path) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                    tar.extractall(tmp, filter="data")
                source = Path(tmp) / self._relative(prefix)
                for link in source.rglob("*"):
                    if link.is_symlink() and not link.resolve().is_relative_to(source):
                        raise ValueError(
                            f"artifact symlink escapes its root: {link.relative_to(source)}"
                        )
                for path in only:
                    target, item = self._target(tree, path), self._target(source, path)
                    self._remove(target)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if item.is_symlink():
                        target.symlink_to(os.readlink(item))
                    elif item.is_dir():
                        shutil.copytree(
                            item,
                            target,
                            symlinks=True,
                            ignore=shutil.ignore_patterns(".git"),
                        )
                    elif item.exists():
                        shutil.copy2(item, target)

        return self._snapshot(base, change, message)
