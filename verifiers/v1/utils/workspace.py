"""Bounded filesystem copies for independent container judge attempts."""

import glob
import io
import shlex
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

from verifiers.v1.runtimes import Runtime
from verifiers.v1.utils.artifacts import (
    Artifact,
    _run,
    _tar_out,
    _validate_restore,
    validate_runtime_mounts,
)

# A workspace includes installed code and verifier staging, independently of the
# smaller published-submission limit. Transfers remain bounded on the runtime.
MAX_WORKSPACE_BYTES = 512 * 1024 * 1024


@dataclass
class WorkspaceSnapshot:
    archives: dict[str, bytes | None]
    mounts: tuple[str, ...]

    @classmethod
    async def capture(cls, runtime: Runtime, roots: set[PurePosixPath]):
        mounts = tuple(getattr(runtime.config, "mounts", {}))
        await validate_runtime_mounts(runtime, map(str, roots), workspace=True)
        archives = {}
        budget = MAX_WORKSPACE_BYTES
        for root in sorted(roots):
            if (
                root == PurePosixPath("/")
                or ".." in root.parts
                or not root.is_absolute()
            ):
                raise ValueError("Workspace roots must be absolute paths below '/'")
            if any(root.is_relative_to(mount) for mount in mounts):
                continue
            if any(parent in roots for parent in root.parents):
                continue
            exists = await runtime.run(
                [
                    "sh",
                    "-c",
                    f"test -e {shlex.quote(str(root))} || test -L {shlex.quote(str(root))}",
                ],
                {},
            )
            archive = None
            if not exists.exit_code:
                archive = await _tar_out(
                    runtime,
                    Artifact(
                        source=str(root),
                        exclude=[
                            m.lstrip("/")
                            for m in mounts
                            if PurePosixPath(m).is_relative_to(root)
                        ],
                    ),
                    budget,
                    literal_excludes=True,
                )
                budget -= len(archive)
                _validate_restore(str(root), archive, workspace=True)
            archives[str(root)] = archive
        return cls(archives, mounts)

    async def restore(self, runtime: Runtime) -> None:
        if runtime.type == "subprocess":
            raise ValueError("Workspace copies require a container runtime")
        if set(getattr(runtime.config, "mounts", {})) != set(self.mounts):
            raise ValueError("Workspace copies require the same read-only mounts")
        await validate_runtime_mounts(runtime, self.archives, workspace=True)
        cleanup = []
        for root, archive in self.archives.items():
            _validate_restore(root, archive, workspace=True)
            mounts = [
                PurePosixPath(m)
                for m in self.mounts
                if PurePosixPath(m).is_relative_to(root)
            ]
            if archive is not None:
                with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                    for member in tar:
                        path = PurePosixPath("/") / member.name
                        if any(
                            path.is_relative_to(m)
                            or (m.is_relative_to(path) and not member.isdir())
                            for m in mounts
                        ):
                            raise ValueError(
                                "Workspace archive overlaps a read-only mount"
                            )
            if not mounts:
                cleanup.append(f"rm -rf -- {shlex.quote(root)}")
                continue
            # Keep mount targets and their ancestors; remove every other subtree
            # without descending into it. Fresh runtimes reattach the same mounts.
            preserve = {
                p for m in mounts for p in (m, *m.parents) if p.is_relative_to(root)
            }
            patterns = {
                p: shlex.quote(glob.escape(str(p).replace("\\", "\\\\")))
                for p in preserve
            }
            prune = " -o ".join(f"-path {patterns[m]}" for m in mounts)
            keep = " ".join(f"! -path {patterns[p]}" for p in sorted(preserve))
            # BusyBox find can leave batched -exec unflushed with -prune;
            # remove each subtree immediately so stale image files cannot survive.
            cleanup.append(
                f"find {shlex.quote(root)} -mindepth 1 \\( {prune} \\) -prune -o "
                f"\\( {keep} -prune -exec rm -rf -- {{}} \\; \\)"
            )
        workdir = shlex.quote(runtime.config.workdir or "/app")
        await _run(
            runtime,
            " && ".join([*cleanup, f"mkdir -p -- {workdir}"]),
            "clear workspace roots",
        )
        for archive in self.archives.values():
            if archive is None:
                continue
            path = f"/tmp/vf-workspace-{uuid.uuid4().hex}.tar"
            await runtime.write(path, archive)
            await _run(
                runtime, f"tar -xf {path} -C / && rm -f {path}", "restore workspace"
            )
