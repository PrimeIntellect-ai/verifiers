"""Sandbox state as git commits: bundle out of the box, host pushes to the run remote.

Adapted from qx/px: one bare remote per run, one write-once ref per node instance
(lease on absence, so at-least-once execution never moves a published tip), and a
three-field handle verified on consume. Git inside the box runs under an isolated
environment; the box never holds a credential because the host does the push.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from verifiers.v1.runtimes import Runtime

BOX_HOME = "/tmp/flow-git"
BOX_OUT = "/tmp/flow-out.bundle"
BOX_IN = "/tmp/flow-in.bundle"
GIT = (
    f"git -c core.hooksPath={BOX_HOME}/hooks -c credential.helper= "
    "-c user.name=flow -c user.email=flow@run -c commit.gpgsign=false"
)
BOX_ENV = {"HOME": BOX_HOME, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}


class SnapshotError(Exception):
    pass


class SnapshotRef(BaseModel):
    kind: Literal["git"] = "git"
    remote: str
    ref: str
    head_sha: str
    base_sha: str | None = None
    """A commit the consumer already holds; bundles carry `base..head`."""


class GitBus:
    def __init__(self, remote: Path) -> None:
        self.remote = remote
        if not (remote / "HEAD").exists():
            remote.parent.mkdir(parents=True, exist_ok=True)
            _host(["git", "init", "-q", "--bare", str(remote)])

    async def snapshot(
        self, runtime: Runtime, *, workdir: str, ref: str, base_sha: str | None = None
    ) -> SnapshotRef:
        """Commit the box's `workdir`, ship the delta since `base_sha` (or everything),
        and publish it under `ref`, which must not exist yet."""
        head = (
            await _box(
                runtime,
                f"{_enter(workdir)} && {GIT} add -A && (({GIT} rev-parse -q --verify HEAD >/dev/null && {GIT} diff --cached --quiet) || {GIT} commit -q --allow-empty -m 'flow snapshot') && {GIT} rev-parse HEAD",
            )
        ).strip()
        base = (
            base_sha
            if base_sha and await asyncio.to_thread(self._has, base_sha)
            else None
        )
        if head != base:
            span = f"{base}..HEAD" if base else "HEAD"
            await _box(
                runtime,
                f"cd {shlex.quote(workdir)} && {GIT} bundle create -q {BOX_OUT} {span}",
            )
            bundle = await runtime.read(BOX_OUT)
            await asyncio.to_thread(self._pull, bundle, head)
        await asyncio.to_thread(self._publish, ref, head)
        return SnapshotRef(
            remote=str(self.remote), ref=ref, head_sha=head, base_sha=base
        )

    async def restore(
        self, runtime: Runtime, snap: SnapshotRef, *, workdir: str
    ) -> None:
        """Check out `snap` in the box's `workdir`, refusing a ref that has moved."""
        tip = await asyncio.to_thread(self._tip, snap.ref)
        if tip != snap.head_sha:
            raise SnapshotError(
                f"source_moved: {snap.ref} is at {tip}, not {snap.head_sha}"
            )
        has_base = (
            bool(snap.base_sha)
            and (
                await runtime.run(
                    [
                        "sh",
                        "-c",
                        f"cd {shlex.quote(workdir)} 2>/dev/null && {GIT} cat-file -e {snap.base_sha}^{{commit}}",
                    ],
                    BOX_ENV,
                )
            ).exit_code
            == 0
        )
        span = f"{snap.base_sha}..{snap.ref}" if has_base else snap.ref
        bundle = await asyncio.to_thread(self._bundle, span)
        await runtime.write(BOX_IN, bundle)
        await _box(
            runtime,
            f"{_enter(workdir)} && {GIT} fetch -q {BOX_IN} {snap.ref} && {GIT} checkout -q --detach {snap.head_sha}",
        )

    def _has(self, sha: str) -> bool:
        return (
            _host(
                ["git", "-C", str(self.remote), "cat-file", "-e", f"{sha}^{{commit}}"],
                check=False,
            ).returncode
            == 0
        )

    def _tip(self, ref: str) -> str | None:
        p = _host(
            ["git", "-C", str(self.remote), "rev-parse", "--verify", "-q", ref],
            check=False,
        )
        return p.stdout.strip() or None

    def _pull(self, bundle: bytes, head: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as f:
            f.write(bundle)
        try:
            _host(["git", "-C", str(self.remote), "fetch", "-q", f.name, "HEAD"])
        finally:
            os.unlink(f.name)
        if not self._has(head):
            raise SnapshotError(f"bundle did not deliver {head}")

    def _publish(self, ref: str, head: str) -> None:
        # An empty old value means "the ref must not exist": git's lease on absence.
        p = _host(
            ["git", "-C", str(self.remote), "update-ref", ref, head, ""], check=False
        )
        if p.returncode != 0:
            raise SnapshotError(f"push_raced: {ref} already exists: {p.stderr.strip()}")

    def _bundle(self, span: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as f:
            pass
        try:
            _host(
                ["git", "-C", str(self.remote), "bundle", "create", "-q", f.name, span]
            )
            return Path(f.name).read_bytes()
        finally:
            os.unlink(f.name)


def _enter(workdir: str) -> str:
    w = shlex.quote(workdir)
    return f"mkdir -p {BOX_HOME}/hooks {w} && cd {w} && ({GIT} rev-parse --git-dir >/dev/null 2>&1 || {GIT} init -q)"


async def _box(runtime: Runtime, script: str) -> str:
    result = await runtime.run(["sh", "-c", script], BOX_ENV)
    if result.exit_code != 0:
        raise SnapshotError(
            f"git in box failed ({result.exit_code}): {result.stderr.strip()[-500:]}"
        )
    return result.stdout


def _host(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}
    p = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    if check and p.returncode != 0:
        raise SnapshotError(f"{' '.join(argv[:3])} failed: {p.stderr.strip()[-500:]}")
    return p
