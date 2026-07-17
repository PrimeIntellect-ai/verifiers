"""`capture_patch` / `resolve_head` against real git repos via a local runtime stub."""

import asyncio
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace


from verifiers.v1.utils import git as patch_capture
from verifiers.v1.utils.git import capture_patch, resolve_head


@dataclass
class LocalGitRuntime:
    """Duck-typed `Runtime` running argv on the host inside a repo dir.

    `/tmp/...` scratch paths the helper uses are remapped under the repo's own
    tmp dir so parallel tests can't collide.
    """

    cwd: Path
    tmp: Path = field(init=False)

    def __post_init__(self):
        self.tmp = self.cwd / ".vf-tmp"
        self.tmp.mkdir(exist_ok=True)

    def _remap(self, s: str) -> str:
        return s.replace("/tmp/vf_agent_patch", str(self.tmp / "vf_agent_patch"))

    async def run(self, argv: list[str], env: dict) -> SimpleNamespace:
        argv = [self._remap(a) for a in argv]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.cwd,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(self.cwd), **env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return SimpleNamespace(
            exit_code=proc.returncode, stdout=out.decode(), stderr=err.decode()
        )

    async def read(self, path: str) -> bytes:
        return Path(self._remap(path)).read_bytes()


def make_repo(tmp_path: Path) -> tuple[LocalGitRuntime, str]:
    """A repo with one committed file; returns (runtime, base sha)."""
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "src.py").write_text("original\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "base"], cwd=tmp_path, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return LocalGitRuntime(tmp_path), sha


def trace_stub() -> SimpleNamespace:
    return SimpleNamespace(info={})


async def test_captures_worktree_untracked_and_committed_changes(tmp_path):
    runtime, base = make_repo(tmp_path)
    (tmp_path / "src.py").write_text("edited\n")
    (tmp_path / "new.py").write_text("added\n")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "add", "new.py"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "agent"], cwd=tmp_path, check=True)
    (tmp_path / "uncommitted.py").write_text("later\n")

    trace = trace_stub()
    await capture_patch(trace, runtime, base_commit=base)

    patch = trace.info["patch"]
    assert "patch_error" not in trace.info
    assert "edited" in patch  # unstaged edit
    assert "new.py" in patch  # committed by the agent, still diffed vs base
    assert "uncommitted.py" in patch  # untracked file staged by add -A

    # the index is left unstaged afterwards
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert staged == ""


async def test_head_fallback_misses_agent_commits(tmp_path):
    runtime, _ = make_repo(tmp_path)
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    (tmp_path / "src.py").write_text("committed edit\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "agent"], cwd=tmp_path, check=True)

    trace = trace_stub()
    await capture_patch(trace, runtime)  # no base -> HEAD

    assert trace.info["patch"] == ""  # documents why callers should pass a base


async def test_add_failure_records_patch_error(tmp_path):
    runtime, base = make_repo(tmp_path)
    (tmp_path / "src.py").write_text("edited\n")
    (tmp_path / ".git" / "index.lock").touch()

    trace = trace_stub()
    await capture_patch(trace, runtime, base_commit=base)

    assert "patch" not in trace.info
    assert "exit=" in trace.info["patch_error"]


async def test_reset_failure_records_patch_error(tmp_path):
    # A git shim that fails only `reset` (e.g. ENOSPC rewriting .git/index): add and
    # diff succeed, but a staged tree must not be reported as a successful capture.
    runtime, base = make_repo(tmp_path)
    (tmp_path / "src.py").write_text("edited\n")
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "git").write_text(
        '#!/bin/sh\n[ "$1" = reset ] && exit 7\nexec /usr/bin/git "$@"\n'
    )
    (shim / "git").chmod(0o755)

    trace = trace_stub()
    await capture_patch(
        trace,
        runtime,
        base_commit=base,
        env={"PATH": f"{shim}:/usr/local/bin:/usr/bin:/bin"},
    )

    assert "patch" not in trace.info
    assert "exit=7" in trace.info["patch_error"]


async def test_bad_base_records_patch_error(tmp_path):
    runtime, _ = make_repo(tmp_path)
    trace = trace_stub()
    await capture_patch(trace, runtime, base_commit="not-a-ref")
    assert "patch" not in trace.info
    assert trace.info["patch_error"].startswith("exit=")


async def test_oversized_patch_truncates(tmp_path, monkeypatch):
    runtime, base = make_repo(tmp_path)
    (tmp_path / "big.txt").write_text("x" * 4096)
    monkeypatch.setattr(patch_capture, "PATCH_CAP_BYTES", 100)

    trace = trace_stub()
    await capture_patch(trace, runtime, base_commit=base)

    assert trace.info["patch_truncated"] is True
    assert len(trace.info["patch"].encode()) == 100


async def test_resolve_head(tmp_path):
    runtime, sha = make_repo(tmp_path)
    assert await resolve_head(runtime) == sha


async def test_resolve_head_outside_repo(tmp_path):
    bare = tmp_path / "empty"
    bare.mkdir()
    assert await resolve_head(LocalGitRuntime(bare)) == ""
