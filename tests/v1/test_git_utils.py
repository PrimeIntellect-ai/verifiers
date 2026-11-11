"""Filename handling in `capture_patch` / `snapshot_untracked` (#2196), driven through
real `git` in a temp repo.

An ignore entry is an untracked filename taken from the image, so it must reach `git
reset` as a literal path and must survive the sandbox's text decoding. Both halves are
exercised here against real git and a byte-accurate local runtime, because the failure
they guard against is silent: a glob entry empties the captured patch, and a lossily
decoded name stops matching its file, so the image file is credited to the agent.
"""

import asyncio
import base64
import os
import subprocess
from pathlib import Path

import pytest

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.utils.git import capture_patch, resolve_head, snapshot_untracked

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "vf",
    "GIT_AUTHOR_EMAIL": "vf@example.com",
    "GIT_COMMITTER_NAME": "vf",
    "GIT_COMMITTER_EMAIL": "vf@example.com",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}


class LocalRuntime:
    """The two Runtime methods these helpers use, over a real local directory: `run`
    returns byte-accurate stdout, `read`/`write` move bytes."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.workdir,
            env={**GIT_ENV, **env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return ProgramResult(
            exit_code=proc.returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        return Path(path).read_bytes()

    async def write(self, path: str, data: bytes) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


class Trace:
    """Duck-typed stand-in for `verifiers.v1.trace.Trace`; only `.info` is used."""

    def __init__(self) -> None:
        self.info: dict = {}


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env=GIT_ENV,
        check=False,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "base.py").write_text("base = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    return repo


@pytest.mark.asyncio
async def test_capture_patch_keeps_agent_work_when_a_shipped_name_is_a_glob(
    repo: Path,
):
    """`*.py` is a file the image shipped, not a pattern: without `:(literal)` the
    unstage step removes every python file the agent touched and the patch is empty."""
    (repo / "*.py").write_text("shipped by the image\n")
    runtime = LocalRuntime(repo)
    base = await resolve_head(runtime)
    ignore = await snapshot_untracked(runtime)
    assert ignore == ["*.py"]

    (repo / "base.py").write_text("base = 1\nagent = 2\n")
    (repo / "agent_change.py").write_text("agent = 2\n")

    trace = Trace()
    await capture_patch(trace, runtime, base_commit=base, ignore=ignore)

    patch = trace.info["patch"]
    assert "agent_change.py" in patch
    assert "agent = 2" in patch
    assert "*.py" not in patch  # the image file stays out


@pytest.mark.asyncio
async def test_capture_patch_treats_pathspec_magic_as_a_filename(repo: Path):
    """A leading `:` is pathspec magic, so a name like `:(glob)*` cannot be passed
    through unchanged."""
    (repo / ":(glob)*").write_text("shipped by the image\n")
    runtime = LocalRuntime(repo)
    ignore = await snapshot_untracked(runtime)
    assert ignore == [":(glob)*"]

    (repo / "base.py").write_text("base = 1\nagent = 2\n")
    trace = Trace()
    await capture_patch(trace, runtime, ignore=ignore)

    assert "agent = 2" in trace.info["patch"]
    assert ":(glob)*" not in trace.info["patch"]


@pytest.mark.asyncio
async def test_snapshot_untracked_preserves_a_legitimate_replacement_char(repo: Path):
    """U+FFFD is a valid character in a filename. Rejecting it outright would break a
    repo that has one, so the name must round-trip unchanged."""
    name = "caf\ufffd.bin"
    (repo / name).write_text("shipped by the image\n")
    assert await snapshot_untracked(LocalRuntime(repo)) == [name]


@pytest.mark.asyncio
async def test_snapshot_untracked_fails_loudly_on_non_utf8(repo: Path):
    """A name that is not valid UTF-8 cannot round-trip through the runtime's text
    decoding, so it must fail before a wrong patch is captured."""

    class NonUtf8Runtime:
        async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
            # What the sandbox command emits for a latin-1 filename: base64 of the raw
            # `git ls-files -z` bytes.
            payload = base64.b64encode(b"caf\xe9.bin\0")
            return ProgramResult(exit_code=0, stdout=payload.decode("ascii"), stderr="")

    with pytest.raises(SandboxError, match="not valid UTF-8"):
        await snapshot_untracked(NonUtf8Runtime())
