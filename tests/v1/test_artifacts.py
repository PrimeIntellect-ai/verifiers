"""Artifact transport bounds and restore validation.

`collect` moves tar archives from an agent's runtime to the host and `restore`
extracts them inside a grading runtime — a trust boundary, since the agent
controls both the archived files and the tooling that creates them. These tests
exercise the boundary without a sandbox: `_validate_restore` is pure, and a
scripted fake runtime covers `collect`/`restore` end to end.
"""

import io
import shlex
import tarfile
from types import SimpleNamespace

import pytest

from verifiers.v1.errors import SandboxError
from verifiers.v1.utils.artifacts import (
    ARTIFACTS_DIR,
    MAX_ARTIFACT_BYTES,
    Artifact,
    _validate_restore,
    collect,
    restore,
)

ROOT = "/work"


def _tar(*members: tuple[tarfile.TarInfo, bytes | None]) -> bytes:
    """An uncompressed tar archive; each member is (info, payload-or-None)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tar:
        for info, payload in members:
            if payload is not None:
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            else:
                tar.addfile(info)
    return buf.getvalue()


def _file(name: str, payload: bytes = b"x") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    return info, payload


def _dir(name: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    return info, None


def _symlink(name: str, target: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info, None


def _hardlink(name: str, target: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    return info, None


def _fifo(name: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.FIFOTYPE
    return info, None


class _FakeRuntime:
    """The `Runtime` surface `collect`/`restore` use: `test -e` answers from a
    set, every `sh -c` command is recorded, `read` enforces its byte cap inside
    the box (the contract that makes archive size ungameable), and `write`
    stores payloads."""

    def __init__(
        self,
        *,
        type: str = "docker",
        workdir: str = "/work",
        exists: set[str] | frozenset[str] = frozenset(),
        payloads: list[bytes] | None = None,
    ) -> None:
        self.config = SimpleNamespace(type=type, workdir=workdir)
        self.exists = set(exists)
        self.payloads = list(payloads or [])
        self.commands: list[str] = []
        self.reads: list[tuple[str, int | None]] = []
        self.writes: dict[str, bytes] = {}

    async def run(self, argv, env):
        if argv[0] != "sh":
            return SimpleNamespace(exit_code=0, stdout="", stderr="")
        cmd = argv[2]
        self.commands.append(cmd)
        if cmd.startswith("for source in "):
            sources = shlex.split(
                cmd.removeprefix("for source in ").split("; do", 1)[0]
            )
            out = "".join(
                "1\n" if source in self.exists else "0\n" for source in sources
            )
            return SimpleNamespace(exit_code=0, stdout=out, stderr="")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def read(self, path, max_bytes=None):
        self.reads.append((path, max_bytes))
        data = self.payloads.pop(0)
        if max_bytes is not None and len(data) > max_bytes:
            raise SandboxError(f"{path} exceeds the {max_bytes}-byte cap")
        return data

    async def write(self, path, data):
        self.writes[path] = data


# --- _validate_restore ------------------------------------------------------


def test_validate_restore_accepts_a_valid_tree():
    archive = _tar(
        _dir("work"),
        _file("work/out/answer.txt", b"42"),
        _symlink("work/out/latest", "answer.txt"),
        _file("work/deep/nested/build.o", b"o"),
        _hardlink("work/deep/nested/dup.o", "work/deep/nested/build.o"),
    )
    _validate_restore(ROOT, archive)


def test_validate_restore_accepts_none_and_empty_archives():
    _validate_restore(ROOT, None)  # a missing optional artifact, not an archive
    _validate_restore(ROOT, _tar())


def test_validate_restore_rejects_parent_traversal():
    with pytest.raises(RuntimeError, match="outside declared root"):
        _validate_restore(ROOT, _tar(_file("work/../etc/passwd")))


def test_validate_restore_rejects_absolute_member_names():
    with pytest.raises(RuntimeError, match="outside declared root"):
        _validate_restore(ROOT, _tar(_file("/work/a")))


def test_validate_restore_rejects_members_outside_the_root():
    with pytest.raises(RuntimeError, match="outside declared root"):
        _validate_restore(ROOT, _tar(_file("etc/passwd")))
    with pytest.raises(RuntimeError, match="outside declared root"):
        _validate_restore(ROOT, _tar(_file("workfoo/x")))


def test_validate_restore_rejects_escaping_symlinks():
    for target in ("../etc/passwd", "/etc/passwd", "../../x"):
        with pytest.raises(RuntimeError, match="unsafe target"):
            _validate_restore(ROOT, _tar(_symlink("work/link", target)))


def test_validate_restore_rejects_extraction_through_a_symlink():
    archive = _tar(
        _symlink("work/a", "sub"),
        _file("work/a/b"),
    )
    with pytest.raises(RuntimeError, match="traverses a symlink"):
        _validate_restore(ROOT, archive)


def test_validate_restore_rejects_unsafe_hardlinks():
    # A forward reference: the target is not an earlier regular member.
    with pytest.raises(RuntimeError, match="unsafe target"):
        _validate_restore(ROOT, _tar(_hardlink("work/b", "work/a")))
    # Absolute and parent-traversing targets.
    for target in ("/etc/passwd", "work/../etc/passwd"):
        with pytest.raises(RuntimeError, match="unsafe target"):
            _validate_restore(ROOT, _tar(_file("work/a"), _hardlink("work/b", target)))
    # A self-link.
    with pytest.raises(RuntimeError, match="unsafe target"):
        _validate_restore(ROOT, _tar(_file("work/a"), _hardlink("work/b", "work/b")))


def test_validate_restore_rejects_special_files():
    with pytest.raises(RuntimeError, match="link or special file"):
        _validate_restore(ROOT, _tar(_fifo("work/pipe")))


def test_validate_restore_rejects_duplicate_members():
    archive = _tar(_file("work/a", b"1"), _file("work/a", b"2"))
    with pytest.raises(RuntimeError, match="duplicate"):
        _validate_restore(ROOT, archive)


@pytest.mark.parametrize("root", ["work", "..", "/", "a/../b"])
def test_validate_restore_rejects_unsafe_roots(root):
    with pytest.raises(RuntimeError, match="must be an absolute path"):
        _validate_restore(root, _tar(_file("work/a")))


def test_validate_restore_rejects_corrupt_archives():
    with pytest.raises(RuntimeError, match="unreadable artifact archive"):
        _validate_restore(ROOT, b"not a tar at all")


# --- collect -----------------------------------------------------------------


async def test_collect_sweeps_the_convention_dir_as_optional():
    rt = _FakeRuntime()
    assert await collect(rt, []) == {ARTIFACTS_DIR: None}


async def test_collect_archives_declared_paths_and_sweeps():
    conv = _tar(_file("logs/artifacts/answer.txt", b"42"))
    decl = _tar(_file("app/out.bin", b"data"))
    rt = _FakeRuntime(exists={ARTIFACTS_DIR, "/app/out.bin"}, payloads=[conv, decl])
    got = await collect(rt, [Artifact(source="/app/out.bin")])
    assert got == {ARTIFACTS_DIR: conv, "/app/out.bin": decl}
    # Every transfer carries the remaining budget into the runtime: the first
    # read gets the whole cap, the second only what is left.
    assert rt.reads[0][1] == MAX_ARTIFACT_BYTES
    assert rt.reads[1][1] == MAX_ARTIFACT_BYTES - len(conv)


async def test_collect_fails_a_missing_required_source():
    rt = _FakeRuntime()
    with pytest.raises(RuntimeError, match="does not exist"):
        await collect(rt, [Artifact(source="/must/exist")])


async def test_collect_keeps_a_missing_optional_source():
    rt = _FakeRuntime(exists={ARTIFACTS_DIR}, payloads=[b"conv"])
    got = await collect(rt, [Artifact(source="/opt/x", required=False)])
    assert got["/opt/x"] is None
    assert got[ARTIFACTS_DIR] == b"conv"


async def test_collect_rejects_duplicate_sources():
    rt = _FakeRuntime()
    with pytest.raises(RuntimeError, match="declared more than once"):
        await collect(
            rt,
            [Artifact(source="/a"), Artifact(source="/a")],
            sweep=False,
        )


async def test_collect_resolves_relative_sources_against_the_workdir():
    rt = _FakeRuntime(workdir="/repo", exists={"/repo/out"}, payloads=[b"r"])
    got = await collect(rt, [Artifact(source="out")], sweep=False)
    assert got == {"/repo/out": b"r"}


async def test_collect_excludes_declared_paths_from_the_convention_sweep():
    rt = _FakeRuntime(
        exists={ARTIFACTS_DIR, f"{ARTIFACTS_DIR}/keep"},
        payloads=[b"conv", b"decl"],
    )
    await collect(rt, [Artifact(source=f"{ARTIFACTS_DIR}/keep")])
    tar_cmds = [c for c in rt.commands if "tar -cf" in c]
    assert any("--exclude=logs/artifacts/keep" in c for c in tar_cmds)


async def test_collect_bounds_the_transfer_inside_the_runtime():
    """An archive replaced or grown after its tar step cannot bypass the budget:
    the cap rides on `runtime.read` itself, not on a size probe."""
    rt = _FakeRuntime(exists={"/big"}, payloads=[b"x" * 64])
    with pytest.raises(SandboxError):
        await collect(rt, [Artifact(source="/big")], max_bytes=8, sweep=False)


# --- restore ------------------------------------------------------------------


async def test_restore_refuses_the_subprocess_runtime():
    rt = _FakeRuntime(type="subprocess")
    with pytest.raises(RuntimeError, match="subprocess"):
        await restore(rt, {"/work": _tar(_file("work/a"))})
    assert rt.commands == []


async def test_restore_validates_before_mutating_the_runtime():
    """Every archive is checked on the host before a single command runs in the
    grading box — a hostile member must not ride along with a root clear."""
    evil = _tar(_file("work/../etc/passwd"))
    rt = _FakeRuntime()
    with pytest.raises(RuntimeError, match="outside declared root"):
        await restore(rt, {"/work": evil})
    assert rt.commands == []
    assert rt.writes == {}


async def test_restore_clears_roots_then_writes_and_extracts():
    archive = _tar(_file("work/out/x"))
    rt = _FakeRuntime()
    await restore(rt, {"/work": archive})
    assert next(iter(rt.writes.values())) == archive
    assert any("rm -rf" in c and "/work" in c for c in rt.commands)
    assert any("tar -xf" in c for c in rt.commands)


async def test_restore_skips_none_archives():
    rt = _FakeRuntime()
    await restore(rt, {"/work": None})
    assert any("rm -rf" in c for c in rt.commands)  # roots are still cleared
    assert rt.writes == {}
    assert not any("tar -xf" in c for c in rt.commands)


async def test_restore_with_nothing_is_a_noop():
    rt = _FakeRuntime()
    await restore(rt, {})
    assert rt.commands == []
