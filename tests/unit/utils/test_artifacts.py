"""Artifact restore: the grading box extracts only what host-side validation read."""

import asyncio
import io
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from verifiers.v1.utils.artifacts import restore


class RecordingRuntime:
    """Records the archives `restore` sends and accepts every command."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(type="docker", mounts={}, workdir="/app")
        self.written: list[bytes] = []

    async def write(self, path: str, data: bytes) -> None:
        self.written.append(data)

    async def run(self, argv, env):
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


def make_archive(build) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:", format=tarfile.GNU_FORMAT) as tar:
        build(tar)
    return buffer.getvalue()


def add_file(tar: tarfile.TarFile, name: str, data: bytes, **fields) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    for key, value in fields.items():
        setattr(info, key, value)
    tar.addfile(info, io.BytesIO(data))


def restored(archive: bytes, root: str = "/app/results") -> bytes:
    runtime = RecordingRuntime()
    asyncio.run(restore(runtime, {root: archive}))
    assert len(runtime.written) == 1
    return runtime.written[0]


def test_restore_keeps_owners_and_drops_special_mode_bits() -> None:
    def build(tar):
        directory = tarfile.TarInfo("app/results")
        directory.type, directory.mode, directory.uid = tarfile.DIRTYPE, 0o1777, 1000
        tar.addfile(directory)
        add_file(
            tar, "app/results/run", b"#!/bin/sh\n", mode=0o4755, uid=1000, gid=1000
        )

    with tarfile.open(fileobj=io.BytesIO(restored(make_archive(build)))) as tar:
        members = {m.name: m for m in tar.getmembers()}

    assert members["app/results"].mode == 0o777
    assert members["app/results/run"].mode == 0o755
    assert (members["app/results/run"].uid, members["app/results/run"].gid) == (
        1000,
        1000,
    )


@pytest.mark.skipif(shutil.which("tar") is None, reason="needs a system tar")
def test_system_tar_lists_exactly_what_python_read() -> None:
    long_name = "app/results/" + "nested/" * 20 + "phase_map.csv"

    def build(tar):
        add_file(tar, "app/results/a.txt", b"a")
        add_file(tar, long_name, b"b")
        link = tarfile.TarInfo("app/results/link")
        link.type, link.linkname = tarfile.SYMTYPE, "a.txt"
        tar.addfile(link)
        hard = tarfile.TarInfo("app/results/hard")
        hard.type, hard.linkname = tarfile.LNKTYPE, "app/results/a.txt"
        tar.addfile(hard)

    original = make_archive(build)
    with tarfile.open(fileobj=io.BytesIO(original)) as tar:
        python_names = [m.name for m in tar.getmembers()]

    listing = subprocess.run(
        ["tar", "-tf", "-"],
        input=restored(original),
        capture_output=True,
        check=True,
    ).stdout.decode()

    assert listing.splitlines() == python_names
