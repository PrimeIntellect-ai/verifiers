"""Prime file transfer: large files cross in parts, failed-in-transit requests are tried
again, and nothing else is. The fake SDK client maps the box onto a local directory and
the runtime's commands run in a local shell, so the join and cut scripts run for real."""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
from prime_sandboxes.exceptions import (
    DownloadTimeoutError,
    SandboxNotRunningError,
)
from prime_sandboxes.models import CommandResponse

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import prime
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime

PART = 10


class LocalBoxClient:
    """The SDK calls the runtime's transfers make, against local files. `failures`
    maps a method name to the errors its next calls raise, in order."""

    def __init__(self, failures=None):
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.uploads: list[tuple[str, int, int | None]] = []
        self.downloads: list[tuple[str, int | None]] = []
        self.commands: list[str] = []

    def _fail(self, method):
        pending = self.failures.get(method)
        if pending:
            raise pending.pop(0)

    async def upload_bytes(
        self, sandbox_id, file_path, file_bytes, filename, timeout=None
    ):
        self._fail("upload_bytes")
        self.uploads.append((file_path, len(file_bytes), timeout))
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_bytes(file_bytes)

    async def download_file(self, sandbox_id, file_path, local_file_path, timeout=None):
        self._fail("download_file")
        self.downloads.append((file_path, timeout))
        shutil.copyfile(file_path, local_file_path)

    async def execute_command(self, sandbox_id, command, timeout=None, user=None):
        self._fail("execute_command")
        self.commands.append(command)
        done = await asyncio.to_thread(
            subprocess.run,
            ["sh", "-c", command],
            capture_output=True,
            text=True,
            check=False,
        )
        return CommandResponse(
            stdout=done.stdout, stderr=done.stderr, exit_code=done.returncode
        )


@pytest.fixture
def runtime(monkeypatch) -> PrimeRuntime:
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime, "_TRANSFER_PART_BYTES", PART)
    runs: list[list[str]] = []

    async def run(self, argv, env):
        runs.append(argv)
        done = await asyncio.to_thread(
            subprocess.run, argv, capture_output=True, text=True, check=False
        )
        return ProgramResult(done.returncode, done.stdout, done.stderr)

    monkeypatch.setattr(PrimeRuntime, "run", run)
    box = PrimeRuntime(PrimeConfig(transfer_timeout=42))
    box.info.id = "sb1"
    box._client = LocalBoxClient()
    box.runs = runs  # type: ignore[attr-defined]
    return box


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    waited: list[float] = []

    async def sleep(delay):
        waited.append(delay)

    monkeypatch.setattr(prime.asyncio, "sleep", sleep)
    return waited


DATA = bytes(range(256)) * 3 + b"tail"  # 772 bytes: 78 parts of 10, the last of 2


async def test_a_small_write_is_one_upload_with_the_configured_timeout(
    runtime, tmp_path
):
    target = tmp_path / "box" / "small.bin"
    await runtime.write(str(target), b"0123456789")
    assert target.read_bytes() == b"0123456789"
    assert runtime._client.uploads == [(str(target), 10, 42)]
    assert runtime.runs == []


async def test_a_large_write_crosses_in_parts_and_leaves_only_the_file(
    runtime, tmp_path
):
    target = tmp_path / "box" / "large.bin"
    await runtime.write(str(target), DATA)
    assert target.read_bytes() == DATA
    uploads = runtime._client.uploads
    assert len(uploads) == 78
    assert all(size <= PART and timeout == 42 for _, size, timeout in uploads)
    assert sorted(p.name for p in target.parent.iterdir()) == ["large.bin"]


async def test_a_large_write_joins_in_about_the_files_own_room(
    runtime, tmp_path, monkeypatch
):
    """The join removes each part once appended: the box never holds every part and the
    joined copy at once (twice the file), which fails a write over half the free disk."""
    shim, log = tmp_path / "bin", tmp_path / "held.log"
    shim.mkdir()
    (shim / "cat").write_text(  # `cat`, then log the bytes the parts and join hold
        "#!/bin/sh\n"
        f'{shutil.which("cat")} "$@" || exit\n'
        "total=0\n"
        f'for f in "{tmp_path}"/box/*.vf-part-*; do\n'
        '  [ -f "$f" ] && total=$((total + $(wc -c < "$f")))\n'
        "done\n"
        f'echo "$total" >> "{log}"\n'
    )
    (shim / "cat").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}:{os.environ['PATH']}")
    target = tmp_path / "box" / "large.bin"
    await runtime.write(str(target), DATA)
    assert target.read_bytes() == DATA
    held = [int(line) for line in log.read_text().split()]
    assert max(held) <= len(DATA) + PART, "at most one part beyond the file"


async def test_a_large_write_replaces_an_existing_file_whole(runtime, tmp_path):
    target = tmp_path / "large.bin"
    target.write_bytes(b"old content that is longer than nothing")
    await runtime.write(str(target), DATA)
    assert target.read_bytes() == DATA


async def test_a_failed_join_keeps_the_old_file_and_removes_the_parts(
    runtime, tmp_path, monkeypatch
):
    target = tmp_path / "large.bin"
    target.write_bytes(b"old")
    real_upload = LocalBoxClient.upload_bytes

    async def short_last_part(
        self, sandbox_id, file_path, file_bytes, filename, timeout=None
    ):
        if file_path.endswith(".00077"):
            file_bytes = file_bytes[:1]  # the box got less than was sent
        await real_upload(self, sandbox_id, file_path, file_bytes, filename, timeout)

    monkeypatch.setattr(LocalBoxClient, "upload_bytes", short_last_part)
    with pytest.raises(
        SandboxError, match="joining 78 uploaded parts failed"
    ) as caught:
        await runtime.write(str(target), DATA)
    assert caught.value.kind == "io"
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["large.bin"]


async def test_an_upload_dropped_in_transit_is_tried_again(runtime, tmp_path, sleeps):
    runtime._client.failures["upload_bytes"] = [
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.WriteTimeout("timed out"),
    ]
    target = tmp_path / "small.bin"
    await runtime.write(str(target), b"abc")
    assert target.read_bytes() == b"abc"
    # Exponential from 5 s, plus up to 1 s of jitter.
    assert [int(delay) for delay in sleeps] == [5, 10]


async def test_an_upload_failing_in_transit_every_try_is_an_io_error(
    runtime, tmp_path, sleeps
):
    runtime._client.failures["upload_bytes"] = [
        httpx.RemoteProtocolError("peer closed connection")
    ] * 3
    with pytest.raises(SandboxError) as caught:
        await runtime.write(str(tmp_path / "x"), b"abc")
    assert caught.value.kind == "io"
    assert len(sleeps) == 2


async def test_a_lost_box_is_not_tried_again(runtime, tmp_path, sleeps):
    runtime._client.failures["upload_bytes"] = [
        SandboxNotRunningError("sb1", status="TERMINATED")
    ]
    with pytest.raises(SandboxError) as caught:
        await runtime.write(str(tmp_path / "x"), b"abc")
    assert caught.value.kind == "box_lost"
    assert sleeps == []


async def test_a_small_read_is_one_download_with_the_configured_timeout(
    runtime, tmp_path
):
    target = tmp_path / "small.bin"
    target.write_bytes(b"0123456789")
    assert await runtime.read(str(target)) == b"0123456789"
    assert runtime._client.downloads == [(str(target), 42)]
    assert runtime.runs == []


async def test_a_large_read_crosses_in_parts_and_leaves_only_the_file(
    runtime, tmp_path
):
    target = tmp_path / "large.bin"
    target.write_bytes(DATA)
    assert await runtime.read(str(target)) == DATA
    downloads = runtime._client.downloads
    assert len(downloads) == 78 and all(path != str(target) for path, _ in downloads)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["large.bin"]


async def test_a_file_that_grows_while_read_gives_the_size_it_had(
    runtime, tmp_path, monkeypatch
):
    """A log appended to during a chunked read returns the bytes it had when the read
    began, instead of failing as changed."""
    target = tmp_path / "large.log"
    target.write_bytes(DATA)
    real_download = LocalBoxClient.download_file

    async def append_after_each(
        self, sandbox_id, file_path, local_file_path, timeout=None
    ):
        await real_download(self, sandbox_id, file_path, local_file_path, timeout)
        with target.open("ab") as log:
            log.write(b"more log lines\n")

    monkeypatch.setattr(LocalBoxClient, "download_file", append_after_each)
    assert await runtime.read(str(target)) == DATA
    parts = [path for path, _ in runtime._client.downloads]
    assert all(path.startswith("/tmp/vf-read-") for path in parts)
    assert not any(Path(path).exists() for path in parts)


async def test_reading_a_missing_file_fails_before_any_download(runtime, tmp_path):
    with pytest.raises(SandboxError) as caught:
        await runtime.read(str(tmp_path / "missing.txt"))
    assert caught.value.kind == "missing"
    assert "no such file" in str(caught.value).lower()
    assert runtime._client.downloads == []


async def test_a_download_timing_out_is_tried_again(runtime, tmp_path, sleeps):
    target = tmp_path / "small.bin"
    target.write_bytes(b"abc")
    runtime._client.failures["download_file"] = [
        DownloadTimeoutError("sb1", str(target), 42)
    ]
    assert await runtime.read(str(target)) == b"abc"
    assert [int(delay) for delay in sleeps] == [5]


async def test_a_file_that_shrinks_while_read_fails_and_leaves_no_parts(
    runtime, tmp_path, monkeypatch
):
    target = tmp_path / "large.bin"
    target.write_bytes(DATA)
    real_download = LocalBoxClient.download_file

    async def truncate_after_first(
        self, sandbox_id, file_path, local_file_path, timeout=None
    ):
        await real_download(self, sandbox_id, file_path, local_file_path, timeout)
        target.write_bytes(DATA[:15])

    monkeypatch.setattr(LocalBoxClient, "download_file", truncate_after_first)
    with pytest.raises(SandboxError, match="changed while it was read") as caught:
        await runtime.read(str(target))
    assert caught.value.kind == "io"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["large.bin"]
