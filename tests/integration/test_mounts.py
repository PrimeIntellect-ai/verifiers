"""Real storage mounts, exercised through the runtime API without model calls."""

import asyncio
import os
import shutil
from uuid import uuid4

import pytest
from prime_sandboxes import APIError, AsyncSandboxClient

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.backends.docker import cli


@pytest.mark.docker
async def test_docker_bind_mounts(tmp_path):
    if shutil.which("docker") is None or (await cli("docker", "info")).exit_code:
        pytest.skip("needs a reachable Docker daemon")

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = dataset / "input.txt"
    source.write_bytes(b"mounted input")
    results = tmp_path / "results"
    results.mkdir()
    config = vf.DockerConfig(
        mounts={
            "/dataset": vf.BindMount(source=str(dataset)),
            "/input.txt": vf.BindMount(source=str(source)),
            "/results": vf.BindMount(source=str(results), read_only=False),
        }
    )
    async with asyncio.timeout(600), provision_runtime(config) as runtime:
        for path in ("/dataset/input.txt", "/input.txt"):
            assert await runtime.read(path) == b"mounted input"
            with pytest.raises(SandboxError, match="Read-only file system"):
                await runtime.write(path, b"overwrite")
        await runtime.write("/results/output.txt", b"persisted output")
        assert (results / "output.txt").read_bytes() == b"persisted output"

    # The caller's source paths and writable data survive container teardown.
    assert source.read_bytes() == b"mounted input"
    assert (results / "output.txt").read_bytes() == b"persisted output"


@pytest.mark.prime
@pytest.mark.skipif(not os.environ.get("PRIME_API_KEY"), reason="needs PRIME_API_KEY")
async def test_prime_volume_publish_and_reuse(monkeypatch):
    # Personal scope lets the test delete its volume without requiring team admin.
    monkeypatch.setenv("PRIME_TEAM_ID", "")
    name = f"vf-ci-{uuid4().hex}"
    writer_config = vf.PrimeConfig(
        labels=["vf-ci"],
        mounts={"/data": vf.PrimeVolume(name=name, read_only=False)},
    )
    reader_config = vf.PrimeConfig(
        labels=["vf-ci"], mounts={"/data": vf.PrimeVolume(name=name)}
    )
    async with AsyncSandboxClient() as client:
        try:
            async with asyncio.timeout(600):
                # Starting the writer creates the previously nonexistent volume.
                async with provision_runtime(writer_config) as writer:
                    await writer.write("/data/published.txt", b"published content")
                    await client.volumes.publish(writer.info.id, name)
                    await writer.write("/data/unpublished.txt", b"private content")

                # A fresh VM reuses the volume after writer teardown, at its published head.
                async with provision_runtime(reader_config) as reader:
                    assert (
                        await reader.read("/data/published.txt") == b"published content"
                    )
                    missing = await reader.run(
                        ["test", "!", "-e", "/data/unpublished.txt"], {}
                    )
                    assert missing.exit_code == 0, missing
                    denied = await reader.run(
                        ["sh", "-c", "printf overwrite > /data/published.txt"], {}
                    )
                    assert denied.exit_code != 0, denied
                    assert "read-only file system" in denied.stderr.lower(), denied
                    assert (
                        await reader.read("/data/published.txt") == b"published content"
                    )
        finally:
            # Runtime teardown owns VMs; the caller owns the persistent volume.
            try:
                await client.volumes.delete(name)
            except APIError as error:
                if not str(error).startswith("HTTP 404:"):
                    raise
