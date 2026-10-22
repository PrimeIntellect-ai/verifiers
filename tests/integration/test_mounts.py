"""Real storage mounts, exercised through the runtime API without model calls."""

import asyncio
import shutil

import pytest

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
