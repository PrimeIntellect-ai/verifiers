"""A flow's sweep runs on hosts whose runtimes are all remote: no docker binary, no failure."""

import shutil

import pytest

from verifiers.v1.runtimes import docker as docker_runtime


@pytest.mark.asyncio
async def test_sweep_containers_is_a_noop_without_docker(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    async def no_engine(*args):
        raise AssertionError("no engine may be invoked when none is installed")

    monkeypatch.setattr(docker_runtime, "cli", no_engine)
    assert await docker_runtime.sweep_containers("verifiers.run:test") == 0
