"""A Compose service on a host VM checkpoints and restores through its host."""

import pytest

from verifiers.v1.runtimes import DockerBackend, DockerConfig
from verifiers.v1.runtimes.prime import PrimeBackend, PrimeConfig


def test_a_local_container_has_no_checkpoints():
    backend = DockerBackend(DockerConfig())
    assert not backend.supports_checkpoint
    assert not backend.restored


async def test_a_service_checkpoints_and_deletes_through_its_host(monkeypatch):
    host = PrimeBackend(PrimeConfig())
    calls = []

    async def checkpoint():
        calls.append("checkpoint")
        return "ck-1"

    async def delete_checkpoint(checkpoint_id):
        calls.append(("delete", checkpoint_id))

    monkeypatch.setattr(host, "checkpoint", checkpoint)
    monkeypatch.setattr(host, "delete_checkpoint", delete_checkpoint)
    service = DockerBackend(PrimeConfig(), host_backend=host)
    assert service.supports_checkpoint
    assert not service.restored
    assert await service.checkpoint() == "ck-1"
    await service.delete_checkpoint("ck-1")
    assert calls == ["checkpoint", ("delete", "ck-1")]


def test_a_service_on_a_restored_host_is_restored():
    host = PrimeBackend(PrimeConfig(checkpoint="ck-1"))
    assert DockerBackend(PrimeConfig(), host_backend=host).restored


async def test_a_local_container_refuses_a_checkpoint():
    with pytest.raises(NotImplementedError):
        await DockerBackend(DockerConfig()).checkpoint()
