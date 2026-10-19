"""A Compose service on a host VM checkpoints and restores through its host."""

from contextlib import asynccontextmanager

import pytest

from verifiers.v1.runtimes import DockerBackend, DockerConfig
from verifiers.v1.runtimes.modal import ModalBackend, ModalConfig
from verifiers.v1.runtimes.prime import PrimeBackend, PrimeConfig


async def test_compose_mirrors_do_not_reconfigure_local_docker(tmp_path):
    from verifiers.v1.runtimes.compose import ComposeSpec, compose_services

    spec = ComposeSpec(str(tmp_path), registry_mirrors=("https://mirror.gcr.io",))
    with pytest.raises(ValueError, match="configure the local Docker daemon"):
        async with compose_services(DockerConfig(), spec, {}, trust_compose=True):
            pytest.fail("local Docker must not be configured by a task")


@pytest.mark.parametrize("workdir", [None, "/explicit"])
@pytest.mark.parametrize(
    "backend_type,config_type",
    [
        (PrimeBackend, PrimeConfig),
        (ModalBackend, ModalConfig),
        (DockerBackend, DockerConfig),
    ],
)
async def test_composition_preserves_the_declared_workspace(
    monkeypatch, tmp_path, workdir, backend_type, config_type
):
    from verifiers.v1.runtimes import prime, strategies
    from verifiers.v1.runtimes.compose import ComposeSpec

    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    seen = []

    @asynccontextmanager
    async def services(config, spec, environment, **kwargs):
        seen.append(config.workdir)
        yield {"main": object()}, None, None

    monkeypatch.setattr(strategies, "compose_services", services)
    backend = backend_type(config_type(workdir=workdir))
    deployment = backend.deploy(spec=ComposeSpec(directory=str(tmp_path)))
    try:
        await deployment.start()
        assert seen == [workdir]
    finally:
        await deployment.close()


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
