"""A prepared Compose host image ships the images the services ran, as the archive and
manifest a deployment loads, with Docker installed."""

import json
import shlex
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from verifiers.v1.runtimes import DockerConfig, PrimeConfig, prime_images
from verifiers.v1.runtimes import compose as module
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.runtimes.compose import ComposeSpec, prepare_host_image


class Host:
    def __init__(self):
        self.commands = []

    def bind(self):
        return self

    async def run(self, argv, env):
        self.commands.append(argv)
        return ProgramResult(0, "", "")


async def test_the_image_ships_what_the_services_ran(monkeypatch, tmp_path):
    host = Host()
    backends = {
        "main": SimpleNamespace(_container="c-main"),
        "db": SimpleNamespace(_container="c-db"),
    }
    started = []

    @asynccontextmanager
    async def services(config, spec, environment, **kwargs):
        started.append(spec)
        yield backends, None, host, None

    pushed = {}

    async def push(upload, name, tag, **kwargs):
        await upload("https://upload")
        pushed.update(name=name, tag=tag)
        return SimpleNamespace(ref=f"prime/team/{name}:{tag}")

    monkeypatch.setattr(module, "compose_services", services)
    monkeypatch.setattr(prime_images, "push_image", push)
    spec = ComposeSpec(str(tmp_path))
    image = await prepare_host_image(PrimeConfig(), spec, "stack", "p-1")
    assert image.ref == "prime/team/stack:p-1" and started == [spec]
    assert pushed == {"name": "stack", "tag": "p-1"}

    pack, put = host.commands
    script = pack[2]
    assert (
        "docker inspect --format '{{.Image}}' c-main)\" vf-prepared/main:p-1" in script
    )
    assert (
        "docker save -o /tmp/vf-prepare/context/images.tar vf-prepared/main:p-1 vf-prepared/db:p-1"
        in script
    )
    manifest = {
        "version": 1,
        "services": {"main": "vf-prepared/main:p-1", "db": "vf-prepared/db:p-1"},
        "source": module.environment_digest(spec),
    }
    assert shlex.quote(json.dumps(manifest)) in script
    assert f"COPY images.tar manifest.json {module.COMPOSE_IMAGES}/" in script
    assert "docker.io" in script  # Docker is installed in the image
    assert put[:2] == ["python3", "-c"] and put[3:] == [
        "https://upload",
        "/tmp/vf-prepare/context.tar.gz",
    ]


async def test_preparing_needs_a_provider_vm(tmp_path):
    with pytest.raises(TypeError):
        await prepare_host_image(DockerConfig(), ComposeSpec(str(tmp_path)), "s", "t")


async def test_a_host_image_prepared_from_another_environment_is_refused(
    monkeypatch, tmp_path
):
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: app\n"
    )
    manifest = {
        "version": 1,
        "services": {"main": "vf-prepared/main:p-1"},
        "source": "an older environment",
    }

    class PreparedHost(Host):
        restored = False

        async def run(self, argv, env):
            self.commands.append(argv)
            if "manifest.json" in " ".join(argv):
                return ProgramResult(0, json.dumps(manifest), "")
            return ProgramResult(0, "", "")

    @asynccontextmanager
    async def vm_host(config, spec):
        yield PreparedHost()

    monkeypatch.setattr(module, "provision_vm_host", vm_host)
    spec = ComposeSpec(str(tmp_path), host_image="prime/team/stack:p-1")
    with pytest.raises(module.SandboxError, match="prepare it again"):
        async with module.compose_services(PrimeConfig(), spec, {}):
            pytest.fail("a stale host image ran")
    # The current environment's image gets past the check, to staging the task.
    manifest["source"] = module.environment_digest(spec)

    async def staged(self, path, data):
        raise RuntimeError("staged")

    monkeypatch.setattr(PreparedHost, "write", staged, raising=False)
    with pytest.raises(RuntimeError, match="staged"):
        async with module.compose_services(PrimeConfig(), spec, {}):
            pass
