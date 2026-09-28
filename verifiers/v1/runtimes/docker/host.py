"""Own a provider VM with a Docker daemon for nested task containers."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import Runtime
from verifiers.v1.runtimes.modal import ModalConfig
from verifiers.v1.runtimes.prime import PrimeConfig

INSTALL_DOCKER = (
    "command -v dockerd >/dev/null || { export DEBIAN_FRONTEND=noninteractive; "
    "apt-get update -qq && apt-get install -y -qq --no-install-recommends docker.io "
    "docker-cli docker-compose iptables > /tmp/docker-install.log 2>&1 "
    "|| { tail -40 /tmp/docker-install.log; exit 1; }; }"
)


@asynccontextmanager
async def docker_host(
    config: PrimeConfig | ModalConfig, *, image: str | None = None
) -> AsyncIterator[Runtime]:
    """Keep the daemon and all its containers inside one disposable VM."""
    from verifiers.v1.runtimes import provision_runtime

    values = {"workdir": "/"}
    if isinstance(config, PrimeConfig):
        values["image"] = image or "python:3.11-slim-trixie"
    else:
        values.update(image=image or "docker:28.3.3-dind", vm=True)
    async with provision_runtime(config.model_copy(update=values)) as host:
        if isinstance(config, PrimeConfig):
            install = await host.run(["sh", "-c", INSTALL_DOCKER], {})
            if install.exit_code:
                raise SandboxError(
                    f"Docker bootstrap failed: {install.stderr} {install.stdout}"
                )
        await host.run_background(
            ["dockerd", "--host=unix:///var/run/docker.sock"], {}, "/tmp/dockerd.log"
        )
        async with asyncio.timeout(60):
            while (await host.run(["docker", "info"], {})).exit_code:
                await asyncio.sleep(1)
        yield host
