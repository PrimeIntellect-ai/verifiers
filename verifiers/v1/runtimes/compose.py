"""Compose deployment on a provider VM that hosts the Docker daemon."""

import asyncio
import io
import json
import logging
import shlex
import shutil
import tarfile
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    DockerBackend,
    ModalConfig,
    PrimeConfig,
    RuntimeBackend,
    RuntimeConfig,
    provision_backend,
)
from verifiers.v1.runtimes.base import SERVICE_PORT, ProgramResult
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

# One provider VM hosts the Docker daemon and every service.
VM_HOST = {
    PrimeConfig: {"image": "python:3.11-slim-trixie", "workdir": "/"},
    ModalConfig: {"image": "docker:28.3.3-dind", "workdir": "/", "vm": True},
}
INSTALL_DOCKER = (
    "command -v dockerd >/dev/null || { export DEBIAN_FRONTEND=noninteractive; "
    "apt-get update -qq && apt-get install -y -qq --no-install-recommends docker.io "
    "docker-cli docker-compose iptables > /tmp/docker-install.log 2>&1 "
    "|| { tail -40 /tmp/docker-install.log; exit 1; }; }"
)
# A host image can ship `docker save` archives here so services need no registry.
# Each archive is removed once loaded, returning its disk space to the services.
COMPOSE_IMAGES = "/opt/verifiers/compose-images"
LOAD_IMAGES = (
    f'for f in {COMPOSE_IMAGES}/*.tar; do [ -e "$f" ] || continue; '
    'docker load -q -i "$f" && rm -f "$f" || exit 1; done'
)


@dataclass(frozen=True)
class ComposeSpec:
    """Task requirements described by a Docker Compose file and its build inputs.

    Services are the named containers in that file; `workspace` selects the one
    used for task commands and files. Deployment manages their running lifetimes.
    """

    directory: str
    file: str = "environment/docker-compose.yaml"
    workspace: str = "main"
    host_image: str | None = None
    image_override: bool = False


@asynccontextmanager
async def compose_services(
    config: RuntimeConfig,
    spec: ComposeSpec,
    environment: dict[str, str],
    *,
    setup_timeout: float | None = None,
) -> AsyncIterator[
    tuple[dict[str, DockerBackend], Callable[[], Awaitable[str]], RuntimeBackend]
]:
    """Own one Compose attempt and lend its services until the context exits."""
    assert isinstance(config, (PrimeConfig, ModalConfig))
    if config.gpu:
        raise ValueError("Compose currently supports CPU tasks")
    if isinstance(config, ModalConfig) and not config.network_access:
        raise ValueError("Compose on Modal requires network_access=True")
    project = ["docker", "compose", "--project-name", f"vf-{uuid.uuid4().hex}"]
    compose_argv = list(project)
    compose_env: dict[str, str] = {}
    host_backend: RuntimeBackend | None = None
    stack = AsyncExitStack()

    async def run_host(*argv: str) -> ProgramResult:
        assert host_backend is not None
        return await host_backend.bind().run(list(argv), compose_env)

    async def compose(*args: str, timeout: float | None = None) -> str:
        async with asyncio.timeout(timeout):
            result = await run_host(*compose_argv, *args)
        if result.exit_code:
            raise SandboxError(
                f"Compose {args[0]} failed: {result.stderr or result.stdout}"
            )
        return result.stdout

    try:
        async with asyncio.timeout(setup_timeout):
            import yaml

            directory = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="vf-compose-"))
            )
            # Relative binds and builds write into this rollout's copy of the task.
            await run_shielded(
                asyncio.to_thread(
                    shutil.copytree,
                    Path(spec.directory).resolve(),
                    directory / "task",
                )
            )
            authored = yaml.safe_load((directory / "task" / spec.file).read_text())
            for kind in ("volumes", "networks"):
                if any(
                    value and (value.get("name") or value.get("external"))
                    for value in authored.get(kind, {}).values()
                ):
                    raise SandboxError(f"Compose {kind} must use project-scoped names")
            if any(
                service.get("container_name")
                for service in authored["services"].values()
            ):
                raise SandboxError(
                    "Remove container_name so Compose can name each rollout's services"
                )
            base = {
                "services": {
                    spec.workspace: {
                        "image": config.image,
                        "command": [
                            "sh",
                            "-c",
                            "trap 'exit 0' TERM; while :; do sleep 3600 & wait $!; done",
                        ],
                    }
                }
            }
            if {"image", "build"} & authored["services"].get(spec.workspace, {}).keys():
                # A template default must not replace an authored image or skip its build.
                del base["services"][spec.workspace]["image"]
            (directory / "base.json").write_text(json.dumps(base))
            (directory / "env.json").write_text(
                json.dumps({"services": {spec.workspace: {"environment": environment}}})
            )

            host_config = VM_HOST[type(config)]
            if spec.host_image is not None:
                host_config = {**host_config, "image": spec.host_image}
            host_backend = await stack.enter_async_context(
                provision_backend(config.model_copy(update=host_config))
            )
            if isinstance(config, PrimeConfig):
                install = await host_backend.bind().run(
                    ["sh", "-c", INSTALL_DOCKER], {}
                )
                if install.exit_code:
                    raise SandboxError(
                        f"Docker bootstrap failed: {install.stderr} {install.stdout}"
                    )
            await host_backend.bind().run_background(
                ["dockerd", "--host=unix:///var/run/docker.sock"],
                {},
                "/tmp/dockerd.log",
            )
            async with asyncio.timeout(60):
                while (await run_host("docker", "info")).exit_code:
                    await asyncio.sleep(1)
            loaded = await run_host("sh", "-c", LOAD_IMAGES)
            if loaded.exit_code:
                raise SandboxError(
                    f"Loading host images failed: {loaded.stderr or loaded.stdout}"
                )
            root = "/vf-deployment"
            await host_backend.bind().write(
                "/tmp/vf-deployment.tar.gz",
                pack_directory(directory),
            )
            staged = await run_host(
                "sh",
                "-c",
                f"mkdir -p {shlex.quote(root)} && tar -xzf /tmp/vf-deployment.tar.gz -C {shlex.quote(root)}",
            )
            if staged.exit_code:
                raise SandboxError(
                    f"Compose environment staging failed: {staged.stderr}"
                )
            probe = await run_host("docker", "compose", "version")
            if probe.exit_code:
                raise SandboxError(
                    "Docker Compose v2 (`docker compose`) is required on the Docker "
                    "host; install the Compose v2 plugin or use a host image that "
                    "includes it. Standalone `docker-compose` v1 is unsupported."
                )
            project_dir = f"{root}/task/{Path(spec.file).parent}"
            compose_argv += ["--project-directory", project_dir]
            for file in (
                "base.json",
                f"task/{spec.file}",
                "env.json",
            ):
                compose_argv += ["-f", f"{root}/{file}"]
            compose_env.update(
                (key, value)
                for key, value in environment.items()
                if key not in ("PATH", "HOME")
                and not key.startswith(("DOCKER_", "COMPOSE_"))
            )
            compose_env.update(
                {
                    "MAIN_IMAGE_NAME": project[-1],
                    "CONTEXT_DIR": project_dir,
                    "PREBUILT_IMAGE_NAME": config.image,
                }
            )

            # Validate authored services after interpolation, before adding our own.
            resolved = json.loads(await compose("config", "--format", "json"))
            services = resolved["services"]
            for service in services.values():
                if service.get("network_mode") == "host":
                    raise SandboxError("Compose requires an isolated service network")
                if service.get("gpus") or (
                    service.get("deploy", {})
                    .get("resources", {})
                    .get("reservations", {})
                    .get("devices")
                ):
                    raise SandboxError("Compose currently supports CPU tasks")
                if service.get("scale", 1) != 1 or (
                    service.get("deploy", {}).get("replicas", 1) != 1
                ):
                    raise SandboxError("Compose requires one container per service")
                if any(
                    port.get("published") not in (None, "", 0, "0")
                    for port in service.get("ports", [])
                ):
                    raise SandboxError(
                        "Compose ports must use dynamically assigned host ports"
                    )
            # Compose rejects cycles, so this ends at the owner of main's network.
            owner = spec.workspace
            while (mode := services[owner].get("network_mode", "")).startswith(
                "service:"
            ):
                owner = mode.removeprefix("service:")

            main = config.model_dump(
                include={"image", "workdir"}, exclude_defaults=True, exclude_none=True
            )
            if spec.image_override:
                main["image"] = config.image
            if "workdir" in main:
                main["working_dir"] = main.pop("workdir")
            overlay = {"services": {spec.workspace: main}}
            # Port publication belongs to the service that owns main's network.
            publish = overlay["services"].setdefault(owner, {})
            if isinstance(config, ModalConfig):
                publish["ports"] = [f"{SERVICE_PORT}:{SERVICE_PORT}"]
            await host_backend.bind().write(
                f"{root}/overlay.json", json.dumps(overlay).encode()
            )
            compose_argv += ["-f", f"{root}/overlay.json"]

            # Cancellation stops the CLI; the stack removes the VM.
            await compose("up", "--detach", "--wait")
            containers = await compose(
                "ps", "--all", "--format", "{{.ID}} {{.Service}}"
            )
            backends: dict[str, DockerBackend] = {}
            for line in containers.splitlines():
                container, service = line.split()
                backends[service] = await stack.enter_async_context(
                    DockerBackend.attach(
                        config,
                        container,
                        host_backend=host_backend,
                    )
                )
            if spec.workspace not in backends:
                raise SandboxError(
                    f"Compose requires workspace service {spec.workspace!r}"
                )
        yield (
            backends,
            partial(compose, "stop", spec.workspace, timeout=60),
            host_backend,
        )
    finally:
        await run_shielded(stack.aclose())


def pack_directory(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for item in directory.iterdir():
            archive.add(item, arcname=item.name)
    return buffer.getvalue()
