"""Compose deployment on local Docker, or on a VM runtime that hosts the Docker daemon."""

import asyncio
import atexit
import io
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
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
    DockerConfig,
    RuntimeBackend,
    RuntimeConfig,
    provision_backend,
)
from verifiers.v1.runtimes.backends.docker import cli
from verifiers.v1.runtimes.base import SERVICE_PORT, ProgramResult
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

# Docker routing and credential-helper lookup need these; local Compose must not see
# unrelated evaluator secrets through shell interpolation.
DOCKER_ENV = (
    "PATH",
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)
# One VM hosts the Docker daemon and every service.
VM_HOST = {"image": "python:3.11-slim-trixie", "workdir": "/"}
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
    trust_compose: bool = False,
    setup_timeout: float | None = None,
) -> AsyncIterator[
    tuple[
        dict[str, DockerBackend],
        Callable[[], Awaitable[str]],
        RuntimeBackend | None,
        str | None,
    ]
]:
    """Own one Compose attempt and lend its services until the context exits."""
    local = isinstance(config, DockerConfig)
    if local and not trust_compose:
        raise ValueError(
            "Local Compose tasks can access host files and Docker privileges; "
            "only run trusted tasks with --env.agent.deployment.trust-compose"
        )
    if getattr(config, "mounts", {}):
        raise ValueError("Runtime mounts are not supported for Compose tasks")
    if local and config.network_restricted:
        raise ValueError(
            "Local Compose does not enforce network policies; run the task on prime"
        )
    if getattr(config, "gpu", None):
        raise ValueError("Compose currently supports CPU tasks")
    project = ["docker", "compose", "--project-name", f"vf-{uuid.uuid4().hex}"]
    compose_argv = list(project)
    compose_env = {
        key: os.environ[key] for key in DOCKER_ENV if local and key in os.environ
    }
    host_backend: RuntimeBackend | None = None
    stack = AsyncExitStack()

    async def run_host(*argv: str) -> ProgramResult:
        if host_backend is None:
            return await cli(*argv, env=compose_env)
        return await host_backend.bind().run(list(argv), compose_env)

    def down() -> None:
        # Compose finds the project by its labels; the atexit backstop stays until
        # removal succeeds.
        subprocess.run(
            [*project, "down", "--volumes", "--remove-orphans"],
            env=compose_env,
            capture_output=True,
            timeout=60,
            check=True,
        )
        atexit.unregister(down)

    async def release() -> None:
        # Like a provider sandbox's deletion, a failed removal must not fail the
        # rollout or mask its error; the atexit backstop retries it.
        try:
            await asyncio.to_thread(down)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            detail = e.stderr.decode(errors="replace").strip() if e.stderr else ""
            logger.warning(
                "Compose project %s removal failed: %s %s", project[-1], e, detail
            )

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
            main = authored["services"].get(spec.workspace, {})
            if {"image", "build"} & main.keys():
                # A template default must not replace an authored image or skip its build.
                del base["services"][spec.workspace]["image"]
            elif (
                not spec.image_override
                and (
                    directory / "task" / Path(spec.file).parent / "Dockerfile"
                ).is_file()
            ):
                main = base["services"][spec.workspace]
                del main["image"]
                main["build"] = {"context": "${CONTEXT_DIR}"}
                main["pull_policy"] = "build"
            (directory / "base.json").write_text(json.dumps(base))
            (directory / "env.json").write_text(
                json.dumps({"services": {spec.workspace: {"environment": environment}}})
            )

            root = str(directory)
            if not local:
                host_backend = await stack.enter_async_context(
                    provision_vm_host(config, spec)
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
                if key not in DOCKER_ENV and not key.startswith(("DOCKER_", "COMPOSE_"))
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
            if local:
                assert isinstance(config, DockerConfig)
                if config.cpu is not None:
                    main["cpus"] = config.cpu
                if config.memory is not None:
                    main["mem_limit"] = f"{config.memory}g"
                # Port publication belongs to the service that owns main's network.
                publish = overlay["services"].setdefault(owner, {})
                publish["ports"] = [f"127.0.0.1::{SERVICE_PORT}"]
                if sys.platform != "linux":
                    publish["extra_hosts"] = {"host.docker.internal": "host-gateway"}
            if host_backend is None:
                (directory / "overlay.json").write_text(json.dumps(overlay))
            else:
                await host_backend.bind().write(
                    f"{root}/overlay.json", json.dumps(overlay).encode()
                )
            compose_argv += ["-f", f"{root}/overlay.json"]

            if local:
                atexit.register(down)
                stack.push_async_callback(release)
            # Cancellation stops the CLI; the stack removes the project or the VM.
            await compose("up", "--detach", "--wait")
            containers = await compose(
                "ps", "--all", "--format", "{{.ID}} {{.Service}}"
            )
            service_url = None
            if local:
                published = await compose("port", owner, str(SERVICE_PORT))
                service_url = "http://" + next(
                    address
                    for address in published.splitlines()
                    if address.startswith("127.0.0.1:")
                )
            backends: dict[str, DockerBackend] = {}
            for line in containers.splitlines():
                container, service = line.split()
                backends[service] = await stack.enter_async_context(
                    DockerBackend.attach(
                        config,
                        container,
                        host_backend=host_backend,
                        service_url=service_url if service == spec.workspace else None,
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
            backends[spec.workspace].config.image
            if services[spec.workspace].get("build")
            else None,
        )
    finally:
        await run_shielded(stack.aclose())


@asynccontextmanager
async def provision_vm_host(
    config: RuntimeConfig, spec: ComposeSpec
) -> AsyncIterator[RuntimeBackend]:
    """Provision the VM that hosts the Docker daemon and every service."""
    host_config = VM_HOST
    if spec.host_image is not None:
        host_config = {**host_config, "image": spec.host_image}
    async with provision_backend(config.model_copy(update=host_config)) as host:
        runtime = host.bind()
        install = await runtime.run(["sh", "-c", INSTALL_DOCKER], {})
        if install.exit_code:
            raise SandboxError(
                f"Docker bootstrap failed: {install.stderr} {install.stdout}"
            )
        await runtime.run_background(
            ["dockerd", "--host=unix:///var/run/docker.sock"], {}, "/tmp/dockerd.log"
        )
        async with asyncio.timeout(60):
            while (await runtime.run(["docker", "info"], {})).exit_code:
                await asyncio.sleep(1)
        loaded = await runtime.run(["sh", "-c", LOAD_IMAGES], {})
        if loaded.exit_code:
            raise SandboxError(
                f"Loading host images failed: {loaded.stderr or loaded.stdout}"
            )
        yield host


def pack_directory(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for item in directory.iterdir():
            archive.add(item, arcname=item.name)
    return buffer.getvalue()
