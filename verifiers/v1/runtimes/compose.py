"""Task services on local Docker Compose or a Prime VM, from Linux or macOS."""

import asyncio
import atexit
import io
import json
import os
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
    PrimeConfig,
    RuntimeBackend,
    provision_backend,
)
from verifiers.v1.runtimes.backends.docker import cli
from verifiers.v1.runtimes.base import SERVICE_PORT
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)

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
VM_HOST_IMAGE = "python:3.11-slim-trixie"
INSTALL_DOCKER = (
    "command -v dockerd >/dev/null && docker compose version >/dev/null 2>&1 || "
    "{ export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && "
    "apt-get install -y -qq --no-install-recommends docker.io docker-cli "
    "docker-compose iptables > /tmp/docker-install.log 2>&1 "
    "|| { tail -40 /tmp/docker-install.log; exit 1; }; }"
)
# A host image can ship docker-save archives, avoiding registry access during setup.
COMPOSE_IMAGES = "/opt/verifiers/compose-images"
LOAD_IMAGES = (
    f'for f in {COMPOSE_IMAGES}/*.tar; do [ -e "$f" ] || continue; '
    'docker load -q -i "$f" && rm -f "$f" || exit 1; done'
)
# Harbor's Compose templates mount its log directories as `${HOST_<NAME>_PATH}:
# ${ENV_<NAME>_PATH}`. Harbor binds one host directory per name, so every service
# that mounts a name shares it; a project volume per name keeps that sharing.
HARBOR_LOG_MOUNTS = {
    "VERIFIER_LOGS": "/logs/verifier",
    "AGENT_LOGS": "/logs/agent",
    "ARTIFACTS": "/logs/artifacts",
}


@dataclass(frozen=True)
class ComposeSpec:
    """Task requirements described by a Docker Compose file and its build inputs.

    Services are the named containers in that file; `workspace` selects the one
    used for task commands and files. Deployment manages their running lifetimes.
    """

    directory: str
    file: str | None = "environment/docker-compose.yaml"
    """None builds directory/Dockerfile as a single workspace service."""
    workspace: str = "main"
    image_override: bool = False
    build_dockerfile: bool = False
    host_image: str | None = None
    """Compose VM image, optionally with Docker and baked-in service archives."""


def _harbor_volume(name: str) -> str:
    return "harbor-" + name.lower().replace("_", "-")


@asynccontextmanager
async def compose_project(
    config: DockerConfig | PrimeConfig,
    spec: ComposeSpec,
    environment: dict[str, str],
    *,
    host_backend: RuntimeBackend | None = None,
) -> AsyncIterator["ComposeProject"]:
    """Resolve a private copy of a task and own any local preparation containers."""
    if config.mounts:
        raise ValueError("Runtime mounts are not supported for Compose tasks")
    project = ["docker", "compose", "--project-name", f"vf-{uuid.uuid4().hex}"]
    compose_argv = list(project)
    compose_env = {
        key: os.environ[key]
        for key in DOCKER_ENV
        if host_backend is None and key in os.environ
    }
    stack = AsyncExitStack()
    started = False

    def down() -> None:
        # Find the project by its labels without reading the evaluator's .env.
        # The atexit backstop stays until removal succeeds.
        subprocess.run(
            [
                *project,
                "--env-file",
                "/dev/null",
                "down",
                "--volumes",
                "--remove-orphans",
            ],
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
        nonlocal started
        if host_backend is None and args[0] in {"create", "up"} and not started:
            started = True
            atexit.register(down)
            stack.push_async_callback(release)
        async with asyncio.timeout(timeout):
            result = (
                await cli(*compose_argv, *args, env=compose_env)
                if host_backend is None
                else await host_backend.bind().run([*compose_argv, *args], compose_env)
            )
        if result.exit_code:
            raise SandboxError(
                f"Compose {args[0]} failed: {result.stderr or result.stdout}"
            )
        return result.stdout

    try:
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
                symlinks=isinstance(config, PrimeConfig),
            )
        )
        task_dir = (directory / "task").resolve()
        compose_file = spec.file or "Dockerfile"
        if isinstance(config, PrimeConfig) and not (
            task_dir / compose_file
        ).resolve().is_relative_to(task_dir):
            raise SandboxError("Native Compose files must be inside the task")
        authored = (
            yaml.safe_load((task_dir / spec.file).read_text())
            if spec.file is not None
            else {"services": {spec.workspace: {}}}
        )
        if (
            isinstance(config, PrimeConfig)
            and host_backend is None
            and (
                authored.get("include")
                or any(
                    service.get("extends")
                    for service in authored.get("services", {}).values()
                )
            )
        ):
            raise SandboxError("Native Compose does not support include or extends")
        for kind in ("volumes", "networks"):
            if host_backend is None and any(
                value and (value.get("name") or value.get("external"))
                for value in authored.get(kind, {}).values()
            ):
                raise SandboxError(f"Compose {kind} must use project-scoped names")
        # Native Compose renames containers on the build host and keeps the name as an alias.
        if isinstance(config, DockerConfig) and any(
            service.get("container_name")
            for service in authored.get("services", {}).values()
        ):
            raise SandboxError(
                "Remove container_name so Compose can name each rollout's services"
            )
        root = str(directory)
        if host_backend is not None:
            root = "/vf-deployment"
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                archive.add(directory, arcname=".")
            await host_backend.bind().write("/vf-deployment.tar.gz", buffer.getvalue())
            staged = await host_backend.bind().run(
                [
                    "sh",
                    "-c",
                    f"mkdir -p {root} && tar -xzf /vf-deployment.tar.gz -C {root} && rm /vf-deployment.tar.gz",
                ],
                {},
            )
            if staged.exit_code:
                raise SandboxError(
                    f"Compose environment staging failed: {staged.stderr}"
                )
        probe = (
            await cli("docker", "compose", "version", env=compose_env)
            if host_backend is None
            else await host_backend.bind().run(["docker", "compose", "version"], {})
        )
        if probe.exit_code:
            raise SandboxError(
                "Docker Compose v2 (`docker compose`) is required on the Docker "
                "host; install the Compose v2 plugin. "
                "Standalone `docker-compose` v1 is unsupported."
            )
        project_dir = f"{root}/task/{Path(compose_file).parent}"
        compose_argv += ["--project-directory", project_dir]
        if isinstance(config, PrimeConfig):
            env_file = directory / "task" / Path(compose_file).parent / ".env"
            if not env_file.resolve().is_relative_to(task_dir):
                raise SandboxError("Native Compose .env must be inside the task")
            compose_argv += [
                "--env-file",
                f"{project_dir}/.env" if env_file.exists() else "/dev/null",
            ]
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
                "TEST_DIR": "/tests",
            }
        )
        for name, path in HARBOR_LOG_MOUNTS.items():
            compose_env[f"HOST_{name}_PATH"] = _harbor_volume(name)
            compose_env[f"ENV_{name}_PATH"] = path
        # Harbor's templates size services from these; Compose rejects them empty.
        if config.cpu is not None:
            compose_env["CPUS"] = f"{config.cpu:g}"
        if config.memory is not None:
            compose_env["MEMORY"] = f"{round(config.memory * 1024)}M"

        authored_file = f"task/{spec.file}" if spec.file else None
        if host_backend is not None and authored.get("include"):
            # Includes own their resources: resolve them before merging workspace
            # defaults and overrides, or Compose rejects duplicate service names.
            compose_argv += ["-f", f"{root}/{authored_file}"]
            authored = json.loads(
                await compose("config", "--no-consistency", "--format", "json")
            )
            del compose_argv[-2:]
            # Config emits null for an absent command; retain our workspace default
            # unless the included service declares a concrete command.
            main = authored.get("services", {}).get(spec.workspace, {})
            if main.get("command") is None:
                main.pop("command", None)
            authored_file = "authored.json"
            await host_backend.bind().write(
                f"{root}/{authored_file}",
                json.dumps(authored).encode(),
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
            },
            "volumes": {_harbor_volume(name): {} for name in HARBOR_LOG_MOUNTS},
        }
        if {"image", "build"} & authored.get("services", {}).get(
            spec.workspace, {}
        ).keys():
            # A template default must not replace an authored image or skip its build.
            del base["services"][spec.workspace]["image"]
        elif (
            (spec.build_dockerfile or spec.file is None)
            and not spec.image_override
            and (task_dir / Path(compose_file).parent / "Dockerfile").is_file()
        ):
            del base["services"][spec.workspace]["image"]
            base["services"][spec.workspace]["build"] = {
                "context": ".",
                "dockerfile": "Dockerfile",
            }
        elif spec.file is None and not spec.image_override:
            raise SandboxError(f"No Dockerfile in {spec.directory}")
        # env.json is a Compose file: escape `$` so values reach the service verbatim.
        escaped = {key: value.replace("$", "$$") for key, value in environment.items()}
        definitions = {
            "base.json": base,
            "env.json": {"services": {spec.workspace: {"environment": escaped}}},
        }
        for file, definition in definitions.items():
            if host_backend is None:
                (directory / file).write_text(json.dumps(definition))
            else:
                await host_backend.bind().write(
                    f"{root}/{file}", json.dumps(definition).encode()
                )
        for file in [
            "base.json",
            *([authored_file] if authored_file else []),
            "env.json",
        ]:
            compose_argv += ["-f", f"{root}/{file}"]

        if isinstance(config, PrimeConfig) and host_backend is None:
            # Resolve filenames without loading them: task env files cannot read host secrets.
            unresolved = json.loads(
                await compose("config", "--no-env-resolution", "--format", "json")
            )
            for service in unresolved["services"].values():
                for env_file in service.get("env_file", []):
                    if not Path(env_file["path"]).resolve().is_relative_to(task_dir):
                        raise SandboxError(
                            "Native Compose env_file must be inside the task"
                        )
        resolved = json.loads(await compose("config", "--format", "json"))
        yield ComposeProject(
            directory,
            compose_argv,
            compose_env,
            resolved,
            compose,
            "build" in base["services"][spec.workspace],
            root,
        )
    finally:
        await run_shielded(stack.aclose())


@dataclass
class ComposeProject:
    directory: Path
    argv: list[str]
    env: dict[str, str]
    config: dict
    run: Callable[..., Awaitable[str]]
    implicit_build: bool
    root: str


@asynccontextmanager
async def compose_services(
    config: DockerConfig | PrimeConfig,
    spec: ComposeSpec,
    environment: dict[str, str],
    *,
    trust_compose: bool = False,
    setup_timeout: float | None = None,
) -> AsyncIterator[tuple[dict[str, DockerBackend], Callable[[], Awaitable[str]]]]:
    """Own one Compose attempt and lend its services until the context exits."""
    local = isinstance(config, DockerConfig)
    if config.mounts:
        raise ValueError("Runtime mounts are not supported for Compose tasks")
    if not local and config.checkpoint:
        raise ValueError("Compose VMs require a host image, not a checkpoint")
    if local and not trust_compose:
        raise ValueError(
            "Local Compose tasks can access host files and Docker privileges; "
            "only run trusted tasks with --env.agent.deployment.trust-compose"
        )
    if local and config.network_restricted:
        raise ValueError(
            "Local Compose does not enforce network policies; use ServicesSpec on Prime"
        )
    if config.gpu:
        raise ValueError("Compose currently supports CPU tasks")
    stack = AsyncExitStack()
    try:
        async with asyncio.timeout(setup_timeout):
            host = (
                None
                if local
                else await stack.enter_async_context(provision_vm_host(config, spec))
            )
            project = await stack.enter_async_context(
                compose_project(config, spec, environment, host_backend=host)
            )
            compose = project.run
            directory, compose_argv = project.directory, project.argv
            root = project.root
            # Validate authored services after interpolation, before adding our own.
            services = project.config["services"]
            for service in services.values():
                if local and service.get("network_mode") == "host":
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
                if local and any(
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
            elif project.implicit_build:
                main.pop("image", None)
            if "workdir" in main:
                main["working_dir"] = main.pop("workdir")
            overlay = {"services": {spec.workspace: main}}
            # Port publication belongs to the service that owns main's network.
            publish = overlay["services"].setdefault(owner, {})
            if local and config.cpu is not None:
                main["cpus"] = config.cpu
            if local and config.memory is not None:
                main["mem_limit"] = f"{config.memory}g"
            limits = {
                key: main[field]
                for key, field in (("cpus", "cpus"), ("memory", "mem_limit"))
                if field in main
            }
            if limits:
                main["deploy"] = {"resources": {"limits": limits}}
            if services[owner].get("network_mode") != "host":
                publish["ports"] = [
                    f"127.0.0.1::{SERVICE_PORT}" if local else str(SERVICE_PORT)
                ]
            if local and sys.platform != "linux":
                publish["extra_hosts"] = {"host.docker.internal": "host-gateway"}
            if host is None:
                (directory / "overlay.json").write_text(json.dumps(overlay))
            else:
                await host.bind().write(
                    f"{root}/overlay.json", json.dumps(overlay).encode()
                )
            compose_argv += ["-f", f"{root}/overlay.json"]

            # Cancellation stops the CLI; the stack removes the project.
            await compose("up", "--detach", "--wait")
            containers = await compose(
                "ps", "--all", "--format", "{{.ID}} {{.Service}}"
            )
            service_url = None
            host_service_port = SERVICE_PORT
            if services[owner].get("network_mode") != "host":
                published = await compose("port", owner, str(SERVICE_PORT))
                # Prefer IPv4 when both families have mappings, but accept an
                # IPv6-only Docker host. Local publication stays on IPv4 loopback.
                # Compose v2 emits :::port; newer releases bracket the host IP.
                prefixes = ("127.0.0.1:",) if local else ("0.0.0.0:", "[::]:", ":::")
                address = next(
                    (
                        address
                        for prefix in prefixes
                        for address in published.splitlines()
                        if address.startswith(prefix)
                    ),
                    None,
                )
                if address is None:
                    raise SandboxError(
                        f"Compose did not publish workspace port {SERVICE_PORT}"
                    )
                if local:
                    service_url = "http://" + address
                else:
                    host_service_port = int(address.rsplit(":", 1)[1])
            backends: dict[str, DockerBackend] = {}
            for line in containers.splitlines():
                container, service = line.split()
                backends[service] = await stack.enter_async_context(
                    DockerBackend.attach(
                        config,
                        container,
                        host_backend=host,
                        host_service_port=host_service_port
                        if service == spec.workspace
                        else None,
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
        )
    finally:
        await run_shielded(stack.aclose())


@asynccontextmanager
async def provision_vm_host(
    config: PrimeConfig, spec: ComposeSpec
) -> AsyncIterator[RuntimeBackend]:
    """Own a fresh Docker host, including partial bootstrap and cancelled setup."""
    host_config = config.model_copy(
        update={"image": spec.host_image or VM_HOST_IMAGE, "workdir": "/"}
    )
    async with provision_backend(host_config) as host:
        runtime = host.bind()
        install = await runtime.run(["sh", "-c", INSTALL_DOCKER], {})
        if install.exit_code:
            raise SandboxError(
                f"Docker bootstrap failed: {install.stderr} {install.stdout}"
            )
        if (await runtime.run(["docker", "info"], {})).exit_code:
            await runtime.run_background(
                ["dockerd", "--host=unix:///var/run/docker.sock"],
                {},
                "/tmp/dockerd.log",
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
