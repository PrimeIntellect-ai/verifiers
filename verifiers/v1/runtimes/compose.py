"""Task services on local Docker Compose."""

import asyncio
import atexit
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import DockerBackend, DockerConfig
from verifiers.v1.runtimes.backends.docker import cli
from verifiers.v1.runtimes.base import SERVICE_PORT
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


@dataclass(frozen=True)
class ComposeSpec:
    """Task requirements described by a Docker Compose file and its build inputs.

    Services are the named containers in that file; `workspace` selects the one
    used for task commands and files. Deployment manages their running lifetimes.
    """

    directory: str
    file: str = "environment/docker-compose.yaml"
    workspace: str = "main"
    image_override: bool = False
    build_dockerfile: bool = False


@asynccontextmanager
async def compose_services(
    config: DockerConfig,
    spec: ComposeSpec,
    environment: dict[str, str],
    *,
    trust_compose: bool = False,
    setup_timeout: float | None = None,
) -> AsyncIterator[tuple[dict[str, DockerBackend], Callable[[], Awaitable[str]]]]:
    """Own one Compose attempt and lend its services until the context exits."""
    if not isinstance(config, DockerConfig):
        raise TypeError("Compose requires local Docker; use ServicesSpec on Prime")
    if not trust_compose:
        raise ValueError(
            "Local Compose tasks can access host files and Docker privileges; "
            "only run trusted tasks with --env.agent.deployment.trust-compose"
        )
    if getattr(config, "mounts", {}):
        raise ValueError("Runtime mounts are not supported for Compose tasks")
    if config.network_restricted:
        raise ValueError(
            "Local Compose does not enforce network policies; use ServicesSpec on Prime"
        )
    if getattr(config, "gpu", None):
        raise ValueError("Compose currently supports CPU tasks")
    project = ["docker", "compose", "--project-name", f"vf-{uuid.uuid4().hex}"]
    compose_argv = list(project)
    compose_env = {key: os.environ[key] for key in DOCKER_ENV if key in os.environ}
    stack = AsyncExitStack()

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
            result = await cli(*compose_argv, *args, env=compose_env)
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
                spec.build_dockerfile
                and not spec.image_override
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
            probe = await cli("docker", "compose", "version", env=compose_env)
            if probe.exit_code:
                raise SandboxError(
                    "Docker Compose v2 (`docker compose`) is required on the Docker "
                    "host; install the Compose v2 plugin. "
                    "Standalone `docker-compose` v1 is unsupported."
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
            elif "build" in base["services"][spec.workspace]:
                main.pop("image", None)
            if "workdir" in main:
                main["working_dir"] = main.pop("workdir")
            overlay = {"services": {spec.workspace: main}}
            # Port publication belongs to the service that owns main's network.
            publish = overlay["services"].setdefault(owner, {})
            if config.cpu is not None:
                main["cpus"] = config.cpu
            if config.memory is not None:
                main["mem_limit"] = f"{config.memory}g"
            limits = {
                key: main[field]
                for key, field in (("cpus", "cpus"), ("memory", "mem_limit"))
                if field in main
            }
            if limits:
                main["deploy"] = {"resources": {"limits": limits}}
            publish["ports"] = [f"127.0.0.1::{SERVICE_PORT}"]
            if sys.platform != "linux":
                publish["extra_hosts"] = {"host.docker.internal": "host-gateway"}
            (directory / "overlay.json").write_text(json.dumps(overlay))
            compose_argv += ["-f", f"{root}/overlay.json"]

            atexit.register(down)
            stack.push_async_callback(release)
            # Cancellation stops the CLI; the stack removes the project.
            await compose("up", "--detach", "--wait")
            containers = await compose(
                "ps", "--all", "--format", "{{.ID}} {{.Service}}"
            )
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
