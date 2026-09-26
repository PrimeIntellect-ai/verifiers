"""Harbor owns one Compose project per rollout, on local Docker or inside a provider VM."""

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
from functools import partial
from pathlib import Path

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    DockerConfig,
    DockerRuntime,
    ModalConfig,
    PrimeConfig,
    RuntimeConfig,
    provision_runtime,
)
from verifiers.v1.runtimes.base import SERVICE_PORT, ProgramResult
from verifiers.v1.runtimes.container import cli
from verifiers.v1.tasksets.harbor.taskset import HarborTask
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


@asynccontextmanager
async def compose_services(
    config: RuntimeConfig,
    task: HarborTask,
    *,
    trust_compose: bool = False,
    setup_timeout: float | None = None,
) -> AsyncIterator[tuple[dict[str, DockerRuntime], Callable[[], Awaitable[str]]]]:
    """Own one Compose attempt and lend its services until the context exits."""
    if not isinstance(config, (DockerConfig, PrimeConfig, ModalConfig)):
        raise TypeError("Harbor Compose requires Docker, Prime VM or Modal VM")
    local = isinstance(config, DockerConfig)
    if local and not trust_compose:
        raise ValueError(
            "Local Compose tasks can access host files and Docker privileges; "
            "only run trusted tasks with --env.trust-compose"
        )
    if config.gpu:
        raise ValueError("Harbor Compose currently supports CPU tasks")
    if local and config.network_restricted:
        raise ValueError("Harbor Compose on local Docker requires public networking")
    if isinstance(config, ModalConfig) and not config.network_access:
        raise ValueError("Harbor Compose on Modal requires network_access=True")
    project = ["docker", "compose", "--project-name", f"vf-{uuid.uuid4().hex}"]
    compose_argv = list(project)
    compose_env = {
        key: os.environ[key] for key in DOCKER_ENV if local and key in os.environ
    }
    host = None
    stack = AsyncExitStack()

    async def run_host(*argv: str) -> ProgramResult:
        if host is None:
            return await cli(*argv, env=compose_env)
        return await host.run(list(argv), compose_env)

    async def compose(*args: str, timeout: float | None = None) -> str:
        async with asyncio.timeout(timeout):
            result = await run_host(*compose_argv, *args)
        if result.exit_code:
            raise SandboxError(
                f"Compose {args[0]} failed: {result.stderr or result.stdout}"
            )
        return result.stdout

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

    try:
        async with asyncio.timeout(setup_timeout):
            import yaml
            from harbor.environments.docker import (
                COMPOSE_PREBUILT_PATH,
                write_env_compose_file,
            )
            from harbor.environments.docker.compose_env import ComposeInfraEnvVars
            from harbor.environments.tar_transfer import (
                pack_dir_to_bytes,
                remote_unpack_command,
            )

            directory = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="vf-harbor-"))
            )
            # Relative binds and builds write into this rollout's copy of the task.
            await run_shielded(
                asyncio.to_thread(
                    shutil.copytree,
                    Path(task.data.task_dir).resolve(),
                    directory / "task",
                )
            )
            authored = yaml.safe_load(
                (directory / "task/environment/docker-compose.yaml").read_text()
            )
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
            base = yaml.safe_load(COMPOSE_PREBUILT_PATH.read_text())
            if {"image", "build"} & authored["services"].get("main", {}).keys():
                # A template default must not replace an authored image or skip its build.
                del base["services"]["main"]["image"]
            (directory / "base.json").write_text(json.dumps(base))
            write_env_compose_file(directory / "env.json", task.runtime_env())

            root = str(directory)
            if not local:
                host_config = VM_HOST[type(config)]
                if task.data.compose_host_image is not None:
                    host_config = {**host_config, "image": task.data.compose_host_image}
                host = await stack.enter_async_context(
                    provision_runtime(config.model_copy(update=host_config))
                )
                if isinstance(config, PrimeConfig):
                    install = await host.run(["sh", "-c", INSTALL_DOCKER], {})
                    if install.exit_code:
                        raise SandboxError(
                            f"Docker bootstrap failed: {install.stderr} {install.stdout}"
                        )
                await host.run_background(
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
                root = "/harbor"
                await host.write(
                    "/tmp/harbor.tar.gz",
                    pack_dir_to_bytes(directory, compress=True).getvalue(),
                )
                staged = await run_host(
                    "sh", "-c", remote_unpack_command("/tmp/harbor.tar.gz", root)
                )
                if staged.exit_code:
                    raise SandboxError(
                        f"Compose environment staging failed: {staged.stderr}"
                    )
            project_dir = f"{root}/task/environment"
            compose_argv += ["--project-directory", project_dir]
            for file in (
                "base.json",
                "task/environment/docker-compose.yaml",
                "env.json",
            ):
                compose_argv += ["-f", f"{root}/{file}"]
            compose_env.update(
                (key, value)
                for key, value in task.runtime_env().items()
                if key not in DOCKER_ENV
            )
            compose_env.update(
                ComposeInfraEnvVars(
                    main_image_name=project[-1],
                    context_dir=project_dir,
                    prebuilt_image_name=config.image,
                ).to_env_dict()
            )

            # Validate authored services after interpolation, before adding our own.
            services = json.loads(await compose("config", "--format", "json"))[
                "services"
            ]
            for service in services.values():
                if service.get("network_mode") == "host":
                    raise SandboxError(
                        "Harbor Compose requires an isolated service network"
                    )
                if service.get("gpus") or (
                    service.get("deploy", {})
                    .get("resources", {})
                    .get("reservations", {})
                    .get("devices")
                ):
                    raise SandboxError("Harbor Compose currently supports CPU tasks")
                if service.get("scale", 1) != 1 or (
                    service.get("deploy", {}).get("replicas", 1) != 1
                ):
                    raise SandboxError(
                        "Harbor Compose requires one container per service"
                    )
                if any(
                    port.get("published") not in (None, "", 0, "0")
                    for port in service.get("ports", [])
                ):
                    raise SandboxError(
                        "Compose ports must use dynamically assigned host ports"
                    )
            # Compose rejects cycles, so this ends at the owner of main's network.
            owner = "main"
            while (mode := services[owner].get("network_mode", "")).startswith(
                "service:"
            ):
                owner = mode.removeprefix("service:")

            main = config.model_dump(
                include={"image", "workdir"}, exclude_defaults=True, exclude_none=True
            )
            if task.data.image is not None:
                main["image"] = config.image
            if "workdir" in main:
                main["working_dir"] = main.pop("workdir")
            overlay = {"services": {"main": main}}
            # Port publication belongs to the service that owns main's network.
            publish = overlay["services"].setdefault(owner, {})
            if local:
                if config.cpu is not None:
                    main["cpus"] = config.cpu
                if config.memory is not None:
                    main["mem_limit"] = f"{config.memory}g"
                publish["ports"] = [f"127.0.0.1::{SERVICE_PORT}"]
                if sys.platform != "linux":
                    publish["extra_hosts"] = {"host.docker.internal": "host-gateway"}
            elif isinstance(config, ModalConfig):
                publish["ports"] = [f"{SERVICE_PORT}:{SERVICE_PORT}"]
            if host is None:
                (directory / "overlay.json").write_text(json.dumps(overlay))
            else:
                await host.write(f"{root}/overlay.json", json.dumps(overlay).encode())
            compose_argv += ["-f", f"{root}/overlay.json"]

            if local:
                atexit.register(down)
                stack.push_async_callback(release)
            # Cancellation kills the local CLI; the stack removes the project or VM.
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
            runtimes: dict[str, DockerRuntime] = {}
            for line in containers.splitlines():
                container, service = line.split()
                runtimes[service] = await stack.enter_async_context(
                    DockerRuntime.attach(
                        config,
                        container,
                        host=host,
                        service_url=service_url if service == "main" else None,
                    )
                )
            if "main" not in runtimes:
                raise SandboxError("Harbor Compose requires a main container")
        yield runtimes, partial(compose, "stop", "main", timeout=60)
    finally:
        await run_shielded(stack.aclose())
