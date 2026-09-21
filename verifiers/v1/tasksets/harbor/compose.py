"""Harbor owns a Compose project locally or inside an existing provider runtime."""

import asyncio
import atexit
import json
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
    ModalRuntime,
    PrimeConfig,
    PrimeRuntime,
)
from verifiers.v1.runtimes.base import SERVICE_PORT, ProgramResult
from verifiers.v1.runtimes.container import cli
from verifiers.v1.tasksets.harbor.taskset import HarborTask
from verifiers.v1.utils.aio import run_shielded


@asynccontextmanager
async def compose_services(
    config: DockerConfig | PrimeConfig | ModalConfig,
    task: HarborTask,
    *,
    trust_compose: bool = False,
    setup_timeout: float | None = None,
) -> AsyncIterator[tuple[dict[str, DockerRuntime], Callable[[], Awaitable[str]]]]:
    """Own one Compose attempt and lend its services until the context exits."""
    if isinstance(config, DockerConfig) and not trust_compose:
        raise ValueError(
            "Local Compose tasks can access host files and Docker privileges; "
            "only run trusted tasks with --env.trust-compose"
        )
    if config.gpu:
        raise ValueError("Harbor Compose currently supports CPU tasks")
    if isinstance(config, DockerConfig) and config.mounts:
        raise ValueError(
            "Docker bind mounts are not supported for Harbor Compose tasks"
        )
    if isinstance(config, DockerConfig) and config.network_restricted:
        raise ValueError("Harbor Compose on local Docker requires public networking")
    host: PrimeRuntime | ModalRuntime | None = None
    if isinstance(config, PrimeConfig):
        host = PrimeRuntime(
            config.model_copy(
                update={"image": "python:3.11-slim-trixie", "workdir": "/"}
            )
        )
    elif isinstance(config, ModalConfig):
        if not config.network_access:
            raise ValueError("Harbor Compose on Modal requires network_access=True")
        host = ModalRuntime(
            config.model_copy(
                update={"image": "docker:28.3.3-dind", "workdir": "/", "vm": True}
            )
        )
    name = f"vf-{uuid.uuid4().hex}"
    temporary = tempfile.TemporaryDirectory(prefix="vf-harbor-")
    stack = AsyncExitStack()
    compose_argv: list[str] = []
    # Docker routing and credential-helper lookup need these; Compose must not see
    # unrelated evaluator secrets through shell interpolation.
    compose_env = {
        key: os.environ[key]
        for key in (
            "PATH",
            "HOME",
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_CONFIG",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
        )
        if host is None and key in os.environ
    }
    runtimes: dict[str, DockerRuntime] = {}

    async def run_host(*args: str, env: dict[str, str] | None = None) -> ProgramResult:
        if host is not None:
            return await host.run(list(args), env or {})
        return await cli(*args, env=env)

    async def compose(*args: str, timeout: float | None = None) -> str:
        async with asyncio.timeout(timeout):
            result = await run_host(*compose_argv, *args, env=compose_env)
        if result.exit_code:
            raise SandboxError(
                f"Compose {args[0]} failed: {result.stderr or result.stdout}"
            )
        return result.stdout

    def cleanup() -> None:
        # Keep the files and atexit backstop if teardown fails so cleanup can retry.
        if host is not None:
            host.cleanup()
        else:
            subprocess.run(
                [*compose_argv, "down", "--volumes", "--remove-orphans"],
                env=compose_env,
                capture_output=True,
                timeout=60,
                check=True,
            )
        temporary.cleanup()
        atexit.unregister(cleanup)

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

            directory = Path(temporary.name)
            task_dir = await run_shielded(
                asyncio.to_thread(
                    shutil.copytree,
                    Path(task.data.task_dir).resolve(),
                    directory / "task",
                )
            )
            environment = task_dir / "environment"
            project_dir = str(environment)
            if host is not None:
                atexit.register(cleanup)
                stack.push_async_callback(host.stop_and_wait)
                await host.start()
                if isinstance(config, PrimeConfig):
                    install = await host.run(
                        [
                            "sh",
                            "-c",
                            (
                                "export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && "
                                "apt-get install -y -qq --no-install-recommends docker.io docker-cli docker-compose iptables "
                                "> /tmp/docker-install.log 2>&1 || { tail -40 /tmp/docker-install.log; exit 1; }"
                            ),
                        ],
                        {},
                    )
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
                project_dir = "/harbor/task/environment"
                await host.write(
                    "/harbor/task.tar.gz",
                    pack_dir_to_bytes(task_dir, compress=True).getvalue(),
                )
                staged = await run_host(
                    "sh",
                    "-c",
                    remote_unpack_command("/harbor/task.tar.gz", "/harbor/task"),
                )
                if staged.exit_code:
                    raise SandboxError(
                        f"Compose environment staging failed: {staged.stderr}"
                    )
            authored = yaml.safe_load((environment / "docker-compose.yaml").read_text())
            services = authored["services"]
            for kind in ("volumes", "networks"):
                if any(
                    value and (value.get("name") or value.get("external"))
                    for value in authored.get(kind, {}).values()
                ):
                    raise SandboxError(f"Compose {kind} must use project-scoped names")
            if any(service.get("container_name") for service in services.values()):
                raise SandboxError(
                    "Remove container_name so Compose can name each rollout's services"
                )
            owner = "main"
            seen: set[str] = set()
            while (
                services.get(owner, {}).get("network_mode", "").startswith("service:")
            ):
                if owner in seen:
                    raise SandboxError("Cyclic Compose network_mode service chain")
                seen.add(owner)
                owner = services[owner]["network_mode"].split(":", 1)[1]
            main = config.model_dump(
                include={"image", "workdir"}, exclude_defaults=True, exclude_none=True
            )
            if task.data.image is not None:
                main["image"] = config.image
            if "workdir" in main:
                main["working_dir"] = main.pop("workdir")
            if host is None:
                if config.cpu is not None:
                    main["cpus"] = config.cpu
                if config.memory is not None:
                    main["mem_limit"] = f"{config.memory}g"
            overlay = {"services": {"main": main}}
            if host is None and sys.platform != "linux":
                overlay["services"].setdefault(owner, {})["extra_hosts"] = {
                    "host.docker.internal": "host-gateway"
                }
            override = directory / "main.json"
            override.write_text(json.dumps(overlay))
            base = yaml.safe_load(COMPOSE_PREBUILT_PATH.read_text())
            if "image" in services["main"] or "build" in services["main"]:
                # A template default must not replace an authored image or skip its build.
                base["services"]["main"].pop("image", None)
            base_file = directory / "base.json"
            base_file.write_text(json.dumps(base))
            paths = [
                base_file,
                environment / "docker-compose.yaml",
                override,
                write_env_compose_file(directory / "env.json", task.runtime_env()),
            ]
            if host is not None:
                for path in paths:
                    await host.write(f"/harbor/{path.name}", path.read_bytes())
                paths = [Path("/harbor") / path.name for path in paths]
            compose_argv = [
                "docker",
                "compose",
                "--project-name",
                name,
                "--project-directory",
                project_dir,
                "--env-file",
                f"{project_dir}/.env"
                if (environment / ".env").is_file()
                else "/dev/null",
                *(arg for path in paths for arg in ("-f", str(path))),
            ]
            compose_env.update(
                ComposeInfraEnvVars(
                    main_image_name=name,
                    context_dir=project_dir,
                    prebuilt_image_name=config.image,
                ).to_env_dict()
            )
            # Validate authored services after interpolation, before adding our callback port.
            rendered = json.loads(await compose("config", "--format", "json"))
            for service in rendered["services"].values():
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
            if any(
                port.get("published") not in (None, "", 0, "0")
                for service in rendered["services"].values()
                for port in service.get("ports", [])
            ):
                raise SandboxError(
                    "Compose ports must use dynamically assigned host ports"
                )
            # Port publication belongs to the service that owns main's network namespace.
            if host is None or isinstance(config, ModalConfig):
                overlay["services"].setdefault(owner, {})["ports"] = [
                    f"127.0.0.1::{SERVICE_PORT}"
                    if host is None
                    else f"{SERVICE_PORT}:{SERVICE_PORT}"
                ]
                override.write_text(json.dumps(overlay))
                if host is not None:
                    await host.write(f"/harbor/{override.name}", override.read_bytes())
            if host is None:
                atexit.register(cleanup)
                stack.push_async_callback(asyncio.to_thread, cleanup)
            # Cancellation kills the local CLI; the rollout removes its project or VM.
            await compose("up", "--detach", "--wait")
            containers = (await compose("ps", "--all", "--quiet")).split()
            inspected = await run_host(
                "docker",
                "inspect",
                "--format",
                '{{.Id}} {{index .Config.Labels "com.docker.compose.service"}}',
                *containers,
            )
            if inspected.exit_code:
                raise SandboxError(
                    f"Compose container inspection failed: {inspected.stderr}"
                )
            service_url = None
            if host is None:
                published = await compose("port", owner, str(SERVICE_PORT))
                endpoint = next(
                    address
                    for address in published.splitlines()
                    if address.startswith("127.0.0.1:")
                )
                service_url = f"http://{endpoint}"
            for container in inspected.stdout.splitlines():
                container_id, name = container.split()
                if name in runtimes:
                    raise SandboxError(
                        f"Harbor Compose requires one container per service: {name}"
                    )
                runtimes[name] = await stack.enter_async_context(
                    DockerRuntime.attach(
                        config,
                        container_id,
                        host=host,
                        service_url=service_url if name == "main" else None,
                    )
                )

            if "main" not in runtimes:
                raise SandboxError("Harbor Compose requires exactly one main container")
        yield runtimes, partial(compose, "stop", "main", timeout=60)
    finally:
        await run_shielded(stack.aclose())
        temporary.cleanup()
        atexit.unregister(cleanup)
