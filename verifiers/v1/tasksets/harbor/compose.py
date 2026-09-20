"""Harbor owns the Compose project and lends its main container to the agent."""

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
from verifiers.v1.runtimes import DockerConfig, DockerRuntime
from verifiers.v1.runtimes.base import SERVICE_PORT
from verifiers.v1.runtimes.container import cli
from verifiers.v1.tasksets.harbor.taskset import HarborTask
from verifiers.v1.utils.aio import run_shielded


@asynccontextmanager
async def compose_services(
    config: DockerConfig,
    task: HarborTask,
    *,
    trust_compose: bool = False,
    setup_timeout: float | None = None,
) -> AsyncIterator[tuple[dict[str, DockerRuntime], Callable[[], Awaitable[str]]]]:
    """Own one Compose attempt and lend its services until the context exits."""
    if not trust_compose:
        raise ValueError(
            "Local Compose tasks can access host files and Docker privileges; "
            "only run trusted tasks with --env.trust-compose"
        )
    if config.network_restricted or config.gpu:
        raise ValueError("This Compose adapter supports public-network CPU tasks")
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
        if key in os.environ
    }

    async def compose(*args: str, timeout: float | None = None) -> str:
        async with asyncio.timeout(timeout):
            result = await cli(*compose_argv, *args, env=compose_env)
        if result.exit_code:
            raise SandboxError(
                f"Compose {args[0]} failed: {result.stderr or result.stdout}"
            )
        return result.stdout

    def cleanup() -> None:
        # Keep the files and atexit backstop if teardown fails so cleanup can retry.
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

            directory = Path(temporary.name)
            task_dir = await run_shielded(
                asyncio.to_thread(
                    shutil.copytree,
                    Path(task.data.task_dir).resolve(),
                    directory / "task",
                )
            )
            environment = task_dir / "environment"
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
            if services.get(owner, {}).get("network_mode") == "host":
                raise SandboxError(
                    "Harbor Compose requires an isolated service network"
                )
            main = config.model_dump(
                include={"image", "workdir"}, exclude_defaults=True, exclude_none=True
            )
            if task.data.image is not None:
                main["image"] = config.image
            if "workdir" in main:
                main["working_dir"] = main.pop("workdir")
            if config.cpu is not None:
                main["cpus"] = config.cpu
            if config.memory is not None:
                main["mem_limit"] = f"{config.memory}g"
            overlay = {"services": {"main": main}}
            overlay["services"].setdefault(owner, {})["ports"] = [
                f"127.0.0.1::{SERVICE_PORT}"
            ]
            if sys.platform != "linux":
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
            env_file = write_env_compose_file(
                directory / "env.json", task.runtime_env()
            )
            compose_argv = [
                "docker",
                "compose",
                "--project-name",
                name,
                "--project-directory",
                str(environment),
                "--env-file",
                str(environment / ".env")
                if (environment / ".env").is_file()
                else os.devnull,
                *(
                    arg
                    for path in (
                        base_file,
                        environment / "docker-compose.yaml",
                        override,
                        env_file,
                    )
                    for arg in ("-f", str(path))
                ),
            ]
            compose_env.update(
                ComposeInfraEnvVars(
                    main_image_name=name,
                    context_dir=str(environment),
                    prebuilt_image_name=config.image,
                ).to_env_dict()
            )
            rendered = json.loads(await compose("config", "--format", "json"))
            for service in rendered["services"].values():
                if service.get("gpus") or (
                    service.get("deploy", {})
                    .get("resources", {})
                    .get("reservations", {})
                    .get("devices")
                ):
                    raise SandboxError("Harbor Compose currently supports CPU tasks")
                if any(
                    port.get("published") not in (None, "", 0, "0")
                    for port in service.get("ports", [])
                ):
                    raise SandboxError(
                        "Compose ports must use dynamically assigned host ports"
                    )
            atexit.register(cleanup)
            stack.push_async_callback(asyncio.to_thread, cleanup)
            # The CLI is killed on cancellation; the rollout then removes the project.
            await compose("up", "--detach", "--wait")
            containers = (await compose("ps", "--all", "--quiet", "main")).split()
            if len(containers) != 1:
                raise SandboxError("Harbor Compose requires exactly one main container")
            published = await compose("port", owner, str(SERVICE_PORT))
            endpoint = next(
                address
                for address in published.splitlines()
                if address.startswith("127.0.0.1:")
            )
            runtime = await stack.enter_async_context(
                DockerRuntime.attach(
                    config, containers[0], service_url=f"http://{endpoint}"
                )
            )
        yield {"main": runtime}, partial(compose, "stop", "main", timeout=60)
    finally:
        await run_shielded(stack.aclose())
        temporary.cleanup()
        atexit.unregister(cleanup)
