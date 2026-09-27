"""Harbor owns Compose projects; Verifiers borrows their service runtimes."""

import asyncio
import json
import os
import shutil
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
    Runtime,
    RuntimeConfig,
)
from verifiers.v1.runtimes.base import SERVICE_PORT
from verifiers.v1.tasksets.harbor.taskset import HarborTask
from verifiers.v1.utils.aio import run_shielded

# Compose interpolation must not inherit unrelated evaluator credentials.
DOCKER_ENV = (
    "PATH",
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)


@asynccontextmanager
async def compose_services(
    config: RuntimeConfig, task: HarborTask, *, setup_timeout: float | None = None
) -> AsyncIterator[tuple[dict[str, Runtime], Callable[[], Awaitable[None]]]]:
    if isinstance(config, (PrimeConfig, ModalConfig)):
        from verifiers.v1.tasksets.harbor.runtime import cloud_services

        async with cloud_services(
            config, task, setup_timeout=setup_timeout
        ) as services:
            yield services
        return
    if not isinstance(config, DockerConfig):
        raise TypeError("Harbor Compose requires Docker, Prime VM or Modal VM")
    if not task.config.trust_compose:
        raise ValueError(
            "Local Compose tasks can access host files and Docker privileges; only run trusted tasks with --env.taskset.task.trust-compose"
        )
    if config.gpu or config.network_restricted:
        raise ValueError(
            "Local Harbor Compose requires CPU tasks and public networking"
        )
    import yaml
    from harbor.environments.docker import COMPOSE_PREBUILT_PATH
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    class LocalEnvironment(DockerEnvironment):
        def _compose_env_vars(self, include_os_env=True):
            return {
                **{key: os.environ[key] for key in DOCKER_ENV if key in os.environ},
                **super()._compose_env_vars(include_os_env=False),
            }

    async with AsyncExitStack() as stack:
        directory = Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix="vf-harbor-"))
        )
        await run_shielded(
            asyncio.to_thread(
                shutil.copytree, Path(task.data.task_dir).resolve(), directory / "task"
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
            service.get("container_name") for service in authored["services"].values()
        ):
            raise SandboxError(
                "Remove container_name so Compose can name each rollout's services"
            )
        # A fallback image must never override an authored image or Dockerfile build.
        base = yaml.safe_load(COMPOSE_PREBUILT_PATH.read_text())
        if {"image", "build"} & authored["services"].get("main", {}).keys():
            del base["services"]["main"]["image"]
        base_path = directory / "base.json"
        base_path.write_text(json.dumps(base))
        paths = TrialPaths(directory / "trial")
        paths.mkdir()
        environment = LocalEnvironment(
            environment_dir=directory / "task/environment",
            environment_name="verifiers",
            session_id=f"vf-{uuid.uuid4().hex}",
            trial_paths=paths,
            task_env_config=EnvironmentConfig(docker_image=config.image),
            persistent_env=task.runtime_env(),
            mounts=[],
        )
        environment._DOCKER_COMPOSE_PREBUILT_PATH = base_path
        # Use Harbor's interpolation and file ordering for validation too.
        environment._use_prebuilt = True
        resolved = await environment._run_docker_compose_command(
            ["config", "--format", "json"]
        )
        services = json.loads(resolved.stdout)["services"]
        for service in services.values():
            if service.get("network_mode") == "host":
                raise SandboxError(
                    "Harbor Compose requires an isolated service network"
                )
            if service.get("gpus") or service.get("deploy", {}).get(
                "resources", {}
            ).get("reservations", {}).get("devices"):
                raise SandboxError("Harbor Compose currently supports CPU tasks")
            if (
                service.get("scale", 1) != 1
                or service.get("deploy", {}).get("replicas", 1) != 1
            ):
                raise SandboxError("Harbor Compose requires one container per service")
            if any(
                port.get("published") not in (None, "", 0, "0")
                for port in service.get("ports", [])
            ):
                raise SandboxError(
                    "Compose ports must use dynamically assigned host ports"
                )
        owner = "main"
        while (mode := services[owner].get("network_mode", "")).startswith("service:"):
            owner = mode.removeprefix("service:")
        main = {}
        if task.data.image is not None:
            main["image"] = config.image
        if config.workdir is not None:
            main["working_dir"] = config.workdir
        if config.cpu is not None:
            main["cpus"] = config.cpu
        if config.memory is not None:
            main["mem_limit"] = f"{config.memory}g"
        overlay = {"services": {"main": main}}
        publish = overlay["services"].setdefault(owner, {})
        publish["ports"] = [f"127.0.0.1::{SERVICE_PORT}"]
        if sys.platform != "linux":
            publish["extra_hosts"] = {"host.docker.internal": "host-gateway"}
        overlay_path = directory / "runtime.json"
        overlay_path.write_text(json.dumps(overlay))
        environment.extra_docker_compose_paths.append(overlay_path)
        try:
            async with asyncio.timeout(setup_timeout):
                await environment.start(force_build=False)
                containers = await environment._run_docker_compose_command(
                    ["ps", "--all", "--format", "{{.ID}} {{.Service}}"]
                )
                published = await environment._run_docker_compose_command(
                    ["port", owner, str(SERVICE_PORT)]
                )
                service_url = "http://" + next(
                    address
                    for address in published.stdout.splitlines()
                    if address.startswith("127.0.0.1:")
                )
                runtimes: dict[str, Runtime] = {}
                for line in containers.stdout.splitlines():
                    container, service = line.split()
                    runtimes[service] = await stack.enter_async_context(
                        DockerRuntime.attach(
                            config,
                            container,
                            service_url=service_url if service == "main" else None,
                        )
                    )
            yield runtimes, partial(environment.stop_service, "main")
        finally:
            await run_shielded(environment.stop(delete=True))
