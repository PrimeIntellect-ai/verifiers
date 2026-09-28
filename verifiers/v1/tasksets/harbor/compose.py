"""Harbor provisions services; Verifiers adapts their runtime and callback APIs."""

import asyncio
import json
import math
import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import Any

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
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.tasksets.harbor.runtime import HarborRuntime
from verifiers.v1.tasksets.harbor.taskset import HarborTask
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.prime import load_prime_config
from verifiers.v1.utils.scope import run_scope

DOCKER_ENV = (
    "PATH",
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)


def _provider(config, task):
    options = {"compose_env": {}}
    if isinstance(config, DockerConfig):
        from harbor.environments.docker.docker import DockerEnvironment

        if not task.config.trust_compose:
            raise ValueError(
                "Local Compose tasks require --env.taskset.task.trust-compose"
            )
        if config.network_restricted:
            raise ValueError("Local Harbor Compose requires public networking")
        options["compose_env"] = {
            key: os.environ[key] for key in DOCKER_ENV if key in os.environ
        }
        return DockerEnvironment, options
    options["region"] = config.region
    if isinstance(config, PrimeConfig):
        from harbor.environments.prime import PrimeEnvironment

        from verifiers.v1.runtimes.prime import BASE_LABELS

        options.update(
            timeout_minutes=-1,
            idle_timeout_minutes=max(1, math.ceil(config.idle_timeout / 60))
            if config.idle_timeout is not None
            else None,
            labels=list(dict.fromkeys([*BASE_LABELS, *config.labels, run_scope()])),
            team_id=os.environ.get("PRIME_TEAM_ID")
            or load_prime_config().get("team_id"),
        )
        if task.data.compose_host_image:
            options["compose_host_image"] = task.data.compose_host_image
        return PrimeEnvironment, options
    from harbor.environments.modal import ModalEnvironment

    if not config.network_access:
        raise ValueError("Harbor Compose on Modal requires network_access=True")
    options.update(modal_vm_runtime=True, encrypted_ports=[SERVICE_PORT])
    if task.data.compose_host_image:
        options["dind_image"] = task.data.compose_host_image
    return ModalEnvironment, options


def _validate_local(document):
    for service in document["services"].values():
        if service.get("container_name"):
            raise SandboxError("Compose container names must be project-scoped")
        if service.get("network_mode") == "host":
            raise SandboxError("Harbor Compose requires an isolated service network")
        if service.get("gpus") or service.get("deploy", {}).get("resources", {}).get(
            "reservations", {}
        ).get("devices"):
            raise SandboxError("Harbor Compose currently supports CPU tasks")
        if any(
            port.get("published") not in (None, "", 0, "0")
            for port in service.get("ports", [])
        ):
            raise SandboxError("Compose ports must use dynamically assigned host ports")


@asynccontextmanager
async def compose_services(
    config: RuntimeConfig, task: HarborTask, *, setup_timeout: float | None = None
) -> AsyncIterator[tuple[dict[str, Runtime], Callable[[], Awaitable[None]]]]:
    import yaml
    from harbor.environments.docker.compose_env import network_service
    from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
    from harbor.models.trial.paths import TrialPaths

    if not isinstance(config, (DockerConfig, PrimeConfig, ModalConfig)):
        raise TypeError("Harbor Compose requires Docker, Prime VM or Modal VM")
    if config.gpu:
        raise ValueError("Harbor Compose currently supports CPU tasks")
    provider, options = _provider(config, task)
    local = isinstance(config, DockerConfig)
    async with AsyncExitStack() as stack:
        directory = Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix="vf-harbor-"))
        )
        task_dir = Path(task.data.task_dir).resolve()
        if local:
            await run_shielded(
                asyncio.to_thread(shutil.copytree, task_dir, directory / "task")
            )
            task_dir = directory / "task"
        paths = TrialPaths(directory / "trial")
        paths.mkdir()
        main: dict[str, Any] = {}
        if task.data.image is not None:
            main["image"] = config.image
        if config.workdir is not None:
            main["working_dir"] = config.workdir
        if local:
            if config.cpu is not None:
                main["cpus"] = config.cpu
            if config.memory is not None:
                main["mem_limit"] = f"{config.memory}g"
        overlay = {"services": {"main": main}}
        overlay_path = directory / "runtime.json"
        overlay_path.write_text(json.dumps(overlay))
        environment = provider(
            environment_dir=task_dir / "environment",
            environment_name=task.data.name or "verifiers",
            session_id=f"vf-{uuid.uuid4().hex}",
            trial_paths=paths,
            task_env_config=EnvironmentConfig(
                docker_image=task.data.image or config.image,
                memory_mb=int(config.memory * 1024)
                if config.memory is not None
                else None,
                storage_mb=int(config.disk * 1024) if config.disk is not None else None,
            ),
            override_cpus=config.cpu,
            persistent_env=task.runtime_env(),
            mounts=[],
            network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
            phase_network_policies=[
                NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=[])
            ]
            if config.network_restricted
            else [],
            extra_docker_compose=[overlay_path],
            **options,
        )
        runtimes: dict[str, Runtime] = {}
        try:
            async with asyncio.timeout(setup_timeout):
                if local or isinstance(config, ModalConfig):
                    authored = yaml.safe_load(
                        (task_dir / "environment/docker-compose.yaml").read_text()
                    )
                    if local:
                        # Authored names are distinguished from Compose's generated names.
                        for kind in ("volumes", "networks"):
                            if any(
                                v and (v.get("name") or v.get("external"))
                                for v in authored.get(kind, {}).values()
                            ):
                                raise SandboxError(
                                    f"Compose {kind} must use project-scoped names"
                                )
                        resolved = await environment.compose_config()
                        _validate_local({"services": resolved["services"]})
                    else:
                        resolved = authored
                    owner = network_service(resolved["services"], "main")
                    publish = overlay["services"].setdefault(owner, {})
                    publish["ports"] = [
                        f"127.0.0.1::{SERVICE_PORT}"
                        if local
                        else f"{SERVICE_PORT}:{SERVICE_PORT}"
                    ]
                    if local and sys.platform != "linux":
                        publish["extra_hosts"] = {
                            "host.docker.internal": "host-gateway"
                        }
                    overlay_path.write_text(json.dumps(overlay))
                rate = (
                    (config.creates_per_min or 0) / 60
                    if isinstance(config, PrimeConfig)
                    else config.creates_per_sec
                    if isinstance(config, ModalConfig)
                    else None
                )
                async with (
                    creation_limiter(rate, f"{config.type}-sandbox", run_scope())
                    or nullcontext()
                ):
                    await environment.start(force_build=False)
                service_url = (
                    "http://" + await environment.service_port(owner, SERVICE_PORT)
                    if local
                    else None
                )
                for name, info in (await environment.services()).items():
                    if local:
                        runtime = await stack.enter_async_context(
                            DockerRuntime.attach(
                                config,
                                info["id"],
                                service_url=service_url if name == "main" else None,
                            )
                        )
                    else:
                        runtime = HarborRuntime(environment, name, config, info)
                    runtimes[name] = runtime
            yield runtimes, partial(environment.stop_service, "main")
        finally:
            for runtime in runtimes.values():
                runtime.stopped = True
            await run_shielded(environment.stop(delete=True))
