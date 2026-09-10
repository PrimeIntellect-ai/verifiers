"""One Compose project owner using Docker execution locally or inside a VM."""

import asyncio
import copy
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    DockerConfig,
    DockerRuntime,
    ModalConfig,
    PrimeConfig,
    Runtime,
    RuntimeConfig,
)
from verifiers.v1.runtimes.base import SERVICE_PORT, register
from verifiers.v1.runtimes.docker.egress import EgressProxy, NetworkPolicy
from verifiers.v1.tasksets.harbor.taskset import HarborTask
from verifiers.v1.utils.aio import run_shielded


class HarborComposeRuntime(DockerRuntime):
    def __init__(
        self,
        config: DockerConfig | PrimeConfig | ModalConfig,
        task: HarborTask,
        *,
        host: Runtime | None = None,
    ):
        if config.gpu:
            raise ValueError("Harbor Compose currently supports CPU tasks")
        if isinstance(config, DockerConfig) and config.network_restricted:
            raise ValueError(
                "Harbor Compose on local Docker requires public networking"
            )
        super().__init__(config, host=host)
        self.task = task
        self._temporary = tempfile.TemporaryDirectory(prefix="vf-harbor-")
        self._compose_argv: list[str] = []
        self._compose_env: dict[str, str] = {}
        self._created = False
        self._owner: HarborComposeRuntime | None = None
        self._services: dict[str, HarborComposeRuntime] = {"main": self}

    async def _compose(self, *args: str) -> str:
        result = await self._run_host(*self._compose_argv, *args, env=self._compose_env)
        if result.exit_code:
            raise SandboxError(
                f"Compose {args[0]} failed: {result.stderr or result.stdout}"
            )
        return result.stdout

    async def start(self) -> None:
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

        if self._owner is not None:
            raise RuntimeError("Compose services are started by their project")
        environment = Path(self.task.data.task_dir).resolve() / "environment"
        project_dir = str(environment)
        if self._host is not None:
            await self._host.start()
            self.info = self._host.info.model_copy(
                update={"image": self.config.image, "workdir": self.config.workdir}
            )
            async with asyncio.timeout(60):
                while (await self._run_host("docker", "info")).exit_code:
                    await asyncio.sleep(1)
            project_dir = "/harbor/environment"
            await self._host.write(
                "/harbor/environment.tar.gz",
                pack_dir_to_bytes(environment, compress=True).getvalue(),
            )
            staged = await self._run_host(
                "sh",
                "-c",
                remote_unpack_command("/harbor/environment.tar.gz", project_dir),
            )
            if staged.exit_code:
                raise SandboxError(
                    f"Compose environment staging failed: {staged.stderr}"
                )
        directory = Path(self._temporary.name)
        services = yaml.safe_load((environment / "docker-compose.yaml").read_text())[
            "services"
        ]
        owner = "main"
        seen: set[str] = set()
        while services.get(owner, {}).get("network_mode", "").startswith("service:"):
            if owner in seen:
                raise SandboxError("Cyclic Compose network_mode service chain")
            seen.add(owner)
            owner = services[owner]["network_mode"].split(":", 1)[1]
        main: dict[str, object] = {
            "image": self.config.image,
            "working_dir": self.config.workdir,
        }
        if self._host is None:
            if self.config.cpu is not None:
                main["cpus"] = self.config.cpu
            if self.config.memory is not None:
                main["mem_limit"] = f"{self.config.memory}g"
        overlay = {"services": {"main": main}}
        # Port publication belongs to the service that owns main's network namespace.
        if self._host is None or isinstance(self.config, ModalConfig):
            overlay["services"].setdefault(owner, {})["ports"] = [
                f"127.0.0.1::{SERVICE_PORT}"
                if self._host is None
                else f"{SERVICE_PORT}:{SERVICE_PORT}"
            ]
        if self._host is None and sys.platform != "linux":
            overlay["services"].setdefault(owner, {})["extra_hosts"] = {
                "host.docker.internal": "host-gateway"
            }
        override = directory / "main.json"
        override.write_text(json.dumps(overlay))
        paths = [
            COMPOSE_PREBUILT_PATH,
            environment / "docker-compose.yaml",
            override,
            write_env_compose_file(directory / "env.json", self.env),
        ]
        if self._host is not None:
            for index, path in enumerate(paths):
                target = Path(project_dir if index == 1 else "/harbor") / path.name
                if index != 1:
                    await self._host.write(str(target), path.read_bytes())
                paths[index] = target
        self._compose_argv = [
            "docker",
            "compose",
            "--project-name",
            self.name,
            "--project-directory",
            project_dir,
            *(arg for path in paths for arg in ("-f", str(path))),
        ]
        self._compose_env = {
            **self.env,
            **ComposeInfraEnvVars(
                main_image_name=self.name,
                context_dir=project_dir,
                prebuilt_image_name=self.config.image,
            ).to_env_dict(),
        }
        await self._compose("config", "--quiet")
        self._created = True
        # Finish an interrupted Compose launch before deleting its partial project.
        await run_shielded(self._compose("up", "--detach", "--wait"))
        containers = (await self._compose("ps", "--all", "--quiet")).split()
        inspected = await self._run_host("docker", "inspect", *containers)
        if inspected.exit_code:
            raise SandboxError(
                f"Compose container inspection failed: {inspected.stderr}"
            )
        for container in json.loads(inspected.stdout):
            name = container["Config"]["Labels"]["com.docker.compose.service"]
            service = self if name == "main" else copy.copy(self)
            service._container = container["Id"]
            service._image_env = dict(
                entry.split("=", 1) for entry in container["Config"]["Env"] or []
            )
            if name != "main":
                service._owner = self
                service.env = {}
                service._uv_interpreters = {}
                service._uv_script_locks = {}
                service._setup_claimed = False
                service.config = self.config.model_copy(
                    update={"workdir": container["Config"]["WorkingDir"] or "/"}
                )
            service.info = self.info.model_copy(
                update={"borrowed": True, "workdir": service.config.workdir}
            )
            if self._host is None:
                service.info.id = service._container
            self._services[name] = service
        if self._host is None:
            published = await self._compose("port", owner, str(SERVICE_PORT))
            self._service_url = f"http://{published.strip()}"
            self._proxy = EgressProxy(NetworkPolicy(NetworkPolicyConfig(), []))
            if sys.platform == "linux":
                await self._proxy.start(listener=await self._container_listener())
            else:
                await self._proxy.start("127.0.0.1")

    def service(self, name: str) -> Runtime:
        return self._services[name]

    async def stop_service(self, name: str) -> None:
        async with asyncio.timeout(60):
            await self._compose("stop", name)

    def host_url(self, url: str) -> str:
        return (
            self._host.host_url(url)
            if self._host is not None
            else super().host_url(url)
        )

    async def prepare_execution(self, routes: list[str] | None) -> None:
        if self._host is not None:
            await self._host.prepare_execution(routes)
        else:
            await super().prepare_execution(routes)

    async def expose(self, port: int) -> str:
        if self._owner is not None:
            raise SandboxError("Only the main Compose service publishes a runtime port")
        url = (
            await self._host.expose(port)
            if self._host is not None
            else await super().expose(port)
        )
        assert url is not None
        return url

    async def teardown(self) -> None:
        if self._owner is not None:
            return
        if self._host is not None:
            await self._host.stop()
            self._stopped = True
            self._temporary.cleanup()
        else:
            await super().teardown()

    def cleanup(self) -> None:
        if self._stopped or self._owner is not None:
            return
        if self._host is not None:
            self._host.cleanup()
        elif self._created:
            # Use the same checked operation on normal exit and the atexit backstop.
            subprocess.run(
                [*self._compose_argv, "down", "--volumes", "--remove-orphans"],
                env={**os.environ, **self._compose_env},
                capture_output=True,
                timeout=60,
                check=True,
            )
        self._stopped = True
        self._temporary.cleanup()


class _RemoteComposeRuntime(HarborComposeRuntime):
    is_local = False


@asynccontextmanager
async def harbor_compose_runtime(
    config: RuntimeConfig, task: HarborTask
) -> AsyncIterator[Runtime]:
    host = None
    if isinstance(config, PrimeConfig):
        from verifiers.v1.tasksets.harbor.prime import PrimeComposeVM

        if not config.vm:
            raise ValueError("Harbor Compose on Prime requires vm=True")
        host = PrimeComposeVM(
            # Trixie supplies Docker's separate CLI and Compose v2 packages.
            config.model_copy(
                update={"image": "python:3.11-slim-trixie", "workdir": "/"}
            )
        )
    elif isinstance(config, ModalConfig):
        from verifiers.v1.tasksets.harbor.modal import ModalComposeVM

        if not config.network_access:
            raise ValueError("Harbor Compose on Modal requires network_access=True")
        host = ModalComposeVM(
            config.model_copy(update={"image": "docker:28.3.3-dind", "workdir": "/"})
        )
    elif not isinstance(config, DockerConfig):
        raise TypeError("Harbor Compose requires Docker, Prime VM or Modal VM")
    if host is not None:
        register(host)
    runtime_cls = _RemoteComposeRuntime if host is not None else HarborComposeRuntime
    runtime = runtime_cls(config, task, host=host)
    runtime.env = task.runtime_env()
    register(runtime)
    try:
        await runtime.start()
        yield runtime
    finally:
        await runtime.stop()
