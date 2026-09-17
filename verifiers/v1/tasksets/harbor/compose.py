"""Harbor owns a Compose project locally or inside an existing provider runtime."""

import asyncio
import atexit
import json
import os
import subprocess
import sys
import tempfile
import uuid
from contextlib import AsyncExitStack
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


class ComposeProject:
    def __init__(
        self,
        config: DockerConfig | PrimeConfig | ModalConfig,
        task: HarborTask,
        *,
        setup_timeout: float | None = None,
    ):
        if config.gpu:
            raise ValueError("Harbor Compose currently supports CPU tasks")
        if isinstance(config, DockerConfig) and config.network_restricted:
            raise ValueError(
                "Harbor Compose on local Docker requires public networking"
            )
        self.config = config
        self.name = f"vf-{uuid.uuid4().hex}"
        self.env = dict(task.runtime_env())
        self._stack = AsyncExitStack()
        self._host: PrimeRuntime | ModalRuntime | None = None
        if isinstance(config, PrimeConfig):
            if not config.vm:
                raise ValueError("Harbor Compose on Prime requires vm=True")
            self._host = PrimeRuntime(
                config.model_copy(
                    update={"image": "python:3.11-slim-trixie", "workdir": "/"}
                )
            )
        elif isinstance(config, ModalConfig):
            if not config.network_access:
                raise ValueError("Harbor Compose on Modal requires network_access=True")
            self._host = ModalRuntime(
                config.model_copy(
                    update={"image": "docker:28.3.3-dind", "workdir": "/", "vm": True}
                )
            )
        self.task = task
        self._setup_timeout = setup_timeout
        self._main_overrides = config.model_dump(
            include={"image", "workdir"}, exclude_defaults=True, exclude_none=True
        )
        if task.data.image is not None:
            self._main_overrides["image"] = config.image
        self._temporary = tempfile.TemporaryDirectory(prefix="vf-harbor-")
        self._compose_argv: list[str] = []
        self._compose_env: dict[str, str] = {}
        self._created = False
        self._closed = False
        self.services: dict[str, DockerRuntime] = {}

    async def _run_host(
        self, *args: str, env: dict[str, str] | None = None
    ) -> ProgramResult:
        if self._host is not None:
            return await self._host.run(list(args), env or {})
        return await cli(*args, env=env)

    async def _compose(self, *args: str) -> str:
        result = await self._run_host(*self._compose_argv, *args, env=self._compose_env)
        if result.exit_code:
            raise SandboxError(
                f"Compose {args[0]} failed: {result.stderr or result.stdout}"
            )
        return result.stdout

    async def __aenter__(self) -> DockerRuntime:
        atexit.register(self.cleanup)
        self._stack.push_async_callback(self._teardown)
        try:
            async with asyncio.timeout(self._setup_timeout):
                await self._start()
        except BaseException:
            await run_shielded(self._stack.aclose())
            raise
        return self.services["main"]

    async def __aexit__(self, *exc) -> None:
        await run_shielded(self._stack.aclose())

    async def _start(self) -> None:
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

        environment = Path(self.task.data.task_dir).resolve() / "environment"
        project_dir = str(environment)
        if self._host is not None:
            await self._host.start()
            if isinstance(self.config, PrimeConfig):
                install = await self._host.run(
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
            await self._host.run_background(
                ["dockerd", "--host=unix:///var/run/docker.sock"],
                {},
                "/tmp/dockerd.log",
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
        if any(service.get("container_name") for service in services.values()):
            raise SandboxError(
                "Remove container_name so Compose can name each rollout's services"
            )
        owner = "main"
        seen: set[str] = set()
        while services.get(owner, {}).get("network_mode", "").startswith("service:"):
            if owner in seen:
                raise SandboxError("Cyclic Compose network_mode service chain")
            seen.add(owner)
            owner = services[owner]["network_mode"].split(":", 1)[1]
        if services.get(owner, {}).get("network_mode") == "host":
            raise SandboxError("Harbor Compose requires an isolated service network")
        main: dict[str, object] = dict(self._main_overrides)
        if "workdir" in main:
            main["working_dir"] = main.pop("workdir")
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
        self._compose_env = ComposeInfraEnvVars(
            main_image_name=self.name,
            context_dir=project_dir,
            prebuilt_image_name=self.config.image,
        ).to_env_dict()
        await self._compose("config", "--quiet")
        self._created = True
        # Cancellation kills the local CLI; the rollout removes its project or VM.
        await self._compose("up", "--detach", "--wait")
        containers = (await self._compose("ps", "--all", "--quiet")).split()
        inspected = await self._run_host("docker", "inspect", *containers)
        if inspected.exit_code:
            raise SandboxError(
                f"Compose container inspection failed: {inspected.stderr}"
            )
        containers = json.loads(inspected.stdout)
        if (
            sum(
                container["Config"]["Labels"]["com.docker.compose.service"] == "main"
                for container in containers
            )
            != 1
        ):
            raise SandboxError("Harbor Compose requires exactly one main container")
        service_url = None
        if self._host is None:
            published = await self._compose("port", owner, str(SERVICE_PORT))
            service_url = f"http://{published.strip()}"
        for container in containers:
            name = container["Config"]["Labels"]["com.docker.compose.service"]
            if name in self.services:
                raise SandboxError(
                    f"Harbor Compose requires one container per service: {name}"
                )
            self.services[name] = await self._stack.enter_async_context(
                DockerRuntime.attach(
                    self.config,
                    container["Id"],
                    host=self._host,
                    service_url=service_url if name == "main" else None,
                )
            )

    async def stop_service(self, name: str) -> None:
        async with asyncio.timeout(60):
            await self._compose("stop", name)

    async def _teardown(self) -> None:
        if self._host is not None:
            await self._host.stop_and_wait()
            self._closed = True
            self._temporary.cleanup()
            atexit.unregister(self.cleanup)
        else:
            await asyncio.to_thread(self.cleanup)

    def cleanup(self) -> None:
        if self._closed:
            return
        if self._host is not None:
            self._host.cleanup()
        elif self._created:
            subprocess.run(
                [*self._compose_argv, "down", "--volumes", "--remove-orphans"],
                env={**os.environ, **self._compose_env},
                capture_output=True,
                timeout=60,
                check=True,
            )
        self._closed = True
        self._temporary.cleanup()
        atexit.unregister(self.cleanup)
