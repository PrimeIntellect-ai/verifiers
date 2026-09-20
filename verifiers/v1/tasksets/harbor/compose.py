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
from contextlib import AsyncExitStack
from pathlib import Path

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import DockerConfig, DockerRuntime
from verifiers.v1.runtimes.base import SERVICE_PORT
from verifiers.v1.runtimes.container import cli
from verifiers.v1.tasksets.harbor.taskset import HarborTask
from verifiers.v1.utils.aio import run_shielded


class ComposeProject:
    def __init__(
        self,
        config: DockerConfig,
        task: HarborTask,
        *,
        setup_timeout: float | None = None,
    ):
        if config.network_restricted or config.gpu:
            raise ValueError("This Compose adapter supports public-network CPU tasks")
        self.config = config
        self.name = f"vf-{uuid.uuid4().hex}"
        self._stack = AsyncExitStack()
        self.task = task
        self._setup_timeout = setup_timeout
        self._temporary = tempfile.TemporaryDirectory(prefix="vf-harbor-")
        self._compose_argv: list[str] = []
        self._compose_env: dict[str, str] = {}
        self._created = False
        self._closed = False

    async def _compose(self, *args: str) -> str:
        result = await cli(*self._compose_argv, *args, env=self._compose_env)
        if result.exit_code:
            raise SandboxError(
                f"Compose {args[0]} failed: {result.stderr or result.stdout}"
            )
        return result.stdout

    async def __aenter__(self) -> DockerRuntime:
        atexit.register(self.cleanup)
        self._stack.push_async_callback(asyncio.to_thread, self.cleanup)
        try:
            async with asyncio.timeout(self._setup_timeout):
                await self._start()
        except BaseException:
            await run_shielded(self._stack.aclose())
            raise
        return self.runtime

    async def __aexit__(self, *exc) -> None:
        await run_shielded(self._stack.aclose())

    async def _start(self) -> None:
        import yaml
        from harbor.environments.docker import (
            COMPOSE_PREBUILT_PATH,
            write_env_compose_file,
        )
        from harbor.environments.docker.compose_env import ComposeInfraEnvVars

        directory = Path(self._temporary.name)
        environment = await run_shielded(
            asyncio.to_thread(
                shutil.copytree,
                Path(self.task.data.task_dir).resolve() / "environment",
                directory / "environment",
            )
        )
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
        main = self.config.model_dump(
            include={"image", "workdir"}, exclude_defaults=True, exclude_none=True
        )
        if self.task.data.image is not None:
            main["image"] = self.config.image
        if "workdir" in main:
            main["working_dir"] = main.pop("workdir")
        if self.config.cpu is not None:
            main["cpus"] = self.config.cpu
        if self.config.memory is not None:
            main["mem_limit"] = f"{self.config.memory}g"
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
            base["services"]["main"].pop("command", None)
        base_file = directory / "base.json"
        base_file.write_text(json.dumps(base))
        env_file = write_env_compose_file(
            directory / "env.json", self.task.runtime_env()
        )
        self._compose_argv = [
            "docker",
            "compose",
            "--project-name",
            self.name,
            "--project-directory",
            str(environment),
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
        self._compose_env = ComposeInfraEnvVars(
            main_image_name=self.name,
            context_dir=str(environment),
            prebuilt_image_name=self.config.image,
        ).to_env_dict()
        await self._compose("config", "--quiet")
        self._created = True
        # The CLI is killed on cancellation; the rollout then removes the project.
        await self._compose("up", "--detach", "--wait")
        containers = (await self._compose("ps", "--all", "--quiet", "main")).split()
        if len(containers) != 1:
            raise SandboxError("Harbor Compose requires exactly one main container")
        published = await self._compose("port", owner, str(SERVICE_PORT))
        endpoint = next(
            address
            for address in published.splitlines()
            if address.startswith("127.0.0.1:")
        )
        self.runtime = await self._stack.enter_async_context(
            DockerRuntime.attach(
                self.config, containers[0], service_url=f"http://{endpoint}"
            )
        )

    def cleanup(self) -> None:
        if self._closed:
            return
        if self._created:
            # Use the same checked operation on normal exit and the atexit backstop.
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
