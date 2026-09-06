"""Compose owns the project; DockerRuntime executes in its main container."""

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
from verifiers.v1.runtimes import DockerConfig, DockerRuntime, Runtime, RuntimeConfig
from verifiers.v1.runtimes.base import SERVICE_PORT, register
from verifiers.v1.runtimes.container import cli
from verifiers.v1.runtimes.docker.egress import EgressProxy, NetworkPolicy
from verifiers.v1.tasksets.harbor.taskset import HarborTask
from verifiers.v1.utils.aio import run_shielded


class HarborComposeRuntime(DockerRuntime):
    def __init__(self, config: DockerConfig, task: HarborTask):
        if config.network_restricted or config.gpu:
            raise ValueError("This Compose adapter supports public-network CPU tasks")
        super().__init__(config)
        self.task = task
        self._temporary = tempfile.TemporaryDirectory(prefix="vf-harbor-")
        self._compose_argv: list[str] = []
        self._compose_env: dict[str, str] = {}
        self._created = False

    async def _compose(self, *args: str) -> str:
        result = await cli(*self._compose_argv, *args, env=self._compose_env)
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

        environment = Path(self.task.data.task_dir) / "environment"
        directory = Path(self._temporary.name)
        services = yaml.safe_load((environment / "docker-compose.yaml").read_text())[
            "services"
        ]
        owner = "main"
        while services.get(owner, {}).get("network_mode", "").startswith("service:"):
            owner = services[owner]["network_mode"].split(":", 1)[1]
        main: dict[str, object] = {
            "image": self.config.image,
            "working_dir": self.config.workdir,
        }
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
        env_file = write_env_compose_file(directory / "env.json", self.env)
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
                    COMPOSE_PREBUILT_PATH,
                    environment / "docker-compose.yaml",
                    override,
                    env_file,
                )
                for arg in ("-f", str(path))
            ),
        ]
        self._compose_env = {
            **self.env,
            **ComposeInfraEnvVars(
                main_image_name=self.name,
                context_dir=str(environment),
                prebuilt_image_name=self.config.image,
            ).to_env_dict(),
        }
        await self._compose("config", "--quiet")
        self._created = True
        # Finish an interrupted Compose launch before deleting its partial project.
        await run_shielded(self._compose("up", "--detach", "--wait"))
        self._container = (
            await self._compose("ps", "--all", "--quiet", "main")
        ).strip()
        self.info.id = self._container
        inspected = await cli(
            "docker", "inspect", "--format", "{{json .Config.Env}}", self._container
        )
        if inspected.exit_code:
            raise SandboxError(
                f"Compose container inspection failed: {inspected.stderr}"
            )
        self._image_env = dict(
            entry.split("=", 1) for entry in json.loads(inspected.stdout) or []
        )
        published = await self._compose("port", owner, str(SERVICE_PORT))
        self._service_url = f"http://{published.strip()}"
        self._proxy = EgressProxy(NetworkPolicy(NetworkPolicyConfig(), []))
        if sys.platform == "linux":
            await self._proxy.start(listener=await self._container_listener())
        else:
            await self._proxy.start("127.0.0.1")

    def cleanup(self) -> None:
        if self._stopped:
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
        self._stopped = True
        self._temporary.cleanup()


@asynccontextmanager
async def harbor_compose_runtime(
    config: RuntimeConfig, task: HarborTask
) -> AsyncIterator[Runtime]:
    if not isinstance(config, DockerConfig):
        raise TypeError("Harbor Compose currently requires local Docker")
    runtime = HarborComposeRuntime(config, task)
    runtime.env = task.runtime_env()
    register(runtime)
    try:
        await runtime.start()
        yield runtime
    finally:
        await runtime.stop()
