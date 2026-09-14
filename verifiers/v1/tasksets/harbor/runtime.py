"""Compose owns the project; DockerRuntime executes in its main container."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import DockerConfig, DockerRuntime, Runtime, RuntimeConfig
from verifiers.v1.runtimes.base import SERVICE_PORT
from verifiers.v1.runtimes.container import cli
from verifiers.v1.runtimes.docker.egress import EgressProxy, NetworkPolicy
from verifiers.v1.tasksets.harbor.taskset import HarborTask


class HarborComposeRuntime(DockerRuntime):
    def __init__(
        self,
        config: DockerConfig,
        task: HarborTask,
        *,
        setup_timeout: float | None = None,
    ):
        if config.network_restricted or config.gpu:
            raise ValueError("This Compose adapter supports public-network CPU tasks")
        super().__init__(config)
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

    async def _compose(self, *args: str) -> str:
        result = await cli(*self._compose_argv, *args, env=self._compose_env)
        if result.exit_code:
            raise SandboxError(
                f"Compose {args[0]} failed: {result.stderr or result.stdout}"
            )
        return result.stdout

    async def start(self) -> None:
        async with asyncio.timeout(self._setup_timeout):
            await self._start()

    async def _start(self) -> None:
        import yaml
        from harbor.environments.docker import (
            COMPOSE_PREBUILT_PATH,
            write_env_compose_file,
        )
        from harbor.environments.docker.compose_env import ComposeInfraEnvVars

        environment = Path(self.task.data.task_dir).resolve() / "environment"
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
        if services.get(owner, {}).get("network_mode") == "host":
            raise SandboxError("Harbor Compose requires an isolated service network")
        main: dict[str, object] = dict(self._main_overrides)
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
        base_file = directory / "base.json"
        base_file.write_text(json.dumps(base))
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
                    base_file,
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
        # The CLI is killed on cancellation; the rollout then removes the project.
        await self._compose("up", "--detach", "--wait")
        containers = (await self._compose("ps", "--all", "--quiet", "main")).split()
        if len(containers) != 1:
            raise SandboxError("Harbor Compose requires exactly one main container")
        self._container = containers[0]
        self.info.id = self._container
        inspected = await cli(
            "docker", "inspect", "--format", "{{json .Config}}", self._container
        )
        if inspected.exit_code:
            raise SandboxError(
                f"Compose container inspection failed: {inspected.stderr}"
            )
        container = json.loads(inspected.stdout)
        self._image_env = dict(entry.split("=", 1) for entry in container["Env"] or [])
        self.config = self.config.model_copy(
            update={
                "image": container["Image"],
                "workdir": container["WorkingDir"] or "/",
            }
        )
        self.info.image, self.info.workdir = self.config.image, self.config.workdir
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


def make_harbor_compose_runtime(
    config: RuntimeConfig, *, task: HarborTask, setup_timeout: float | None = None
) -> Runtime:
    if not isinstance(config, DockerConfig):
        raise TypeError("Harbor Compose currently requires local Docker")
    return HarborComposeRuntime(config, task, setup_timeout=setup_timeout)
