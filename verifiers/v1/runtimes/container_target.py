"""An owned task container inside a trusted harness runtime."""

import asyncio
import subprocess
import uuid

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import RuntimeBackend
from verifiers.v1.runtimes.docker import DockerBackend, DockerConfig, DockerRuntimeInfo
from verifiers.v1.utils.aio import run_shielded


class ContainerTarget(DockerBackend):
    def __init__(
        self,
        config: DockerConfig,
        name: str | None = None,
        *,
        host_backend: RuntimeBackend | None = None,
    ):
        super().__init__(config, name, host_backend=host_backend)
        self.info = DockerRuntimeInfo(**self.config.model_dump())

    async def start(self) -> None:
        result = await self._run_host("sh", "-c", "command -v docker")
        if result.exit_code:
            result = await self._run_host(
                "sh",
                "-c",
                "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io",
            )
            if result.exit_code:
                raise SandboxError(
                    "split execution requires Docker; install it in the harness image or use a Debian-based image"
                )
        result = await self._run_host("docker", "info")
        if result.exit_code:
            if self._host_backend is None:
                raise SandboxError("Docker daemon is unavailable")
            await self._host_backend.bind().run_background(
                ["dockerd"], {}, "/tmp/vf-dockerd.log"
            )
            for _ in range(30):
                result = await self._run_host("docker", "info")
                if not result.exit_code:
                    break
                await asyncio.sleep(1)
            else:
                raise SandboxError(
                    "Docker daemon did not start; see /tmp/vf-dockerd.log in the harness runtime"
                )
        if self.config.gpu:
            raise ValueError("container execution does not yet support GPU limits")
        if self.network_restricted and self.config.allow:
            raise ValueError(
                "nested task networking supports unrestricted access or an empty allowlist"
            )
        engine_name = f"{self.name}-{uuid.uuid4().hex}"
        argv = [
            "docker",
            "create",
            "--name",
            engine_name,
            "--init",
            "--pids-limit",
            "512",
            "--log-driver",
            "local",
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=2",
            "--cap-drop",
            "NET_RAW",
            *self._label_args,
        ]
        if self.config.cpu is not None:
            argv += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            argv += [
                "--memory",
                f"{self.config.memory}g",
                "--memory-swap",
                f"{self.config.memory}g",
            ]
        argv += [
            "--entrypoint",
            "/bin/sh",
            self.config.image,
            "-c",
            "trap 'exit 0' TERM; while :; do sleep 3600 & wait $!; done",
        ]

        async def create():
            result = await self._run_host(*argv)
            if result.exit_code:
                raise SandboxError(f"task container creation failed: {result.stderr}")
            self._container = self.info.id = result.stdout.strip()

        # Capture ownership before cancellation can trigger teardown.
        await run_shielded(create())
        result = await self._run_host("docker", "start", self._container)
        if result.exit_code:
            raise SandboxError(f"task container startup failed: {result.stderr}")
        await self._initialize_environment()

    async def prepare_execution(self, routes: list[str] | None) -> None:
        if routes is None:
            if self._cut:
                result = await self._run_host(
                    "docker", "network", "connect", "bridge", self._container
                )
                if result.exit_code:
                    raise SandboxError(
                        f"cannot reopen verifier networking: {result.stderr}"
                    )
                self._cut = False
            return
        if self.network_restricted and not self._cut:
            result = await self._run_host(
                "docker", "network", "disconnect", "bridge", self._container
            )
            if result.exit_code:
                raise SandboxError(f"task network isolation failed: {result.stderr}")
            self._cut = True

    def _exec(
        self, env: dict[str, str], *, stdin: bool = False, user: str | None = None
    ) -> list[str]:
        return [
            "docker",
            *(
                ["--host", self._host_backend.env["DOCKER_HOST"]]
                if self._host_backend and "DOCKER_HOST" in self._host_backend.env
                else []
            ),
            "exec",
            *(["-i"] if stdin else []),
            *(
                part
                for key, value in env.items()
                for part in ("--env", f"{key}={value}")
            ),
            *(["--user", user] if user else []),
            "--workdir",
            self.config.workdir,
            self._container,
        ]

    async def expose(self, port: int) -> str:
        raise SandboxError("nested task runtimes do not expose services")

    async def teardown(self) -> None:
        if self.info.borrowed:
            return
        if self._container is not None and not self._stopped:
            result = await self._run_host("docker", "rm", "--force", self._container)
            if result.exit_code:
                raise SandboxError(f"task container cleanup failed: {result.stderr}")
            self._stopped = True

    def cleanup(self) -> None:
        # Remote resources disappear with their registered owning VM. Local
        # targets need their own exit backstop when cancellation skips async close.
        if self.info.borrowed or self._container is None or self._stopped:
            return
        if self._host_backend is not None and not self._host_backend.is_local:
            return
        result = subprocess.run(
            ["docker", "rm", "--force", self._container],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            self._stopped = True
