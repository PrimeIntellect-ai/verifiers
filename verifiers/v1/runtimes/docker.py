"""Local Docker runtime."""

import asyncio
import contextlib
import logging
import shlex
import subprocess
import sys
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    parse_gpu,
)

logger = logging.getLogger(__name__)


class DockerConfig(BaseConfig):
    type: Literal["docker"] = "docker"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    network_access: bool = True
    """Whether the container can access the network. A harness runtime needs this to
    reach model interception; disable it only for isolated runtimes such as verifiers."""
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float | None = None
    """Pin the container to this many CPU cores (docker `--cpus`). None = unlimited."""
    memory: float | None = None
    """Hard memory limit in GB (docker `--memory`). None = unlimited."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "2" (docker `--gpus` uses the count; needs the nvidia
    container toolkit). None = none."""
    disk: float | None = None
    """Advisory disk request in GB. Docker has no portable per-container size limit, so
    this is accepted (so a task can declare it without a warning) but not enforced."""


class DockerRuntimeInfo(DockerConfig, BaseRuntimeInfo):
    pass


async def docker(*args: str) -> ProgramResult:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return ProgramResult(
        exit_code=proc.returncode or 0,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )


class DockerRuntime(Runtime):
    def __init__(self, config: DockerConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = DockerRuntimeInfo(**config.model_dump())
        self._container: str | None = None  # our `--name` (used for exec/rm)
        self._stopped = False

    async def start(self) -> None:
        try:
            version = await docker("version", "--format", "{{.Server.Version}}")
        except FileNotFoundError as e:
            raise RuntimeError(
                "docker runtime selected but the `docker` CLI is not installed"
            ) from e
        if version.exit_code != 0:
            detail = (version.stderr or version.stdout).strip()
            hint = ""
            if "permission denied" in detail.lower():
                hint = (
                    "\nYour user isn't in the `docker` group. Either run the command "
                    'under `sg docker -c "..."`, or add yourself with '
                    "`sudo usermod -aG docker $USER` and start a new login shell."
                )
            raise RuntimeError(
                f"docker runtime selected but the Docker daemon is not reachable: {detail}{hint}"
            )
        self._container = self.name
        limits: list[str] = []
        if self.config.cpu is not None:
            limits += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            limits += ["--memory", f"{self.config.memory}g"]
        _, gpu_count = parse_gpu(self.config.gpu)
        if gpu_count:
            limits += ["--gpus", str(gpu_count)]
        run = await docker(
            "run",
            "--detach",
            "--network",
            "host" if self.config.network_access else "none",
            *limits,
            "--workdir",
            self.config.workdir,
            "--entrypoint",
            "sleep",
            "--name",
            self._container,
            self.config.image,
            "infinity",
        )
        if run.exit_code != 0:
            raise SandboxError(f"docker run failed: {run.stderr.strip()}")
        self.info.id = run.stdout.strip()[
            :12
        ]  # `docker run -d` prints the container id
        logger.info(
            "docker: started container %s (image=%s)",
            self._container,
            self.config.image,
        )

    def host_url(self, url: str) -> str:
        # Docker Desktop (macOS/Windows) runs containers in a VM, so `--network host`
        # doesn't reach the host's loopback; `host.docker.internal` does.
        host = urlsplit(url).hostname
        if sys.platform != "linux" and host in ("127.0.0.1", "localhost"):
            return url.replace(host, "host.docker.internal", 1)
        return url

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        env_args = [arg for k, v in env.items() for arg in ("--env", f"{k}={v}")]
        return await docker(
            "exec", *env_args, "--workdir", self.config.workdir, self._container, *argv
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        env_args = [arg for k, v in env.items() for arg in ("--env", f"{k}={v}")]
        inner = f"{' '.join(shlex.quote(a) for a in argv)} > {shlex.quote(log)} 2>&1"
        run = await docker(
            "exec",
            "--detach",
            *env_args,
            "--workdir",
            self.config.workdir,
            self._container,
            "sh",
            "-c",
            inner,
        )  # detached → lives in the container until it's removed in stop()
        if run.exit_code != 0:
            raise SandboxError(f"docker exec -d failed: {run.stderr.strip()}")

    async def read(self, path: str) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "--workdir",
            self.config.workdir,
            self._container,
            "cat",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise SandboxError(
                f"read {path!r}: {stderr.decode(errors='replace').strip()}"
            )
        return stdout

    async def read_bounded(self, path: str, max_bytes: int) -> bytes:
        self._validate_read_limit(max_bytes)
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "--workdir",
            self.config.workdir,
            self._container,
            "cat",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            try:
                stdout = await proc.stdout.readexactly(max_bytes + 1)
            except asyncio.IncompleteReadError as exc:
                stdout = exc.partial
            else:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
                raise SandboxError(f"read {path!r}: exceeds the {max_bytes} byte limit")
            await proc.wait()
            if proc.returncode != 0:
                raise SandboxError(f"read {path!r}: docker exec failed")
            return stdout
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()

    async def write(self, path: str, data: bytes) -> None:
        parent = shlex.quote(str(PurePosixPath(path).parent))
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "--workdir",
            self.config.workdir,
            self._container,
            "sh",
            "-c",
            f"mkdir -p {parent} && cat > {shlex.quote(path)}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate(input=data)
        if proc.returncode != 0:
            raise SandboxError(
                f"write {path!r}: {stderr.decode(errors='replace').strip()}"
            )

    async def stop_confirmed(self) -> None:
        if self._container is None:
            return
        removed = await docker("rm", "--force", self._container)
        remaining = await docker(
            "ps",
            "--all",
            "--filter",
            f"name=^/{self._container}$",
            "--format",
            "{{.Names}}",
        )
        if remaining.exit_code or self._container in remaining.stdout.splitlines():
            detail = (removed.stderr or remaining.stderr or removed.stdout).strip()
            raise SandboxError(
                f"docker container {self._container!r} deletion was not confirmed: {detail}"
            )
        self._stopped = True

    def cleanup(self) -> None:
        if self._container is None or self._stopped:
            return
        self._stopped = (
            True  # idempotency guard; keep `_container` so the name still shows
        )
        logger.debug("docker: removing container %s", self._container)
        with contextlib.suppress(Exception):
            subprocess.run(
                ["docker", "rm", "--force", self._container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
