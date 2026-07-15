"""Local Docker runtime using host networking."""

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
    network_access: bool = True
    """Whether the workload can reach the internet. When False, the container's DNS is disabled
    so hostname fetches (pip, git clone, curl) fail, while the model tunnel still works — it
    egresses from the host, which the container reaches DNS-lessly. A DNS block, not an IP
    airgap. Mirrors prime/modal (which also sever the tunnel)."""


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
            *self._network_args(),
            *limits,
            "--workdir",
            self.config.workdir,
            "--name",
            self._container,
            self.config.image,
            "sleep",
            "infinity",
        )
        if run.exit_code != 0:
            raise SandboxError(f"docker run failed: {run.stderr.strip()}")
        self.info.id = run.stdout.strip()[
            :12
        ]  # `docker run -d` prints the container id
        logger.info(
            "docker: started container %s (image=%s%s)",
            self._container,
            self.config.image,
            "" if self.config.network_access else ", internet blocked",
        )

    def _network_args(self) -> list[str]:
        """`docker run` networking. network_access=False breaks DNS to block the workload's
        internet while keeping the DNS-less host route the model tunnel needs."""
        if self.config.network_access:
            return ["--network", "host"]
        if sys.platform == "linux":
            # Host netns reaches the interception at 127.0.0.1; only DNS is broken. --dns writes
            # a container-private resolv.conf, so the host's DNS is untouched.
            return ["--network", "host", "--dns", "0.0.0.0"]
        # Docker Desktop: bridge instead of host, host.docker.internal pinned via /etc/hosts
        # (host_url() already routes the tunnel there off-Linux).
        return ["--dns", "0.0.0.0", "--add-host", "host.docker.internal:host-gateway"]

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
