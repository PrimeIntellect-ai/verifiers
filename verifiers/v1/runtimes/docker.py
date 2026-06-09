"""Local Docker runtime: run the program in a container sharing the host network.

On Linux the container shares the host network (`--network host`), so it reaches
the interception server on localhost directly — no tunnel needed.
"""

import asyncio
import contextlib
import logging
import shlex
import subprocess
import uuid
from pathlib import PurePosixPath
from typing import Literal

from pydantic_config import BaseConfig

from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes.base import ProgramResult, Runtime, parse_gpu

logger = logging.getLogger(__name__)


class DockerConfig(BaseConfig):
    type: Literal["docker"] = "docker"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    # Resources in Modal's units (also settable per-task via Task.resources).
    cpu: float | None = None
    """Pin the container to this many CPU cores (docker `--cpus`). None = unlimited."""
    memory: int | None = None
    """Hard memory limit in MB (docker `--memory`). None = unlimited."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "2" (docker `--gpus` uses the count; needs the nvidia
    container toolkit). None = none."""
    disk: int | None = None
    """Advisory disk request in MB. Docker has no portable per-container size limit, so
    this is accepted (so a task can declare it without a warning) but not enforced."""


async def docker(*args: str) -> ProgramResult:
    """Run a `docker` CLI command, capturing its result."""
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
    """Runs the program in a local Docker container reachable over the host network."""

    def __init__(self, config: DockerConfig) -> None:
        self.config = config
        self._container: str | None = None  # our `--name` (used for exec/rm)
        self._container_id: str | None = None  # docker's short id (for display)
        self._stopped = False

    @property
    def descriptor(self) -> str | None:
        return self._container_id

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
        self._container = f"vf-{uuid.uuid4().hex[:12]}"
        limits: list[str] = []
        if self.config.cpu is not None:
            limits += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            limits += ["--memory", f"{self.config.memory}m"]
        _, gpu_count = parse_gpu(self.config.gpu)
        if gpu_count:
            limits += ["--gpus", str(gpu_count)]
        run = await docker(
            "run",
            "--detach",
            "--network",
            "host",
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
            raise ProgramError(f"docker run failed: {run.stderr.strip()}")
        self._container_id = run.stdout.strip()[
            :12
        ]  # `docker run -d` prints the container id
        logger.info(
            "docker: started container %s (image=%s)",
            self._container,
            self.config.image,
        )

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
            raise ProgramError(f"docker exec -d failed: {run.stderr.strip()}")

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
            raise ProgramError(
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
            raise ProgramError(
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
