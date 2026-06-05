"""Local Docker runtime: run the program in a container sharing the host network.

On Linux the container shares the host network (`--network host`), so it reaches
the interception server on localhost directly — no tunnel needed.
"""

import asyncio
from typing import Literal

from pydantic_config import BaseConfig

from verifiers.nano.errors import ProgramError
from verifiers.nano.runtime.base import ProgramResult, Runtime


class DockerConfig(BaseConfig):
    kind: Literal["docker"] = "docker"
    image: str = "python:3.11-slim"
    workdir: str = "/app"


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
        self._container: str | None = None

    async def start(self, port: int) -> str:
        try:
            version = await docker("version", "--format", "{{.Server.Version}}")
        except FileNotFoundError as e:
            raise RuntimeError(
                "docker runtime selected but the `docker` CLI is not installed"
            ) from e
        if version.exit_code != 0:
            raise RuntimeError(
                "docker runtime selected but the Docker daemon is not reachable: "
                f"{(version.stderr or version.stdout).strip()}"
            )
        self._container = f"vf-nano-{port}"
        run = await docker(
            "run",
            "--detach",
            "--network",
            "host",
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
        return f"http://127.0.0.1:{port}/v1"

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        env_args = [arg for k, v in env.items() for arg in ("--env", f"{k}={v}")]
        return await docker(
            "exec", *env_args, "--workdir", self.config.workdir, self._container, *argv
        )

    async def stop(self) -> None:
        if self._container is not None:
            await docker("rm", "--force", self._container)
