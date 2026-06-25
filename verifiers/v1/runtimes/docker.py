"""Local Docker runtime with full or interception-only agent networking.

Full mode shares the host network. Interception mode provisions on Docker's bridge, then installs
a network-namespace firewall that limits the agent to the host interception port.
"""

import asyncio
import contextlib
import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import ProgramResult, Runtime, parse_gpu

logger = logging.getLogger(__name__)


class DockerConfig(BaseConfig):
    type: Literal["docker"] = "docker"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    network_access: bool = True
    """Agent-execution web access. When false, setup can still use the internet, then the agent
    is limited to the rollout's interception endpoint."""
    # TaskResources in Modal's units (also settable per-task via Task.resources).
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


async def network_iptables(pid: int, *args: str) -> ProgramResult:
    """Run host iptables inside a container's network namespace."""
    proc = await asyncio.create_subprocess_exec(
        "nsenter",
        "--target",
        str(pid),
        "--net",
        "iptables",
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


async def docker_bridge_gateway() -> str:
    """Return the IPv4 address through which containers reach host services."""
    result = await docker(
        "network",
        "inspect",
        "--format",
        "{{(index .IPAM.Config 0).Gateway}}",
        "bridge",
    )
    gateway = result.stdout.strip()
    if result.exit_code != 0 or not gateway:
        raise SandboxError(
            "could not resolve the Docker host gateway: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return gateway


class DockerRuntime(Runtime):
    """Runs programs in a local container, optionally sealing agent egress to interception."""

    def __init__(self, config: DockerConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self._container: str | None = None  # our `--name` (used for exec/rm)
        self._container_id: str | None = None  # docker's short id (for display)
        self._host_gateway: str | None = None
        self._stopped = False

    @property
    def descriptor(self) -> str | None:
        return self._container_id

    @property
    def interception_only(self) -> bool:
        return not self.config.network_access

    @property
    def interception_host(self) -> str | None:
        return self._host_gateway if self.interception_only else None

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

        network = "host"
        options: list[str] = []
        if self.interception_only:
            missing = [
                cmd for cmd in ("iptables", "nsenter") if shutil.which(cmd) is None
            ]
            if os.geteuid() != 0 or missing:
                detail = f"; missing {', '.join(missing)}" if missing else ""
                raise RuntimeError(
                    "Docker network_access=false requires root on a Linux Docker host "
                    f"with iptables and nsenter{detail}"
                )
            self._host_gateway = await docker_bridge_gateway()
            network = "bridge"
            options = [
                "--add-host",
                f"host.docker.internal:{self._host_gateway}",
                "--cap-drop",
                "NET_ADMIN",
                "--cap-drop",
                "NET_RAW",
                "--sysctl",
                "net.ipv6.conf.all.disable_ipv6=1",
            ]

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
            network,
            *options,
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
        self._container_id = run.stdout.strip()[
            :12
        ]  # `docker run -d` prints the container id
        logger.info(
            "docker: started container %s (image=%s, network=%s)",
            self._container,
            self.config.image,
            network,
        )

    async def seal_agent_network(self, endpoint: str) -> str:
        if not self.interception_only:
            return endpoint
        if self._container is None:
            raise SandboxError("docker container is not running")

        if self._host_gateway is None:
            raise SandboxError("docker host gateway is not available")

        parsed = urlsplit(endpoint)
        try:
            port = parsed.port
        except ValueError as e:
            raise SandboxError(
                f"invalid interception endpoint {endpoint!r}: {e}"
            ) from e
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or port is None
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise SandboxError(f"invalid interception endpoint {endpoint!r}")

        inspected = await docker("inspect", self._container)
        if inspected.exit_code != 0:
            raise SandboxError(f"docker inspect failed: {inspected.stderr.strip()}")
        details = json.loads(inspected.stdout)[0]
        pid = details["State"]["Pid"]
        chain = "VF-AGENT"
        # Build the chain before activating it. Docker DNS can proxy through the host namespace,
        # so reject its loopback listener before allowing the container's own loopback traffic.
        commands = [
            ("-N", chain),
            (
                "-A",
                chain,
                "-d",
                self._host_gateway,
                "-p",
                "tcp",
                "--dport",
                str(port),
                "-j",
                "ACCEPT",
            ),
            ("-A", chain, "-d", "127.0.0.11", "-j", "REJECT"),
            ("-A", chain, "-o", "lo", "-j", "ACCEPT"),
            ("-A", chain, "-j", "REJECT"),
            ("-I", "OUTPUT", "1", "-j", chain),
        ]
        for command in commands:
            installed = await network_iptables(pid, "--wait", *command)
            if installed.exit_code != 0:
                raise SandboxError(
                    "docker agent firewall failed: "
                    f"{(installed.stderr or installed.stdout).strip()}"
                )

        logger.info(
            "docker: sealed container %s to interception port %s:%d",
            self._container,
            self._host_gateway,
            port,
        )
        return urlunsplit(
            (
                "http",
                f"host.docker.internal:{port}",
                parsed.path,
                parsed.query,
                "",
            )
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
        if self._stopped or self._container is None:
            return
        self._stopped = True  # keep the names so descriptors and logs survive teardown
        logger.debug("docker: removing container %s", self._container)
        with contextlib.suppress(Exception):
            subprocess.run(
                ["docker", "rm", "--force", self._container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
