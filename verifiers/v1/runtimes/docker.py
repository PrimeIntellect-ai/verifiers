"""Local Docker runtime with full or interception-only agent networking.

Full mode shares the host network. Interception mode provisions over an ordinary bridge, then
moves the container onto an internal per-rollout network whose only peer is a fixed reverse proxy
to the interception tunnel before the agent starts.
"""

import asyncio
import contextlib
import logging
import re
import shlex
import subprocess
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import ProgramResult, Runtime, parse_gpu

logger = logging.getLogger(__name__)

# The trusted, fixed-upstream reverse relay. It never receives the Docker socket.
_RELAY_IMAGE = "nginx:1.28.3-alpine"
_RELAY_HOST = "vf-interception"
_RELAY_PORT = 8080


class DockerConfig(BaseConfig):
    type: Literal["docker"] = "docker"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    network_access: Literal["full", "interception"] = "full"
    """Agent-execution networking. `full` preserves host networking; `interception` allows
    internet-backed setup, then limits the agent to the rollout's interception endpoint."""
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


class DockerRuntime(Runtime):
    """Runs programs in a local container, optionally sealing agent egress to interception."""

    def __init__(self, config: DockerConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self._container: str | None = None  # our `--name` (used for exec/rm)
        self._container_id: str | None = None  # docker's short id (for display)
        self._relay: str | None = None
        self._network: str | None = None
        self._stopped = False

    @property
    def descriptor(self) -> str | None:
        return self._container_id

    @classmethod
    def config_reaches_host_locally(cls, config: object) -> bool:
        return isinstance(config, DockerConfig) and config.network_access == "full"

    @property
    def interception_only(self) -> bool:
        return self.config.network_access == "interception"

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
        network = "bridge" if self.interception_only else "host"
        capabilities = ["--cap-drop", "NET_ADMIN"] if self.interception_only else []
        run = await docker(
            "run",
            "--detach",
            "--network",
            network,
            *capabilities,
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
            self.config.network_access,
        )

    async def seal_agent_network(self, endpoint: str) -> str:
        if not self.interception_only:
            return endpoint
        if self._container is None:
            raise SandboxError("docker container is not running")

        parsed = urlsplit(endpoint)
        host = parsed.hostname
        try:
            port = parsed.port
        except ValueError as e:
            raise SandboxError(
                f"invalid interception endpoint {endpoint!r}: {e}"
            ) from e
        if (
            parsed.scheme not in {"http", "https"}
            or host is None
            or parsed.username is not None
            or parsed.password is not None
            or re.fullmatch(r"[A-Za-z0-9.-]+", host) is None
        ):
            raise SandboxError(f"invalid interception endpoint {endpoint!r}")

        authority = f"{host}:{port}" if port is not None else host
        upstream = f"{parsed.scheme}://{authority}"
        self._network = f"{self.name}-network"
        self._relay = f"{self.name}-relay"
        # Static proxy_pass + Host/SNI prevent the agent from selecting another upstream.
        config = f"""
worker_processes 1;
pid /tmp/nginx.pid;
error_log stderr warn;
events {{ worker_connections 128; }}
http {{
    access_log off;
    client_body_temp_path /tmp/client_body;
    proxy_temp_path /tmp/proxy;
    fastcgi_temp_path /tmp/fastcgi;
    uwsgi_temp_path /tmp/uwsgi;
    scgi_temp_path /tmp/scgi;
    server {{
        listen {_RELAY_PORT};
        client_max_body_size 0;
        location / {{
            proxy_pass {upstream};
            proxy_set_header Host {authority};
            proxy_ssl_server_name on;
            proxy_ssl_name {host};
            proxy_ssl_verify on;
            proxy_ssl_trusted_certificate /etc/ssl/certs/ca-certificates.crt;
            proxy_ssl_verify_depth 3;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_request_buffering off;
            proxy_buffering off;
            proxy_next_upstream off;
            proxy_read_timeout 86400s;
            proxy_send_timeout 86400s;
        }}
    }}
}}
"""

        created = await docker(
            "network",
            "create",
            "--internal",
            "--opt",
            "com.docker.network.bridge.gateway_mode_ipv4=isolated",
            "--label",
            f"verifiers.runtime={self.name}",
            self._network,
        )
        if created.exit_code != 0:
            raise SandboxError(
                f"docker internal network creation failed: {created.stderr.strip()}"
            )
        relay = await docker(
            "create",
            "--name",
            self._relay,
            "--network",
            "bridge",
            "--label",
            f"verifiers.runtime={self.name}",
            "--user",
            "101:101",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m,uid=101,gid=101",
            "--entrypoint",
            "sh",
            _RELAY_IMAGE,
            "-c",
            'printf "%s" "$1" > /tmp/nginx.conf && '
            'exec nginx -c /tmp/nginx.conf -g "daemon off;"',
            "vf-relay",
            config,
        )
        if relay.exit_code != 0:
            raise SandboxError(
                f"docker interception relay creation failed: {relay.stderr.strip()}"
            )
        connected = await docker(
            "network",
            "connect",
            "--alias",
            _RELAY_HOST,
            self._network,
            self._relay,
        )
        if connected.exit_code != 0:
            raise SandboxError(
                f"docker interception relay network failed: {connected.stderr.strip()}"
            )
        started = await docker("start", self._relay)
        if started.exit_code != 0:
            raise SandboxError(
                f"docker interception relay start failed: {started.stderr.strip()}"
            )
        for _ in range(20):
            ready = await docker(
                "exec", self._relay, "nginx", "-t", "-c", "/tmp/nginx.conf"
            )
            if ready.exit_code == 0:
                break
            await asyncio.sleep(0.05)
        else:
            logs = await docker("logs", self._relay)
            raise SandboxError(
                "docker interception relay did not become ready: "
                f"{(logs.stderr or logs.stdout).strip()[-2000:]}"
            )

        connected = await docker("network", "connect", self._network, self._container)
        if connected.exit_code != 0:
            raise SandboxError(
                f"docker agent network connection failed: {connected.stderr.strip()}"
            )
        disconnected = await docker("network", "disconnect", "bridge", self._container)
        if disconnected.exit_code != 0:
            raise SandboxError(
                f"docker setup network removal failed: {disconnected.stderr.strip()}"
            )
        logger.info(
            "docker: sealed container %s to interception relay %s",
            self._container,
            self._relay,
        )
        return urlunsplit(
            (
                "http",
                f"{_RELAY_HOST}:{_RELAY_PORT}",
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
        if self._stopped or not any((self._container, self._relay, self._network)):
            return
        self._stopped = True  # keep the names so descriptors and logs survive teardown
        for container in (self._relay, self._container):
            if container is None:
                continue
            logger.debug("docker: removing container %s", container)
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["docker", "rm", "--force", container],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
        if self._network is not None:
            logger.debug("docker: removing network %s", self._network)
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["docker", "network", "rm", self._network],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
