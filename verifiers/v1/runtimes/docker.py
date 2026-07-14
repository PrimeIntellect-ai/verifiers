"""Local Docker runtime with policy-controlled agent networking."""

import asyncio
import json
import logging
import re
import shlex
import subprocess
import sys
from pathlib import PurePosixPath
from typing import Literal, cast
from urllib.parse import urlsplit, urlunsplit

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    parse_gpu,
)
from verifiers.v1.runtimes.network import NetworkPolicy

logger = logging.getLogger(__name__)

_EGRESS_IMAGE = "nginx:1.28.3-alpine"
_EGRESS_HOST = "egress"
_EGRESS_PORT = 8080


class DockerConfig(BaseConfig):
    type: Literal["docker"] = "docker"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    network_access: bool = True
    """Allow arbitrary agent egress. Setup has access; false then allows framework routes only."""
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


async def docker_checked(*args: str, error: str) -> ProgramResult:
    result = await docker(*args)
    if result.exit_code == 0:
        return result
    detail = (result.stderr or result.stdout).strip()
    raise SandboxError(f"{error}: {detail}")


async def docker_interception_host() -> str | None:
    """Return the host address used by native Linux containers."""
    if sys.platform != "linux":
        return None
    result = await docker(
        "network", "inspect", "--format", "{{(index .IPAM.Config 0).Gateway}}", "bridge"
    )
    gateway = result.stdout.strip()
    if result.exit_code != 0 or not gateway:
        raise SandboxError(
            "could not resolve the Docker host gateway: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return gateway


class DockerRuntime(Runtime):
    def __init__(self, config: DockerConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = DockerRuntimeInfo(**config.model_dump())
        self._container: str | None = None  # our `--name` (used for exec/rm)
        self._egress: str | None = None
        self._setup_network: str | None = None
        self._execution_network: str | None = None
        self._interception_host: str | None = None
        self._stopped = False

    @classmethod
    def config_reaches_host_locally(cls, config: BaseConfig) -> bool:
        return cast(DockerConfig, config).network_access

    @property
    def reachable_from_host_locally(self) -> bool:
        return self.config.network_access

    @property
    def interception_host(self) -> str | None:
        return self._interception_host

    @property
    def network_policy(self) -> NetworkPolicy:
        return NetworkPolicy(allow=None if self.config.network_access else [])

    @property
    def supports_colocated_tools(self) -> bool:
        return self.config.network_access

    @property
    def supports_colocated_user(self) -> bool:
        return self.config.network_access

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
        if self.network_policy.restricted:
            self._interception_host = await docker_interception_host()
            self._setup_network = f"{self.name}-setup"
            await docker_checked(
                "network",
                "create",
                "--label",
                f"verifiers.runtime={self.name}",
                self._setup_network,
                error="docker setup network creation failed",
            )
        network = self._setup_network or "host"
        restrictions = (
            []
            if self.config.network_access
            else [
                "--cap-drop",
                "NET_ADMIN",
                "--cap-drop",
                "NET_RAW",
                "--sysctl",
                "net.ipv6.conf.all.disable_ipv6=1",
            ]
        )
        run = await docker_checked(
            "run",
            "--detach",
            "--network",
            network,
            *restrictions,
            *limits,
            "--workdir",
            self.config.workdir,
            "--name",
            self._container,
            self.config.image,
            "sleep",
            "infinity",
            error="docker run failed",
        )
        self.info.id = run.stdout.strip()[
            :12
        ]  # `docker run -d` prints the container id
        logger.info(
            "docker: started container %s (image=%s, network=%s)",
            self._container,
            self.config.image,
            network,
        )

    async def apply_network_policy(self, routes: dict[str, str]) -> dict[str, str]:
        if not self.network_policy.restricted:
            return routes
        if self._container is None or self._setup_network is None:
            raise SandboxError("docker container is not running")

        parsed_routes = {}
        servers = []
        for offset, (name, url) in enumerate(routes.items()):
            parsed = urlsplit(url)
            host = parsed.hostname
            try:
                port = parsed.port
            except ValueError as e:
                raise SandboxError(
                    f"invalid required network route {url!r}: {e}"
                ) from e
            if (
                parsed.scheme not in {"http", "https"}
                or host is None
                or parsed.username is not None
                or parsed.password is not None
                or re.fullmatch(r"[A-Za-z0-9.-]+", host) is None
            ):
                raise SandboxError(f"invalid required network route {url!r}")
            path = parsed.path or "/"
            if (
                not path.isascii()
                or (path != "/" and not path.strip("/"))
                or any(
                    not char.isprintable() or char.isspace() or char == "$"
                    for char in path
                )
            ):
                raise SandboxError(f"invalid required network route {url!r}")
            authority = f"{host}:{port}" if port is not None else host
            upstream_host = (
                "host.docker.internal" if host in {"127.0.0.1", "localhost"} else host
            )
            upstream_authority = (
                f"{upstream_host}:{port}" if port is not None else upstream_host
            )
            relay_port = _EGRESS_PORT + offset
            parsed_routes[name] = parsed, relay_port
            locations = (
                f"location / {{ proxy_pass {parsed.scheme}://{upstream_authority}; }}"
                if path == "/"
                else f"""
        location = {json.dumps(path)} {{ proxy_pass {parsed.scheme}://{upstream_authority}; }}
        location ^~ {json.dumps(f"{path.rstrip('/')}/")} {{
            proxy_pass {parsed.scheme}://{upstream_authority};
        }}
        location / {{ return 403; }}"""
            )
            servers.append(
                f"""
    server {{
        listen {relay_port};
        client_max_body_size 0;
        proxy_set_header Host {authority};
        proxy_ssl_server_name on;
        proxy_ssl_name {host};
        proxy_ssl_verify on;
        proxy_ssl_trusted_certificate /etc/ssl/certs/ca-certificates.crt;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_request_buffering off;
        proxy_buffering off;
        proxy_next_upstream off;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
        {locations}
    }}
"""
            )
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
{"".join(servers)}}}
"""

        self._egress = f"{self.name}-egress"
        self._execution_network = f"{self.name}-execution"
        await docker_checked(
            "network",
            "create",
            "--internal",
            "--opt",
            "com.docker.network.bridge.gateway_mode_ipv4=isolated",
            "--label",
            f"verifiers.runtime={self.name}",
            self._execution_network,
            error="docker execution network creation failed",
        )
        await docker_checked(
            "create",
            "--name",
            self._egress,
            "--network",
            self._setup_network,
            "--add-host",
            f"host.docker.internal:{self._interception_host or 'host-gateway'}",
            "--label",
            f"verifiers.runtime={self.name}",
            "--user",
            "101:101",
            "--cap-drop",
            "ALL",
            "--sysctl",
            "net.ipv4.ip_forward=0",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m,uid=101,gid=101",
            "--entrypoint",
            "sh",
            _EGRESS_IMAGE,
            "-c",
            'printf "%s" "$1" > /tmp/nginx.conf && '
            'exec nginx -c /tmp/nginx.conf -g "daemon off;"',
            "vf-egress",
            config,
            error="docker egress creation failed",
        )
        await docker_checked(
            "network",
            "connect",
            "--alias",
            _EGRESS_HOST,
            self._execution_network,
            self._egress,
            error="docker egress network connection failed",
        )
        await docker_checked("start", self._egress, error="docker egress start failed")
        for _ in range(20):
            ready = await docker(
                "exec", self._egress, "nginx", "-t", "-c", "/tmp/nginx.conf"
            )
            if ready.exit_code == 0:
                break
            await asyncio.sleep(0.05)
        else:
            logs = await docker("logs", self._egress)
            raise SandboxError(
                "docker egress did not become ready: "
                f"{(logs.stderr or logs.stdout).strip()[-2000:]}"
            )
        await docker_checked(
            "network",
            "connect",
            self._execution_network,
            self._container,
            error="docker execution network connection failed",
        )
        await docker_checked(
            "network",
            "disconnect",
            self._setup_network,
            self._container,
            error="docker setup network disconnection failed",
        )
        logger.info(
            "docker: applied network policy to %s via %s",
            self._container,
            self._egress,
        )
        return {
            name: urlunsplit(
                (
                    "http",
                    f"{_EGRESS_HOST}:{port}",
                    parsed.path,
                    parsed.query,
                    "",
                )
            )
            for name, (parsed, port) in parsed_routes.items()
        }

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
        await docker_checked(
            "exec",
            "--detach",
            *env_args,
            "--workdir",
            self.config.workdir,
            self._container,
            "sh",
            "-c",
            inner,
            error="docker exec -d failed",
        )  # detached → lives in the container until it's removed in stop()

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
        resources = (
            self._container,
            self._egress,
            self._setup_network,
            self._execution_network,
        )
        if self._stopped or not any(resources):
            return
        for attr in ("_egress", "_container"):
            container = getattr(self, attr)
            if container is None:
                continue
            logger.debug("docker: removing container %s", container)
            try:
                result = subprocess.run(
                    ["docker", "rm", "--force", container],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except Exception:
                continue
            if result.returncode == 0:
                setattr(self, attr, None)
        for attr in ("_execution_network", "_setup_network"):
            network = getattr(self, attr)
            if network is None:
                continue
            logger.debug("docker: removing network %s", network)
            try:
                result = subprocess.run(
                    ["docker", "network", "rm", network],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except Exception:
                continue
            if result.returncode == 0:
                setattr(self, attr, None)
        self._stopped = not any(
            (
                self._container,
                self._egress,
                self._setup_network,
                self._execution_network,
            )
        )
