"""Local Docker runtime with optional execution-time URL filtering."""

import array
import asyncio
import contextlib
import logging
import shlex
import socket
import subprocess
import sys
import tempfile
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    parse_gpu,
)
from verifiers.v1.runtimes.egress import EgressProxy, NetworkPolicy

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
    """False = keep internet only through setup (package installs, provisioning), then
    allow only the interception URL, MCP URLs, and `allow` entries."""
    allow: list[str] = []
    """URL origins or host patterns the agent may reach with `network_access=False`.
    Wildcards are supported and `*.example.com` also matches the apex."""
    block: list[str] = []
    """URL origins or host patterns denied during execution. Block rules win over
    `allow`; framework interception and MCP routes always remain reachable."""

    @property
    def network_isolated(self) -> bool:
        """True when the runtime narrows networking after setup: network_access=False,
        or a block list to enforce."""
        return not self.network_access or bool(self.block)


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


_PROXY_HOST = "host.docker.internal"
_PASS_LISTENER = r"""
import array, socket
control = socket.socket(socket.AF_UNIX)
control.connect("/run/vf/control.sock")
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", 0))
listener.listen()
control.sendmsg([b"listener"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [listener.fileno()]))])
"""


class DockerRuntime(Runtime):
    def __init__(self, config: DockerConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = DockerRuntimeInfo(**config.model_dump())
        self._container: str | None = None  # our `--name` (used for exec/rm)
        self._proxy: EgressProxy | None = None
        self._proxy_host_ip: str | None = None
        self._stopped = False
        self._cut = False

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
        isolated = self.network_isolated
        if isolated:
            network = [
                "--network",
                "bridge",
                "--cap-drop",
                "NET_ADMIN",
                "--cap-drop",
                "NET_RAW",
                "--security-opt",
                "no-new-privileges",
                "--sysctl",
                "net.ipv6.conf.all.disable_ipv6=1",
            ]
            if sys.platform != "linux":
                network += ["--add-host", f"{_PROXY_HOST}:host-gateway"]
        else:
            network = ["--network", "host"]
        run = await docker(
            "run",
            "--detach",
            *network,
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
        if isolated:
            # Setup is trusted and stays online; install the real policy at the cut.
            self._proxy = EgressProxy(NetworkPolicy([], [], [], True))
            if sys.platform == "linux":
                await self._proxy.start(listener=await self._container_listener())
            else:
                host = await docker(
                    "exec",
                    self._container,
                    "sh",
                    "-c",
                    f"awk '$2 == \"{_PROXY_HOST}\" {{ print $1; exit }}' /etc/hosts",
                )
                self._proxy_host_ip = host.stdout.strip()
                if host.exit_code != 0 or not self._proxy_host_ip:
                    raise SandboxError(
                        f"could not resolve {_PROXY_HOST} in Docker: {host.stderr.strip()}"
                    )
                await self._proxy.start("127.0.0.1")
        logger.info(
            "docker: started container %s (image=%s)",
            self._container,
            self.config.image,
        )

    async def _container_listener(self) -> socket.socket:
        """Create the proxy listener inside the container netns, serviced here."""
        with tempfile.TemporaryDirectory(prefix="vf-proxy-") as directory:
            path = f"{directory}/control.sock"
            with socket.socket(socket.AF_UNIX) as control:
                control.bind(path)
                control.listen(1)
                helper = await docker(
                    "run",
                    "--rm",
                    "--network",
                    f"container:{self._container}",
                    "--cap-drop",
                    "ALL",
                    "--cap-add",
                    "DAC_OVERRIDE",
                    "--security-opt",
                    "no-new-privileges",
                    "--mount",
                    f"type=bind,source={directory},target=/run/vf",
                    "python:3.11-alpine",
                    "python3",
                    "-c",
                    _PASS_LISTENER,
                )
                if helper.exit_code != 0:
                    raise SandboxError(
                        f"docker proxy listener failed: {helper.stderr.strip()}"
                    )
                connection, _ = control.accept()
                with connection:
                    _, ancillary, *_ = connection.recvmsg(
                        64, socket.CMSG_SPACE(array.array("i").itemsize)
                    )
        descriptor = array.array("i")
        descriptor.frombytes(ancillary[0][2][: descriptor.itemsize])
        listener = socket.socket(fileno=descriptor[0])
        listener.setblocking(False)
        return listener

    def host_url(self, url: str) -> str:
        host = urlsplit(url).hostname
        if self.network_isolated and host == "localhost":
            return url.replace(host, "127.0.0.1", 1)
        if (
            not self.network_isolated
            and sys.platform != "linux"
            and host in ("127.0.0.1", "localhost")
        ):
            return url.replace(host, "host.docker.internal", 1)
        return url

    @property
    def network_isolated(self) -> bool:
        return self.config.network_isolated

    async def prepare_execution(self, routes: list[str]) -> None:
        """Allow the declared framework routes, then leave the proxy as the only route."""
        if not self.network_isolated or self._cut:
            return
        assert self._proxy is not None
        framework: list[str] = []
        for url in routes:
            parsed = urlsplit(url)
            framework.append(urlunsplit((parsed.scheme, parsed.netloc, "", "", "")))
        self._proxy.policy = NetworkPolicy(
            self.config.allow,
            self.config.block,
            framework,
            self.config.network_access,
        )
        script = (
            "set -eu; HOST=$1; "
            "PORT=$2; "
            'if [ -n "$HOST" ]; then apk add --no-cache iptables >/dev/null; fi; '
            "GW=$(ip route show default | awk '/^default via/{print $3; exit}'); "
            "SUBNET=$(ip route show | awk '/scope link/{print $1; exit}'); "
            'if [ -n "$HOST" ]; then '
            'if [ "$HOST" = "$GW" ]; then ip route add "$HOST/32" dev eth0; '
            'else ip route add "$HOST/32" via "$GW"; '
            'ip route add "$GW/32" dev eth0; fi; '
            "fi; "
            "ip route del default; "
            'ip route del "$SUBNET" dev eth0; '
            "ip route add blackhole 127.0.0.11/32 table local; "
            'if [ -n "$HOST" ]; then iptables -F OUTPUT; '
            "iptables -A OUTPUT -o lo -j ACCEPT; "
            'iptables -A OUTPUT -d "$HOST" -p tcp --dport "$PORT" -j ACCEPT; '
            "iptables -A OUTPUT -j REJECT; fi"
        )
        cut = await docker(
            "run",
            "--rm",
            "--network",
            f"container:{self._container}",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "NET_ADMIN",
            "alpine:3.22",
            "sh",
            "-c",
            script,
            "cut",
            self._proxy_host_ip or "",
            str(self._proxy.port),
        )
        if cut.exit_code != 0:
            raise SandboxError(f"docker network cut failed: {cut.stderr.strip()}")
        self._cut = True

    def _proxy_env(self) -> dict[str, str]:
        if self._proxy is None:
            return {}
        host = "127.0.0.1" if sys.platform == "linux" else _PROXY_HOST
        proxy = f"http://{host}:{self._proxy.port}"
        return {
            "HTTP_PROXY": proxy,
            "HTTPS_PROXY": proxy,
            "http_proxy": proxy,
            "https_proxy": proxy,
            "NO_PROXY": "localhost",
            "no_proxy": "localhost",
        }

    async def teardown(self) -> None:
        await super().teardown()
        if self._proxy is not None:
            await self._proxy.stop()

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        env = {**(self._proxy_env() if self._cut else {}), **env}
        env_args = [arg for k, v in env.items() for arg in ("--env", f"{k}={v}")]
        return await docker(
            "exec", *env_args, "--workdir", self.config.workdir, self._container, *argv
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        env = {**self._proxy_env(), **env}
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
