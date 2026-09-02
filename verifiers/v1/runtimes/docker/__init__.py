"""Local Docker runtime with optional execution-time URL filtering. Podman drives the same
CLI surface (and aliases `host.docker.internal`), so it reuses the whole implementation.

Containers get the engine's private network, so a task's ports never collide with the
host's. Host loopback stays reachable from inside at its own port: on Linux through a
listener planted on the container's loopback and relayed here (a "door"), off Linux
through the engine's host alias (which a restricted runtime reaches through its egress
proxy). A service hosted in the container is published the other way, to a host
loopback port."""

import array
import asyncio
import contextlib
import functools
import logging
import socket
import subprocess
import sys
import tempfile
from typing import ClassVar, Literal
from urllib.parse import urlsplit

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import SERVICE_PORT, BaseRuntimeInfo, parse_gpu
from verifiers.v1.runtimes.container import ContainerConfig, ContainerRuntime, cli
from verifiers.v1.runtimes.docker.egress import (
    HOST_ALIAS,
    EgressProxy,
    NetworkPolicy,
    relay,
)

logger = logging.getLogger(__name__)


class DockerConfig(ContainerConfig, NetworkPolicyConfig):
    type: Literal["docker"] = "docker"


class PodmanConfig(ContainerConfig, NetworkPolicyConfig):
    type: Literal["podman"] = "podman"


class DockerRuntimeInfo(DockerConfig, BaseRuntimeInfo):
    pass


class PodmanRuntimeInfo(PodmanConfig, BaseRuntimeInfo):
    pass


_NO_PROXY = "localhost,127.0.0.1"
_PLANT_LISTENERS = r"""
import array, socket, sys
control = socket.socket(socket.AF_UNIX)
control.connect("/run/vf/control.sock")
listeners = []
for port in sys.argv[1:]:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", int(port)))
    listener.listen()
    listeners.append(listener)
fds = array.array("i", [listener.fileno() for listener in listeners])
control.sendmsg([b"listeners"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])
"""


async def _door(
    port: int, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Relay a connection accepted on the container's loopback to the host's."""
    try:
        upstream = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        writer.close()
        return
    await relay(reader, writer, *upstream, timeout=None)


class DockerRuntime(ContainerRuntime):
    engine: ClassVar[str] = "docker"
    """The CLI binary; Podman's is a drop-in for every command used here."""
    info_cls: ClassVar[type[BaseRuntimeInfo]] = DockerRuntimeInfo

    def __init__(
        self, config: DockerConfig | PodmanConfig, name: str | None = None
    ) -> None:
        super().__init__(name)
        self.config = config
        self.info = self.info_cls(**config.model_dump())
        self._container: str | None = None  # our `--name` (used for exec/rm)
        self._service_url: str | None = None
        self._proxy: EgressProxy | None = None
        # Host loopback port -> the door serving it inside; None until the next exec.
        self._doors: dict[int, asyncio.Server | None] = {}
        self._doors_lock = asyncio.Lock()
        self._stopped = False
        self._cut = False

    @property
    def published_port(self) -> int:
        return SERVICE_PORT

    async def start(self) -> None:
        try:
            version = await cli(self.engine, "version")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"{self.engine} runtime selected but the `{self.engine}` CLI is not installed"
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
                f"{self.engine} runtime selected but the engine is not reachable: {detail}{hint}"
            )
        self._container = self.name
        # The engine's NAT'd bridge: a private namespace with `eth0`, which the cut relies
        # on (rootless Podman would otherwise default to pasta).
        options = ["--network", "bridge", "--publish", f"127.0.0.1::{SERVICE_PORT}"]
        if self.config.cpu is not None:
            options += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            options += ["--memory", f"{self.config.memory}g"]
        _, gpu_count = parse_gpu(self.config.gpu)
        if gpu_count:
            # Docker takes a device count; Podman takes `all` or a device id.
            options += ["--gpus", str(gpu_count) if self.engine == "docker" else "all"]
        restricted = self.network_restricted
        if restricted:
            options += [
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
            options += ["--add-host", f"{HOST_ALIAS}:host-gateway"]
        env_args = [
            arg
            for key, value in self.env.items()
            for arg in ("--env", f"{key}={value}")
        ]
        run = await cli(
            self.engine,
            "run",
            "--detach",
            *options,
            *env_args,
            "--entrypoint",
            "sleep",
            "--name",
            self._container,
            self.config.image,
            "infinity",
        )
        if run.exit_code != 0:
            raise SandboxError(f"{self.engine} run failed: {run.stderr.strip()}")
        self.info.id = run.stdout.strip()[:12]  # `run -d` prints the container id
        # Docker creates a missing `--workdir` (as root) for `exec`; Podman refuses one.
        made = await cli(
            self.engine,
            "exec",
            "--user",
            "0",
            self._container,
            "mkdir",
            "-p",
            self.config.workdir,
        )
        if made.exit_code != 0:
            raise SandboxError(
                f"{self.engine} workdir setup failed: {made.stderr.strip()}"
            )
        published = await cli(
            self.engine, "port", self._container, f"{SERVICE_PORT}/tcp"
        )
        host_port = (published.stdout.split() or [""])[0].rpartition(":")[2]
        if published.exit_code != 0 or not host_port.isdigit():
            detail = (published.stderr or published.stdout).strip()
            raise SandboxError(
                f"{self.engine} did not publish the service port: {detail}"
            )
        self._service_url = f"http://127.0.0.1:{host_port}"
        if restricted:
            # Setup is trusted; colocated servers fetch their task from host interception
            # before the final framework routes are known.
            self._proxy = EgressProxy(
                NetworkPolicy(NetworkPolicyConfig(), [], allow_non_global=True)
            )
            await self._proxy.start("127.0.0.1")
        logger.info(
            "%s: started container %s (image=%s)",
            self.engine,
            self._container,
            self.config.image,
        )

    def host_url(self, url: str) -> str:
        parts = urlsplit(url)
        if parts.hostname not in ("127.0.0.1", "localhost"):
            return url
        if sys.platform != "linux":
            return url.replace(parts.hostname, HOST_ALIAS, 1)
        # Same address inside: the next exec opens a door at this port.
        default_port = 443 if parts.scheme == "https" else 80
        self._doors.setdefault(parts.port or default_port, None)
        return url

    async def expose(self, port: int) -> str:
        if port != SERVICE_PORT or self._service_url is None:
            raise SandboxError(
                f"{self.engine} publishes only port {SERVICE_PORT}, not {port}"
            )
        return self._service_url

    async def _open_doors(self) -> None:
        async with self._doors_lock:
            ports = [port for port, door in self._doors.items() if door is None]
            if not ports:
                return
            for port, listener in zip(ports, await self._plant_listeners(ports)):
                self._doors[port] = await asyncio.start_server(
                    functools.partial(_door, port), sock=listener
                )

    async def _plant_listeners(self, ports: list[int]) -> list[socket.socket]:
        """Bind listeners at `ports` on the container's loopback, serviced here: a helper
        sharing the container's network namespace binds them and passes them back."""
        with (
            tempfile.TemporaryDirectory(prefix="vf-door-") as directory,
            socket.socket(socket.AF_UNIX) as control,
        ):
            control.bind(f"{directory}/control.sock")
            control.listen(1)
            helper = await cli(
                self.engine,
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
                "--volume",
                f"{directory}:/run/vf:Z",
                "docker.io/library/python:3.11-alpine",
                "python3",
                "-c",
                _PLANT_LISTENERS,
                *map(str, ports),
            )
            if helper.exit_code != 0:
                raise SandboxError(
                    f"{self.engine} loopback listener failed: {helper.stderr.strip()}"
                )
            connection, _ = control.accept()
            with connection:
                _, ancillary, *_ = connection.recvmsg(
                    64, socket.CMSG_SPACE(len(ports) * array.array("i").itemsize)
                )
        descriptors = array.array("i")
        descriptors.frombytes(ancillary[0][2][: len(ports) * descriptors.itemsize])
        return [socket.socket(fileno=fd) for fd in descriptors]

    async def prepare_execution(self, routes: list[str] | None) -> None:
        """Allow the declared framework routes, then leave the proxy as the only way out
        (on Linux besides the loopback doors)."""
        if not self.network_restricted:
            return
        assert self._proxy is not None
        if routes is None:
            self._proxy.policy = NetworkPolicy(
                NetworkPolicyConfig(), [], allow_non_global=True
            )
            return
        framework = [
            urlsplit(url)._replace(path="", query="", fragment="").geturl()
            for url in routes
        ]
        self._proxy.policy = NetworkPolicy(self.config, framework)
        if self._cut:
            return
        # Off Linux the host is a real address reached through the engine's gateway, so
        # the cut keeps a route to it open on the proxy port only.
        host = ""
        if sys.platform != "linux":
            found = await cli(
                self.engine,
                "exec",
                self._container,
                "sh",
                "-c",
                f"awk '$2 == \"{HOST_ALIAS}\" {{ print $1; exit }}' /etc/hosts",
            )
            host = found.stdout.strip()
            if found.exit_code != 0 or not host:
                raise SandboxError(
                    f"could not resolve {HOST_ALIAS} in {self.engine}: {found.stderr.strip()}"
                )
        script = (
            "set -eu; HOST=$1; PORT=$2; "
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
        cut = await cli(
            self.engine,
            "run",
            "--rm",
            "--network",
            f"container:{self._container}",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "NET_ADMIN",
            "docker.io/library/alpine:3.22",
            "sh",
            "-c",
            script,
            "cut",
            host,
            str(self._proxy.port),
        )
        if cut.exit_code != 0:
            raise SandboxError(
                f"{self.engine} network cut failed: {cut.stderr.strip()}"
            )
        self._cut = True

    def _proxy_env(self) -> dict[str, str]:
        assert self._proxy is not None
        proxy = self.host_url(
            f"http://verifiers:{self._proxy.token}@127.0.0.1:{self._proxy.port}"
        )
        return {
            "HTTP_PROXY": proxy,
            "HTTPS_PROXY": proxy,
            "http_proxy": proxy,
            "https_proxy": proxy,
            "NO_PROXY": _NO_PROXY,
            "no_proxy": _NO_PROXY,
        }

    async def teardown(self) -> None:
        for door in self._doors.values():
            if door is not None:
                door.close()
        if self._proxy is not None:
            await self._proxy.stop()
        await super().teardown()

    async def _exec(self, env: dict[str, str], *, stdin: bool = False) -> list[str]:
        assert self._container is not None
        if self._proxy is not None:
            env = {**env, **self._proxy_env()}
        await self._open_doors()
        return [
            self.engine,
            "exec",
            *(("-i",) if stdin else ()),
            *(arg for key, value in env.items() for arg in ("--env", f"{key}={value}")),
            "--workdir",
            self.config.workdir,
            self._container,
        ]

    def cleanup(self) -> None:
        if self._container is None or self._stopped:
            return
        self._stopped = (
            True  # idempotency guard; keep `_container` so the name still shows
        )
        logger.debug("%s: removing container %s", self.engine, self._container)
        with contextlib.suppress(Exception):
            subprocess.run(
                [self.engine, "rm", "--force", self._container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )


class PodmanRuntime(DockerRuntime):
    engine = "podman"
    info_cls = PodmanRuntimeInfo
