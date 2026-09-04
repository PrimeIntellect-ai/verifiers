"""Docker and Podman share engine-managed bridges, port publication, and the
Verifiers proxy for host callbacks and optional execution-time URL filtering."""

import array
import contextlib
import json
import logging
import socket
import subprocess
import sys
import tempfile
import uuid
from typing import ClassVar, Literal
from urllib.parse import urlsplit

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import SERVICE_PORT, BaseRuntimeInfo, parse_gpu
from verifiers.v1.runtimes.container import ContainerConfig, ContainerRuntime, cli
from verifiers.v1.runtimes.docker.egress import (
    EgressProxy,
    NetworkPolicy,
    is_loopback_host,
)

logger = logging.getLogger(__name__)


class DockerConfig(ContainerConfig, NetworkPolicyConfig):
    type: Literal["docker"] = "docker"


class PodmanConfig(ContainerConfig, NetworkPolicyConfig):
    type: Literal["podman"] = "podman"
    image: str = "docker.io/library/python:3.11-slim"


class DockerRuntimeInfo(DockerConfig, BaseRuntimeInfo):
    pass


class PodmanRuntimeInfo(PodmanConfig, BaseRuntimeInfo):
    pass


_PROXY_HOST = "host.docker.internal"
_NETWORK_IMAGE = "localhost/verifiers-network:1"
_PASS_LISTENER = r"""
import array, socket
control = socket.socket(socket.AF_UNIX)
control.connect("/run/vf/control.sock")
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
control.sendmsg([b"listener"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [listener.fileno()]))])
"""


class DockerRuntime(ContainerRuntime):
    engine: ClassVar[str] = "docker"
    """The CLI binary for the shared OCI container operations."""
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
        self._image_env: dict[str, str] = {}
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
            if self.engine == "docker" and "permission denied" in detail.lower():
                hint = (
                    "\nYour user isn't in the `docker` group. Either run the command "
                    'under `sg docker -c "..."`, or add yourself with '
                    "`sudo usermod -aG docker $USER` and start a new login shell."
                )
            raise RuntimeError(
                f"{self.engine} runtime selected but the engine is not reachable: {detail}{hint}"
            )
        restricted = self.network_restricted
        if restricted:
            # Cache the tools with the image, so execution never needs package mirrors.
            cached = await cli(self.engine, "image", "inspect", _NETWORK_IMAGE)
            if cached.exit_code != 0:
                with tempfile.TemporaryDirectory(prefix="vf-network-") as directory:
                    built = await cli(
                        self.engine,
                        "build",
                        "--tag",
                        _NETWORK_IMAGE,
                        "--file",
                        "-",
                        directory,
                        input=b"FROM docker.io/library/alpine:3.22\nRUN apk add --no-cache iptables\n",
                    )
                if built.exit_code != 0:
                    raise SandboxError(
                        f"{self.engine} network image build failed: {built.stderr.strip()}"
                    )
        # Engine names are global; cleanup must never target a caller's existing container.
        self._container = f"vf-{uuid.uuid4().hex}"
        # Both engines manage the bridge; rootless Podman's default is otherwise pasta.
        options = ["--network", "bridge", "--publish", f"127.0.0.1::{SERVICE_PORT}"]
        if self.config.cpu is not None:
            options += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            options += ["--memory", f"{self.config.memory}g"]
        _, gpu_count = parse_gpu(self.config.gpu)
        if gpu_count:
            if self.engine == "docker":
                options += ["--gpus", str(gpu_count)]
            else:
                options += [
                    arg
                    for index in range(gpu_count)
                    for arg in ("--device", f"nvidia.com/gpu={index}")
                ]
        if self.engine == "podman":
            options += ["--http-proxy=false"]
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
            options += ["--add-host", f"{_PROXY_HOST}:host-gateway"]
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
        inspected = await cli(
            self.engine, "inspect", "--format", "{{json .Config.Env}}", self._container
        )
        if inspected.exit_code != 0:
            raise SandboxError(
                f"{self.engine} environment inspection failed: {inspected.stderr.strip()}"
            )
        self._image_env = dict(
            entry.split("=", 1) for entry in json.loads(inspected.stdout) or []
        )
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
        # One listener handles every callback, without reserving the host service's
        # port inside the container. Setup egress is trusted until prepare_execution.
        self._proxy = EgressProxy(
            NetworkPolicy(
                NetworkPolicyConfig(allow=["*"] if restricted else []),
                [],
                allow_non_global=restricted,
            )
        )
        if sys.platform == "linux":
            await self._proxy.start(listener=await self._container_listener())
        else:
            await self._proxy.start("127.0.0.1")
        logger.info(
            "%s: started container %s (image=%s)",
            self.engine,
            self._container,
            self.config.image,
        )

    def host_url(self, url: str) -> str:
        parts = urlsplit(url)
        if not is_loopback_host(parts.hostname or ""):
            return url
        assert self._proxy is not None
        host = "127.0.0.1" if sys.platform == "linux" else _PROXY_HOST
        return self._proxy.callback_url(url, host)

    async def expose(self, port: int) -> str:
        if port != SERVICE_PORT or self._service_url is None:
            raise SandboxError(
                f"{self.engine} publishes only port {SERVICE_PORT}, not {port}"
            )
        return self._service_url

    async def _container_listener(self) -> socket.socket:
        """Create the proxy listener inside the container netns, serviced here."""
        with (
            tempfile.TemporaryDirectory(prefix="vf-proxy-") as directory,
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
                _PASS_LISTENER,
            )
            if helper.exit_code != 0:
                raise SandboxError(
                    f"{self.engine} proxy listener failed: {helper.stderr.strip()}"
                )
            connection, _ = control.accept()
            with connection:
                _, ancillary, *_ = connection.recvmsg(
                    64, socket.CMSG_SPACE(array.array("i").itemsize)
                )
        descriptors = array.array("i")
        descriptors.frombytes(ancillary[0][2][: descriptors.itemsize])
        return socket.socket(fileno=descriptors[0])

    async def prepare_execution(self, routes: list[str] | None) -> None:
        """Allow the declared framework routes, then leave the proxy as the only way out."""
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
                f"awk '$2 == \"{_PROXY_HOST}\" {{ print $1; exit }}' /etc/hosts",
            )
            host = found.stdout.strip()
            if found.exit_code != 0 or not host:
                raise SandboxError(
                    f"could not resolve {_PROXY_HOST} in {self.engine}: {found.stderr.strip()}"
                )
        script = (
            "set -eu; HOST=$1; PORT=$2; "
            # Docker's embedded DNS otherwise forwards requests outside the namespace.
            "ip route add blackhole 127.0.0.11/32 table local; "
            "iptables -F OUTPUT; iptables -A OUTPUT -o lo -j ACCEPT; "
            "iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED "
            "--ctdir REPLY -j ACCEPT; "
            'if [ -n "$HOST" ]; then iptables -A OUTPUT -d "$HOST" '
            '-p tcp --dport "$PORT" -j ACCEPT; fi; '
            "iptables -A OUTPUT -j REJECT"
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
            _NETWORK_IMAGE,
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
        host = "127.0.0.1" if sys.platform == "linux" else _PROXY_HOST
        proxy = f"http://verifiers:{self._proxy.token}@{host}:{self._proxy.port}"
        return {
            "HTTP_PROXY": proxy,
            "HTTPS_PROXY": proxy,
            "http_proxy": proxy,
            "https_proxy": proxy,
            "NO_PROXY": f"localhost,127.0.0.1,{_PROXY_HOST}",
            "no_proxy": f"localhost,127.0.0.1,{_PROXY_HOST}",
        }

    async def teardown(self) -> None:
        if self._proxy is not None:
            await self._proxy.stop()
        await super().teardown()

    def _exec(self, env: dict[str, str], *, stdin: bool = False) -> list[str]:
        assert self._container is not None
        if self.network_restricted and self._cut:
            env = {**env, **self._proxy_env()}
        else:
            values = {**self._image_env, **env}
            exclusions = dict.fromkeys(
                entry.strip()
                for key in ("NO_PROXY", "no_proxy")
                for entry in values.get(key, "").split(",")
                if entry.strip()
            )
            exclusions.update(dict.fromkeys(("localhost", "127.0.0.1", _PROXY_HOST)))
            env = {
                **env,
                "NO_PROXY": ",".join(exclusions),
                "no_proxy": ",".join(exclusions),
            }
        return [
            self.engine,
            "exec",
            *(("-i",) if stdin else ()),
            *(arg for key, value in env.items() for arg in ("--env", f"{key}={value}")),
            "--workdir",
            self.config.workdir,
            self._container,
        ]

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # A setup server outlives the network cut and needs the initially open proxy.
        if self.network_restricted:
            env = {**env, **self._proxy_env()}
        await super().run_background(argv, env, log)

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
