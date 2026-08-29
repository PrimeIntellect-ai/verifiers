"""Local OCI-container runtime backed by Docker or Podman."""

import array
import asyncio
import contextlib
import json
import logging
import os
import shlex
import socket
import subprocess
import sys
import tempfile
from pathlib import PurePosixPath
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import model_validator

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.attached import open_attached_process
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeProcess,
    parse_gpu,
)
from verifiers.v1.runtimes.docker.egress import (
    HOST_ALIAS,
    EgressProxy,
    NetworkPolicy,
    is_loopback_host,
)
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)


class _ContainerConfig(NetworkPolicyConfig):
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float | None = None
    """Pin the container to this many CPU cores (`--cpus`). None = unlimited."""
    memory: float | None = None
    """Hard memory limit in GB (`--memory`). None = unlimited."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "2". Docker uses the count with `--gpus`;
    Podman maps it to NVIDIA CDI device indexes. None = none."""
    disk: float | None = None
    """Advisory disk request in GB. OCI engines have no portable per-container size
    limit, so this is accepted (so a task can declare it without a warning) but not
    enforced."""


class DockerConfig(_ContainerConfig):
    type: Literal["docker"] = "docker"


class PodmanConfig(_ContainerConfig):
    type: Literal["podman"] = "podman"
    image: str = "docker.io/library/python:3.11-slim"

    @model_validator(mode="after")
    def validate_support(self) -> Self:
        if self.network_restricted:
            raise ValueError(
                "restricted network policies are not supported by the podman runtime; "
                "the network cut has not been validated against rootless Podman"
            )
        if not 0 <= parse_gpu(self.gpu)[1] <= 1024:
            raise ValueError(
                "podman supports between 0 and 1024 indexed GPUs per container"
            )
        return self


class DockerRuntimeInfo(DockerConfig, BaseRuntimeInfo):
    pass


class PodmanRuntimeInfo(PodmanConfig, BaseRuntimeInfo):
    pass


def _environment_args(env: dict[str, str]) -> list[str]:
    return [arg for key, value in env.items() for arg in ("--env", f"{key}={value}")]


_DOCKER_HOST = "host.docker.internal"
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


class _ContainerRuntime(Runtime):
    engine: Literal["docker", "podman"]
    info_class: type[BaseRuntimeInfo]
    callback_host: str

    def __init__(self, config: _ContainerConfig, name: str | None = None) -> None:
        super().__init__(name)
        # Task resource overlays use model_copy(), so validate the resolved values
        # before constructing an external container command.
        self.config = type(config).model_validate(config.model_dump())
        self.info = self.info_class(**self.config.model_dump())
        self._container: str | None = None  # our `--name` (used for exec/rm)
        self._image_env: dict[str, str] = {}
        self._proxy: EgressProxy | None = None
        self._proxy_host_ip: str | None = None
        self._service_url: str | None = None
        self._stopped = False
        self._cut = False

    @property
    def published_port(self) -> int | None:
        return SERVICE_PORT

    async def _run_cli(self, *args: str) -> ProgramResult:
        process = await asyncio.create_subprocess_exec(
            self.engine,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
        except BaseException:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await run_shielded(process.communicate())
            raise
        return ProgramResult(
            exit_code=process.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    async def start(self) -> None:
        try:
            version_field = "Server" if self.engine == "docker" else "Client"
            version = await self._run_cli(
                "version", "--format", f"{{{{.{version_field}.Version}}}}"
            )
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
            unavailable = (
                "the Docker daemon is not reachable"
                if self.engine == "docker"
                else "Podman is not usable"
            )
            raise RuntimeError(
                f"{self.engine} runtime selected but {unavailable}: {detail}{hint}"
            )
        podman_host = None
        if self.engine == "podman":
            system = await self._run_cli("info", "--format", "json")
            if system.exit_code != 0:
                raise SandboxError(
                    f"Podman prerequisite inspection failed: {system.stderr.strip()}"
                )
            try:
                podman_host = json.loads(system.stdout)["host"]
                remote = podman_host["serviceIsRemote"]
            except (KeyError, TypeError, ValueError) as error:
                raise SandboxError(
                    "Podman did not report its backend location"
                ) from error
            if self.config.cpu is not None or self.config.memory is not None:
                try:
                    rootless = podman_host["security"]["rootless"]
                    cgroup_version = podman_host["cgroupVersion"]
                except (KeyError, TypeError, ValueError) as error:
                    raise SandboxError(
                        "Podman did not report the resource-limit prerequisites"
                    ) from error
                if rootless and cgroup_version == "v1":
                    raise SandboxError(
                        "rootless Podman on cgroups v1 does not support CPU or memory limits"
                    )
            if remote and sys.platform == "linux":
                raise SandboxError("Podman remote clients are not supported")
            if remote:
                connections = await self._run_cli(
                    "system", "connection", "list", "--format", "json"
                )
                local_machine = False
                if connections.exit_code == 0:
                    try:
                        entries = json.loads(connections.stdout)
                        container_host = os.environ.get("CONTAINER_HOST")
                        if container_host is not None:
                            selected = next(
                                entry
                                for entry in entries
                                if entry["URI"] == container_host
                            )
                        else:
                            connection_name = os.environ.get("CONTAINER_CONNECTION")
                            selected = next(
                                entry
                                for entry in entries
                                if entry["Name"] == connection_name
                                or (connection_name is None and entry["Default"])
                            )
                        local_machine = selected["IsMachine"] is True
                    except (KeyError, StopIteration, TypeError, ValueError) as error:
                        raise SandboxError(
                            "Podman did not identify its active connection"
                        ) from error
                if not local_machine:
                    raise SandboxError(
                        "Podman remote clients are supported only through a local "
                        "Podman Machine"
                    )
        limits: list[str] = []
        if self.config.cpu is not None:
            limits += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            limits += ["--memory", f"{self.config.memory}g"]
        _, gpu_count = parse_gpu(self.config.gpu)
        if gpu_count:
            if self.engine == "docker":
                limits += ["--gpus", str(gpu_count)]
            else:
                assert podman_host is not None
                try:
                    rootless = podman_host["security"]["rootless"]
                    selinux = podman_host["security"]["selinuxEnabled"]
                    oci_runtime = podman_host["ociRuntime"]["name"]
                except (KeyError, TypeError, ValueError) as error:
                    raise SandboxError(
                        "Podman did not report the GPU runtime prerequisites"
                    ) from error
                if selinux:
                    raise SandboxError(
                        "Podman GPU passthrough is not supported with SELinux enabled"
                    )
                if rootless and oci_runtime != "crun":
                    raise SandboxError(
                        "rootless Podman GPU passthrough requires the crun OCI runtime"
                    )
                for index in range(gpu_count):
                    limits += ["--device", f"nvidia.com/gpu={index}"]
                if rootless:
                    limits += ["--group-add", "keep-groups"]
        restricted = self.network_restricted
        options = ["--publish", f"127.0.0.1::{SERVICE_PORT}"]
        if self.engine == "podman":
            options += ["--http-proxy=false", "--network", "private"]
        else:
            options += ["--network", "bridge"]
        if self.engine == "docker" and restricted:
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
                options += ["--add-host", f"{_DOCKER_HOST}:host-gateway"]
        elif self.engine == "docker":
            options += ["--add-host", f"{HOST_ALIAS}:host-gateway"]
        self._container = self.name
        run = await self._run_cli(
            "run",
            "--detach",
            *options,
            *limits,
            *_environment_args(self.env),
            "--workdir",
            self.config.workdir,
            "--entrypoint",
            "sleep",
            "--name",
            self.name,
            self.config.image,
            "infinity",
        )
        if run.exit_code != 0:
            raise SandboxError(f"{self.engine} run failed: {run.stderr.strip()}")
        self.info.id = run.stdout.strip()[:12]
        try:
            image_env = await self._run_cli(
                "inspect", "--format", "{{json .Config.Env}}", self._container
            )
            if image_env.exit_code != 0:
                raise SandboxError(
                    f"{self.engine} environment inspection failed: "
                    f"{image_env.stderr.strip()}"
                )
            try:
                values = json.loads(image_env.stdout)
                self._image_env = dict(value.split("=", 1) for value in values or [])
            except (TypeError, ValueError) as error:
                raise SandboxError(
                    f"{self.engine} returned invalid container environment"
                ) from error
            published = await self._run_cli(
                "port", self._container, f"{SERVICE_PORT}/tcp"
            )
            lines = published.stdout.strip().splitlines()
            host, separator, port = lines[0].rpartition(":") if lines else ("", "", "")
            if (
                published.exit_code != 0
                or host != "127.0.0.1"
                or not separator
                or not port.isdigit()
            ):
                detail = (published.stderr or published.stdout).strip()
                raise SandboxError(
                    "container engine did not publish the runtime service port on "
                    f"host loopback: {detail}"
                )
            self._service_url = f"http://127.0.0.1:{port}"
            # Restricted runtimes enforce policy through the proxy; unrestricted
            # runtimes use it only for tokenized callbacks to host-loopback services.
            self._proxy = EgressProxy(
                NetworkPolicy(
                    NetworkPolicyConfig(),
                    [HOST_ALIAS] if restricted else [],
                    allow_non_global=restricted,
                )
            )
            if not restricted:
                host_gateway = None
                required_alias = (
                    self.callback_host
                    if self.engine == "docker" or sys.platform == "linux"
                    else None
                )
                if required_alias is not None:
                    hosts = await self._run_cli(
                        "exec", self._container, "cat", "/etc/hosts"
                    )
                    for line in hosts.stdout.splitlines():
                        fields = line.split()
                        if required_alias in fields[1:]:
                            host_gateway = fields[0]
                            break
                    if hosts.exit_code != 0 or host_gateway is None:
                        raise SandboxError(
                            f"{self.engine} did not provide the {required_alias!r} host "
                            "mapping required for runtime callbacks"
                        )
                await self._proxy.start(
                    host_gateway if sys.platform == "linux" else "127.0.0.1"
                )
            elif sys.platform == "linux":
                await self._proxy.start(listener=await self._container_listener())
            else:
                host = await self._run_cli(
                    "exec",
                    self._container,
                    "sh",
                    "-c",
                    f"awk '$2 == \"{_DOCKER_HOST}\" {{ print $1; exit }}' /etc/hosts",
                )
                self._proxy_host_ip = host.stdout.strip()
                if host.exit_code != 0 or not self._proxy_host_ip:
                    raise SandboxError(
                        f"could not resolve {_DOCKER_HOST} in Docker: {host.stderr.strip()}"
                    )
                await self._proxy.start("127.0.0.1")
        except BaseException:
            try:
                if self._proxy is not None:
                    await self._proxy.stop()
            finally:
                await asyncio.to_thread(self.cleanup)
            raise
        logger.info(
            "%s: started container %s (image=%s)",
            self.engine,
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
                helper = await self._run_cli(
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
        descriptor = array.array("i")
        descriptor.frombytes(ancillary[0][2][: descriptor.itemsize])
        listener = socket.socket(fileno=descriptor[0])
        listener.setblocking(False)
        return listener

    def host_url(self, url: str) -> str:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if not is_loopback_host(host):
            return url
        if parsed.scheme != "http":
            raise ValueError(
                f"{self.engine} host callbacks support only http URLs; transparent "
                "TLS forwarding is unavailable across a private network namespace"
            )
        if self.network_restricted:
            if not (
                host == "localhost"
                or host.endswith(".localhost")
                or host.startswith("127.")
            ):
                raise ValueError(
                    "restricted Docker host callbacks require an IPv4 loopback URL"
                )
            userinfo, separator, _ = parsed.netloc.rpartition("@")
            netloc = f"{userinfo}{separator}{HOST_ALIAS}"
            if parsed.port is not None:
                netloc = f"{netloc}:{parsed.port}"
            callback = parsed._replace(netloc=netloc).geturl()
            assert self._proxy is not None
            route = urlsplit(callback)._replace(path="", query="", fragment="").geturl()
            if route not in self._proxy.policy.routes:
                self._proxy.policy.routes.append(route)
            return callback
        assert self._proxy is not None
        return self._proxy.callback_url(url, self.callback_host)

    async def expose(self, port: int) -> str:
        if port != SERVICE_PORT or self._service_url is None:
            raise SandboxError(
                f"{self.engine} service port {port} was not published by this runtime"
            )
        return self._service_url

    async def prepare_execution(self, routes: list[str] | None) -> None:
        """Allow the declared framework routes, then leave the proxy as the only route."""
        if not self.network_restricted:
            return
        assert self._proxy is not None
        if routes is None:
            self._proxy.policy = NetworkPolicy(
                NetworkPolicyConfig(), [HOST_ALIAS], allow_non_global=True
            )
            return
        framework = [
            urlsplit(url)._replace(path="", query="", fragment="").geturl()
            for url in routes
        ]
        self._proxy.policy = NetworkPolicy(self.config, framework)
        if self._cut:
            return
        script = (
            "set -eu; HOST=$1; PORT=$2; "
            "apk add --no-cache iptables >/dev/null; "
            "ROUTE=$(ip -4 route show default); "
            "DEV=$(printf '%s\\n' \"$ROUTE\" | awk '{ for (i = 1; i <= NF; i++) "
            'if ($i == "dev") { print $(i + 1); exit } }\'); '
            "GW=$(printf '%s\\n' \"$ROUTE\" | awk '{ for (i = 1; i <= NF; i++) "
            'if ($i == "via") { print $(i + 1); exit } }\'); '
            "ip -4 route flush table main; ip -6 route flush table main || true; "
            "ip route add blackhole 127.0.0.11/32 table local; "
            'if [ -n "$GW" ]; then '
            'ip -4 route add "$GW/32" dev "$DEV"; '
            'ip -4 route add default via "$GW" dev "$DEV"; '
            'else ip -4 route add default dev "$DEV"; fi; '
            "iptables -F OUTPUT; iptables -A OUTPUT -o lo -j ACCEPT; "
            "iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED "
            "--ctdir REPLY -j ACCEPT; "
            'if [ -n "$HOST" ]; then iptables -A OUTPUT -d "$HOST" '
            '-p tcp --dport "$PORT" -j ACCEPT; fi; '
            "iptables -A OUTPUT -j REJECT"
        )
        cut = await self._run_cli(
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
            self._proxy_host_ip or "",
            str(self._proxy.port),
        )
        if cut.exit_code != 0:
            raise SandboxError(
                f"{self.engine} network cut failed: {cut.stderr.strip()}"
            )
        self._cut = True

    def _proxy_env(self) -> dict[str, str]:
        assert self._proxy is not None
        host = "127.0.0.1" if sys.platform == "linux" else _DOCKER_HOST
        proxy = f"http://verifiers:{self._proxy.token}@{host}:{self._proxy.port}"
        return {
            "HTTP_PROXY": proxy,
            "HTTPS_PROXY": proxy,
            "http_proxy": proxy,
            "https_proxy": proxy,
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
        }

    def _container_env(
        self, env: dict[str, str], *, use_policy_proxy: bool
    ) -> dict[str, str]:
        env = self.process_env(env)
        if use_policy_proxy:
            return {**env, **self._proxy_env()}
        if self.network_restricted:
            return env
        no_proxy = dict.fromkeys(
            entry.strip()
            for key in ("NO_PROXY", "no_proxy")
            for value in (env.get(key, ""), self._image_env.get(key, ""))
            for entry in value.split(",")
            if entry.strip()
        )
        no_proxy.update(dict.fromkeys(("localhost", "127.0.0.1", self.callback_host)))
        bypass = ",".join(no_proxy)
        return {**env, "NO_PROXY": bypass, "no_proxy": bypass}

    async def teardown(self) -> None:
        if self._proxy is not None:
            await self._proxy.stop()
        await super().teardown()

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        env_args = _environment_args(
            self._container_env(env, use_policy_proxy=self._cut)
        )
        return await self._run_cli(
            "exec", *env_args, "--workdir", self.config.workdir, self._container, *argv
        )

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        assert self._container is not None
        env_args = _environment_args(
            self._container_env(env, use_policy_proxy=self._cut)
        )

        async def runtime_exec(command: list[str]) -> ProgramResult:
            return await self._run_cli("exec", self._container, *command)

        return await open_attached_process(
            argv,
            command=[
                self.engine,
                "exec",
                "-i",
                *env_args,
                "--workdir",
                self.config.workdir,
                self._container,
            ],
            runtime_exec=runtime_exec,
            runtime_name=self.engine,
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # A detached restricted server survives the cut, so it needs the policy proxy.
        env_args = _environment_args(
            self._container_env(env, use_policy_proxy=self.network_restricted)
        )
        inner = f"{' '.join(shlex.quote(a) for a in argv)} > {shlex.quote(log)} 2>&1"
        run = await self._run_cli(
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
            raise SandboxError(f"{self.engine} exec -d failed: {run.stderr.strip()}")

    async def _read(self, path: str) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            self.engine,
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
            self.engine,
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
        self._stopped = True
        logger.debug("%s: removing container %s", self.engine, self._container)
        with contextlib.suppress(Exception):
            subprocess.run(
                [self.engine, "rm", "--force", self._container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )


class DockerRuntime(_ContainerRuntime):
    engine = "docker"
    info_class = DockerRuntimeInfo
    callback_host = HOST_ALIAS


class PodmanRuntime(_ContainerRuntime):
    engine = "podman"
    info_class = PodmanRuntimeInfo
    callback_host = "host.containers.internal"
