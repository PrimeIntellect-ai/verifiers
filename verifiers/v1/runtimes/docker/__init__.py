"""Local Docker runtime with optional execution-time URL filtering."""

import array
import asyncio
import contextlib
import json
import logging
import shlex
import socket
import subprocess
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
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


class DockerConfig(NetworkPolicyConfig):
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


class DockerRuntimeInfo(DockerConfig, BaseRuntimeInfo):
    pass


async def _read_stream(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    while chunk := await reader.read(64 * 1024):
        yield chunk


class DockerProcess(RuntimeProcess):
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        container: str,
        pid: int,
    ) -> None:
        self._process = process
        self._container = container
        self._pid = pid
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        self._stdin = process.stdin
        self.stdout = _read_stream(process.stdout)
        self.stderr = _read_stream(process.stderr)

    async def write(self, data: bytes) -> None:
        self._stdin.write(data)
        await self._stdin.drain()

    async def wait(self) -> int:
        return await self._process.wait()

    async def terminate(self) -> None:
        await self._signal("TERM")

    async def kill(self) -> None:
        await self._signal("KILL")

    async def _signal(self, signal: str) -> None:
        if self._process.returncode is not None:
            return
        result = await docker(
            "exec",
            self._container,
            "sh",
            "-c",
            'kill -"$1" "-$2" 2>/dev/null || kill -"$1" "$2"',
            "vf-signal",
            signal,
            str(self._pid),
        )
        if result.exit_code != 0 and self._process.returncode is None:
            raise SandboxError(
                f"docker exec process signal failed: {result.stderr.strip()}"
            )


async def docker(*args: str) -> ProgramResult:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await proc.communicate()
    except BaseException:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        await run_shielded(proc.communicate())
        raise
    return ProgramResult(
        exit_code=proc.returncode or 0,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )


def _environment_args(env: dict[str, str]) -> list[str]:
    return [arg for key, value in env.items() for arg in ("--env", f"{key}={value}")]


async def _abort_process_startup(
    proc: asyncio.subprocess.Process, container: str, pidfile: str
) -> str:
    """Kill a partially opened container process and reap its local docker client."""
    # The target normally writes its PID immediately, but cancellation can win
    # that race. Wait briefly for the file before signalling the process group.
    cleanup = (
        'i=0; while [ "$i" -lt 20 ]; do '
        'if [ -s "$1" ]; then pid=$(cat "$1"); '
        'kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; '
        'rm -f "$1"; exit 0; fi; '
        "i=$((i + 1)); sleep 0.05; done; exit 1"
    )
    try:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                docker(
                    "exec",
                    container,
                    "sh",
                    "-c",
                    cleanup,
                    "vf-process-cleanup",
                    pidfile,
                ),
                timeout=2,
            )
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        _, stderr = await proc.communicate()
    return stderr.decode(errors="replace").strip()


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
        self._image_env: dict[str, str] = {}
        self._proxy: EgressProxy | None = None
        self._proxy_host_ip: str | None = None
        self._service_url: str | None = None
        self._stopped = False
        self._cut = False

    @property
    def published_port(self) -> int | None:
        return SERVICE_PORT

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
        limits: list[str] = []
        if self.config.cpu is not None:
            limits += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            limits += ["--memory", f"{self.config.memory}g"]
        _, gpu_count = parse_gpu(self.config.gpu)
        if gpu_count:
            limits += ["--gpus", str(gpu_count)]
        restricted = self.network_restricted
        options = [
            "--network",
            "bridge",
            "--publish",
            f"127.0.0.1::{SERVICE_PORT}",
        ]
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
        else:
            options += ["--add-host", f"{HOST_ALIAS}:host-gateway"]
        self._container = self.name
        run = await docker(
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
            raise SandboxError(f"docker run failed: {run.stderr.strip()}")
        self.info.id = run.stdout.strip()[:12]
        try:
            image_env = await docker(
                "inspect", "--format", "{{json .Config.Env}}", self._container
            )
            if image_env.exit_code != 0:
                raise SandboxError(
                    f"docker environment inspection failed: {image_env.stderr.strip()}"
                )
            try:
                values = json.loads(image_env.stdout)
                self._image_env = dict(value.split("=", 1) for value in values or [])
            except (TypeError, ValueError) as error:
                raise SandboxError(
                    "docker returned invalid container environment"
                ) from error
            published = await docker("port", self._container, f"{SERVICE_PORT}/tcp")
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
                    "container engine did not publish the Docker runtime service port "
                    f"on host loopback: {detail}"
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
                hosts = await docker("exec", self._container, "cat", "/etc/hosts")
                host_gateway = None
                for line in hosts.stdout.splitlines():
                    fields = line.split()
                    if HOST_ALIAS in fields[1:]:
                        host_gateway = fields[0]
                        break
                if hosts.exit_code != 0 or host_gateway is None:
                    raise SandboxError(
                        "container engine did not provide the host-gateway mapping "
                        "required for Docker runtime callbacks"
                    )
                await self._proxy.start(
                    host_gateway if sys.platform == "linux" else "127.0.0.1"
                )
            elif sys.platform == "linux":
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
        except BaseException:
            try:
                if self._proxy is not None:
                    await self._proxy.stop()
            finally:
                await asyncio.to_thread(self.cleanup)
            raise
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
                    "--volume",
                    f"{directory}:/run/vf:Z",
                    "docker.io/library/python:3.11-alpine",
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
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if not is_loopback_host(host):
            return url
        if parsed.scheme != "http":
            raise ValueError(
                "Docker host callbacks support only http URLs; transparent TLS "
                "forwarding is unavailable across a private network namespace"
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
        return self._proxy.callback_url(url)

    async def expose(self, port: int) -> str:
        if port != SERVICE_PORT or self._service_url is None:
            raise SandboxError(
                f"docker service port {port} was not published by this runtime"
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
        cut = await docker(
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
            raise SandboxError(f"docker network cut failed: {cut.stderr.strip()}")
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
        no_proxy.update(dict.fromkeys(("localhost", "127.0.0.1", HOST_ALIAS)))
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
        return await docker(
            "exec", *env_args, "--workdir", self.config.workdir, self._container, *argv
        )

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        assert self._container is not None
        env_args = _environment_args(
            self._container_env(env, use_policy_proxy=self._cut)
        )
        pidfile = f"/tmp/vf-process-{uuid.uuid4().hex}.pid"
        # Give the target its own process group when `setsid -w` is available so
        # terminate()/kill() reap its descendants while docker exec remains
        # attached if setsid needs to fork. The inner shell records the
        # post-setsid PID before exec preserves it as the target PID.
        wrapper = (
            "if setsid -w true >/dev/null 2>&1; then "
            'exec setsid -w sh -c \'echo $$ > "$1"; shift; exec "$@"\' '
            'vf-process "$@"; '
            'fi; echo $$ > "$1"; shift; exec "$@"'
        )
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            *env_args,
            "--workdir",
            self.config.workdir,
            self._container,
            "sh",
            "-c",
            wrapper,
            "vf-process",
            pidfile,
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        try:
            while True:
                ready = await docker("exec", self._container, "cat", pidfile)
                if ready.exit_code == 0 and ready.stdout.strip().isdigit():
                    return DockerProcess(
                        proc, self._container, int(ready.stdout.strip())
                    )
                if proc.returncode is not None or loop.time() >= deadline:
                    break
                await asyncio.sleep(0.05)
        except BaseException:
            await run_shielded(_abort_process_startup(proc, self._container, pidfile))
            raise

        stderr = await run_shielded(
            _abort_process_startup(proc, self._container, pidfile)
        )
        detail = stderr or ready.stderr.strip()
        raise SandboxError(
            f"docker live process failed to start: {detail or 'PID unavailable'}"
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # A detached restricted server survives the cut, so it needs the policy proxy.
        env_args = _environment_args(
            self._container_env(env, use_policy_proxy=self.network_restricted)
        )
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

    async def _read(self, path: str) -> bytes:
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
        self._stopped = True
        logger.debug("docker: removing container %s", self._container)
        with contextlib.suppress(Exception):
            subprocess.run(
                ["docker", "rm", "--force", self._container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
