"""Local Docker runtime. A container runs on the bridge network and publishes one
service port. It reaches host-loopback services, such as the interception server,
through an in-process proxy: on Linux the proxy listens in the container's network
namespace, elsewhere on host loopback behind `host.docker.internal`.

For a container with a network policy, the same proxy also carries its egress and
enforces the policy by host name. Before the agent starts, an iptables rule in the
container's network namespace leaves the proxy as the only way out.

The same backend also execs into a task's Compose services, on local Docker or on a
provider VM through that VM's runtime."""

from __future__ import annotations

import array
import asyncio
import contextlib
import csv
import io
import json
import logging
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, ClassVar
from urllib.parse import urlsplit

from verifiers.v1.configs.runtime import DockerConfig, NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.backends.subprocess import SubprocessProcess
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
    parse_gpu,
)
from verifiers.v1.runtimes.egress import EgressProxy, NetworkPolicy, is_loopback_host
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.artifacts import validate_runtime_mounts
from verifiers.v1.utils.scope import run_scope

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from verifiers.v1.runtimes.backends.prime import PrimeConfig

_LINUX = sys.platform == "linux"
_HOST_ALIAS = "host.docker.internal"
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


class DockerRuntimeInfo(DockerConfig, BaseRuntimeInfo):
    pass


async def _communicate(
    *argv: str, input: bytes | None = None, env: dict[str, str] | None = None
) -> tuple[int, bytes, bytes]:
    """Run a host command; explicit env replaces inheritance, cancellation kills it."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        env=env,
        stdin=asyncio.subprocess.PIPE
        if input is not None
        else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await proc.communicate(input)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        await run_shielded(proc.communicate())
        raise
    return proc.returncode or 0, stdout, stderr


async def cli(
    *argv: str, input: bytes | None = None, env: dict[str, str] | None = None
) -> ProgramResult:
    code, stdout, stderr = await _communicate(*argv, input=input, env=env)
    return ProgramResult(
        code, stdout.decode(errors="replace"), stderr.decode(errors="replace")
    )


class ContainerProcess(RuntimeProcess):
    """A process attached through the CLI client. The client does not forward signals,
    so the container runtime delivers them directly to the inner process group."""

    def __init__(
        self,
        process: RuntimeProcess,
        runtime: Runtime,
        pid: int,
        pidfile: str,
    ) -> None:
        self._process = process
        self._runtime = runtime.backend.bind(env={}).with_user(runtime.user)
        self._pid = pid
        self._pidfile = pidfile
        self.stdout, self.stderr = process.stdout, process.stderr

    async def write(self, data: bytes) -> None:
        await self._process.write(data)

    async def wait(self) -> int:
        try:
            return await self._process.wait()
        finally:
            # Removing bookkeeping files must not replace the process result.
            with contextlib.suppress(Exception):
                await run_shielded(self._runtime.run(["rm", "-f", self._pidfile], {}))

    async def poll(self) -> int | None:
        return await self._process.poll()

    async def terminate(self) -> None:
        await self._signal("TERM")

    async def kill(self) -> None:
        await self._signal("KILL")

    async def _signal(self, signal: str) -> None:
        if await self.poll() is not None:
            return
        result = await self._runtime.run(
            [
                "sh",
                "-c",
                'kill -"$1" "-$2" 2>/dev/null || kill -"$1" "$2" 2>/dev/null',
                "vf-signal",
                signal,
                str(self._pid),
            ],
            {},
        )
        if result.exit_code != 0 and await self.poll() is None:
            raise SandboxError(
                f"container process signal failed: {result.stderr.strip()}"
            )


async def _abort_process_startup(
    proc: RuntimeProcess, runtime: Runtime, pidfile: str
) -> str:
    """Kill a partially opened container process and reap its CLI client."""
    runtime = runtime.backend.bind(env={}).with_user(runtime.user)
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
            # An exited CLI client may have left its inner process running.
            await asyncio.wait_for(
                runtime.run(
                    [
                        "sh",
                        "-c",
                        cleanup,
                        "vf-process-cleanup",
                        pidfile,
                    ],
                    {},
                ),
                timeout=5,
            )
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.kill(), 5)
        stderr = b""
        with contextlib.suppress(Exception):
            async with asyncio.timeout(5):
                async for _ in proc.stdout:
                    pass
                stderr = b"".join([chunk async for chunk in proc.stderr])
                await proc.wait()
    return stderr.decode(errors="replace").strip()


class DockerBackend(RuntimeBackend):
    """A container reached through the docker CLI: every operation is an `exec`."""

    SUPPORTS_TASK_IMAGES: ClassVar[bool] = True
    SUPPORTS_USER_SWITCHING: ClassVar[bool] = True
    SUPPORTS_PORT_EXPOSURE: ClassVar[bool] = True
    SUPPORTS_COMPOSE: ClassVar[bool] = True
    is_local = True

    def __init__(
        self,
        config: DockerConfig | PrimeConfig,
        name: str | None = None,
        *,
        host_backend: RuntimeBackend | None = None,
    ) -> None:
        super().__init__(name)
        self.config = config
        self._host_backend = host_backend
        self.is_local = host_backend is None or host_backend.is_local
        self.info = (
            host_backend.info.model_copy(
                update={"image": self.config.image, "workdir": self.config.workdir}
            )
            if host_backend
            else DockerRuntimeInfo(**self.config.model_dump())
        )
        self._container: str | None = None  # our `--name` (used for exec/rm)
        self._service_url: str | None = None
        self._image_env: dict[str, str] = {}
        self._stopped = False
        self._proxy: EgressProxy | None = None
        self._cut = False

    @classmethod
    @contextlib.asynccontextmanager
    async def attach(
        cls,
        config: DockerConfig | PrimeConfig,
        container: str,
        *,
        host_backend: RuntimeBackend | None = None,
        service_url: str | None = None,
    ) -> AsyncIterator[DockerBackend]:
        """Borrow a Compose service container; its caller owns creation and removal.
        `service_url` is the host URL of a local workspace's published service port."""
        if getattr(config, "mounts", {}):
            raise ValueError("Cannot attach runtime mounts to an existing container")
        backend = cls(config, host_backend=host_backend)
        backend._container = container
        backend.info.borrowed = True  # cleanup never removes a borrowed container
        try:
            code, data, stderr = await backend._communicate_host(
                "docker", "inspect", "--format", "{{json .Config}}", container
            )
            if code:
                raise SandboxError(
                    f"Container inspection failed: {stderr.decode(errors='replace')}"
                )
            info = json.loads(data)
            backend._image_env = dict(
                entry.split("=", 1) for entry in info["Env"] or []
            )
            backend.config = config.model_copy(
                update={"image": info["Image"], "workdir": info["WorkingDir"] or "/"}
            )
            if host_backend is None:
                backend.info.id = container
            backend.info.image = backend.config.image
            backend.info.workdir = backend.config.workdir
            backend._service_url = service_url
            if host_backend is None and service_url is not None:
                await backend._start_proxy(
                    NetworkPolicy(NetworkPolicyConfig(allow=[]), [])
                )
            yield backend
        finally:
            await backend.stop()

    @property
    def _restricted(self) -> bool:
        return (
            self._host_backend is None
            and not self.info.borrowed
            and isinstance(self.config, NetworkPolicyConfig)
            and self.config.network_restricted
        )

    @property
    def published_port(self) -> int:
        return SERVICE_PORT

    async def start(self) -> None:
        # Compose keeps its authored workdir; only standalone containers default to /app.
        self.config = self.config.model_copy(
            update={"workdir": self.config.workdir or "/app"}
        )
        self.info.workdir = self.config.workdir
        try:
            version = await cli("docker", "version")
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
                f"docker runtime selected but the engine is not reachable: {detail}{hint}"
            )
        if self._restricted:
            await _ensure_network_image()
        # Engine names are global; cleanup must never target a caller's existing container.
        self._container = f"vf-{uuid.uuid4().hex}"
        options = ["--network", "bridge", "--publish", f"127.0.0.1::{SERVICE_PORT}"]
        if not _LINUX:
            options += ["--add-host", f"{_HOST_ALIAS}:host-gateway"]
        if self._restricted:
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
        if self.config.cpu is not None:
            options += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            options += ["--memory", f"{self.config.memory}g"]
        gpu_type, gpu_count = parse_gpu(self.config.gpu)
        if gpu_count:
            options += ["--gpus", str(gpu_count)]
        env_args = [
            arg
            for key, value in self.env.items()
            for arg in ("--env", f"{key}={value}")
        ]
        for target, mount in getattr(self.config, "mounts", {}).items():
            bind_options = ["type=bind", f"source={mount.source}", f"target={target}"]
            if mount.read_only:
                # Refuse kernels that would leave nested mounts writable.
                bind_options += [
                    "readonly",
                    "bind-recursive=readonly",
                    "bind-propagation=rprivate",
                ]
            value = io.StringIO()
            csv.writer(value).writerow(bind_options)
            options += ["--mount", value.getvalue().removesuffix("\r\n")]
        run = await cli(
            "docker",
            "run",
            "--label",
            f"verifiers.run={run_scope()}",
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
            raise SandboxError(f"docker run failed: {run.stderr.strip()}")
        self.info.id = run.stdout.strip()[:12]  # `run -d` prints the container id
        if gpu_type and gpu_count:
            # Check the devices Docker actually exposed, including on remote daemons.
            gpus = await cli(
                "docker",
                "exec",
                self._container,
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            )
            if gpus.exit_code != 0:
                raise SandboxError(
                    f"Cannot verify Docker GPU type: {(gpus.stderr or gpus.stdout).strip()}"
                )
            names = gpus.stdout.strip().splitlines()
            if len(names) != gpu_count or any(
                not re.search(
                    rf"(?:^|[\s-]){re.escape(gpu_type)}(?:$|[\s-])", name, re.IGNORECASE
                )
                for name in names
            ):
                raise SandboxError(
                    f"Requested {self.config.gpu!r}, but Docker exposed: {', '.join(names) or 'no GPUs'}"
                )
        inspected = await cli(
            "docker", "inspect", "--format", "{{json .Config}}", self._container
        )
        if inspected.exit_code != 0:
            raise SandboxError(
                f"docker environment inspection failed: {inspected.stderr.strip()}"
            )
        container = json.loads(inspected.stdout)
        self._image_env = dict(entry.split("=", 1) for entry in container["Env"] or [])
        # Create a missing workdir, owned by the image's execution user.
        made = await cli(
            "docker",
            "exec",
            "--user",
            "0",
            self._container,
            "sh",
            "-c",
            'if [ ! -d "$1" ]; then mkdir -p "$1" && chown "$2" "$1"; fi',
            "vf-workdir",
            self.config.workdir,
            container["User"] or "0",
        )
        if made.exit_code != 0:
            raise SandboxError(f"docker workdir setup failed: {made.stderr.strip()}")
        await validate_runtime_mounts(self.bind(), [])
        published = await cli("docker", "port", self._container, f"{SERVICE_PORT}/tcp")
        host_port = (published.stdout.split() or [""])[0].rpartition(":")[2]
        if published.exit_code != 0 or not host_port.isdigit():
            detail = (published.stderr or published.stdout).strip()
            raise SandboxError(f"docker did not publish the service port: {detail}")
        self._service_url = f"http://127.0.0.1:{host_port}"
        # An unrestricted container uses the proxy for host callbacks only. A restricted
        # one sends its egress through it too, open until prepare_execution applies the
        # policy.
        await self._start_proxy(
            NetworkPolicy(NetworkPolicyConfig(), [], allow_non_global=True)
            if self._restricted
            else NetworkPolicy(NetworkPolicyConfig(allow=[]), [])
        )
        logger.info(
            "docker: started container %s (image=%s)",
            self._container,
            self.config.image,
        )

    def host_url(self, url: str) -> str:
        if self._host_backend is not None:
            return self._host_backend.host_url(url)
        if self._proxy is None or not is_loopback_host(urlsplit(url).hostname or ""):
            return url
        return self._proxy.callback_url(url, self._proxy_host)

    async def expose(self, port: int) -> str:
        if self._host_backend is not None:
            return await self._host_backend.expose(port)
        if port != SERVICE_PORT or self._service_url is None:
            raise SandboxError(f"docker publishes only port {SERVICE_PORT}, not {port}")
        return self._service_url

    @property
    def _proxy_host(self) -> str:
        """Where the container reaches the proxy: on Linux its listener lives in the
        container's own network namespace."""
        return "127.0.0.1" if _LINUX else _HOST_ALIAS

    async def _start_proxy(self, policy: NetworkPolicy) -> None:
        self._proxy = EgressProxy(policy)
        if _LINUX:
            await self._proxy.start(listener=await self._container_listener())
        else:
            await self._proxy.start("127.0.0.1")

    async def _container_listener(self) -> socket.socket:
        """Create the proxy listener inside the container netns, serviced here."""
        # Reuse the task image offline; only images without Python need the shared helper.
        assert self._container is not None
        python = await cli(
            "docker",
            "exec",
            self._container,
            "python3",
            "-I",
            "-S",
            "-c",
            "import array, socket",
        )
        image = (
            self.config.image
            if python.exit_code == 0
            else "docker.io/library/python:3.11-alpine"
        )
        with (
            tempfile.TemporaryDirectory(prefix="vf-proxy-") as directory,
            socket.socket(socket.AF_UNIX) as control,
        ):
            control.bind(f"{directory}/control.sock")
            control.listen(1)
            helper = await cli(
                "docker",
                "run",
                "--label",
                f"verifiers.run={run_scope()}",
                "--rm",
                "--user",
                "0",
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
                "--entrypoint",
                "python3",
                image,
                "-I",
                "-S",
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
        descriptors = array.array("i")
        descriptors.frombytes(ancillary[0][2][: descriptors.itemsize])
        return socket.socket(fileno=descriptors[0])

    async def prepare_execution(
        self, routes: list[str] | None, network: NetworkPolicyConfig | None = None
    ) -> None:
        """Allow the declared framework routes, then leave the proxy, enforcing the
        configured policy or `network`, as the only way out. `None` routes reopen the
        network for a trusted phase, undoing the cut."""
        if self._host_backend is not None:
            await self._host_backend.prepare_execution(routes, network)
            return
        if self._proxy is None or not self._restricted:
            return
        if routes is None:
            self._proxy.policy = NetworkPolicy(
                NetworkPolicyConfig(), [], allow_non_global=True
            )
            if self._cut:
                await self._net_admin(
                    "ip route del blackhole 127.0.0.11/32 table local; "
                    "iptables -F INPUT; iptables -F OUTPUT",
                    "reopen",
                )
                self._cut = False
            return
        framework = [
            urlsplit(url)._replace(path="", query="", fragment="").geturl()
            for url in routes
        ]
        assert isinstance(self.config, NetworkPolicyConfig)
        self._proxy.policy = NetworkPolicy(network or self.config, framework)
        if self._cut:
            return
        # Off Linux the host proxy has a real address; allow only its listening port.
        proxy_address = ""
        if not _LINUX:
            found = await self.bind().run(
                [
                    "sh",
                    "-c",
                    f"awk '$2 == \"{_HOST_ALIAS}\" {{ print $1; exit }}' /etc/hosts",
                ],
                {},
            )
            proxy_address = found.stdout.strip()
            if found.exit_code != 0 or not proxy_address:
                raise SandboxError(
                    f"could not resolve {_HOST_ALIAS} in docker: {found.stderr.strip()}"
                )
        script = (
            "set -eu; HOST=$1; PORT=$2; "
            "GW=$(ip -4 route show default | awk '/^default via/{print $3; exit}'); "
            # Docker's embedded DNS otherwise forwards requests outside the namespace.
            "ip route add blackhole 127.0.0.11/32 table local; "
            # Published loopback traffic comes from the gateway. Peers must not send
            # requests or get replies.
            "iptables -F INPUT; "
            "iptables -A INPUT -m addrtype --src-type LOCAL -j ACCEPT; "
            'iptables -A INPUT -s "$GW" -j ACCEPT; '
            "iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED "
            "--ctdir REPLY -j ACCEPT; iptables -A INPUT -j REJECT; "
            "iptables -F OUTPUT; iptables -A OUTPUT -o lo -j ACCEPT; "
            "iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED "
            '--ctdir REPLY -d "$GW" -j ACCEPT; '
            'if [ -n "$HOST" ]; then iptables -A OUTPUT -d "$HOST" '
            '-p tcp --dport "$PORT" -j ACCEPT; fi; '
            "iptables -A OUTPUT -j REJECT"
        )
        await self._net_admin(script, "cut", proxy_address, str(self._proxy.port))
        self._cut = True

    async def _net_admin(self, script: str, action: str, *args: str) -> None:
        """Run `script` with NET_ADMIN in the container's network namespace."""
        assert self._container is not None
        result = await cli(
            "docker",
            "run",
            "--label",
            f"verifiers.run={run_scope()}",
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
            action,
            *args,
        )
        if result.exit_code != 0:
            raise SandboxError(
                f"docker network {action} failed: {result.stderr.strip()}"
            )

    async def teardown(self) -> None:
        if self._proxy is not None:
            await self._proxy.stop()
        await super().teardown()

    def _exec_argv(
        self, env: dict[str, str], *, stdin: bool = False, user: str | None = None
    ) -> list[str]:
        assert self._container is not None
        if self._proxy is not None and self._restricted:
            # Every process goes through the proxy, so one started during setup keeps
            # its allowed egress after the cut.
            # No credentials: only this container can reach its proxy, and tools that
            # strip secret-looking variables would drop a URL that carried them.
            proxy = f"http://{self._proxy_host}:{self._proxy.port}"
            exclusions = f"localhost,127.0.0.1,{_HOST_ALIAS}"
            env = {
                **env,
                **dict.fromkeys(
                    ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"), proxy
                ),
                "NO_PROXY": exclusions,
                "no_proxy": exclusions,
            }
            return self._exec_command(env, stdin=stdin, user=user)
        # Callback URLs must bypass any proxy the image or task configures.
        values = {**self._image_env, **env}
        exclusions = dict.fromkeys(
            entry.strip()
            for key in ("NO_PROXY", "no_proxy")
            for entry in values.get(key, "").split(",")
            if entry.strip()
        )
        exclusions.update(dict.fromkeys(("localhost", "127.0.0.1", _HOST_ALIAS)))
        env = {
            **env,
            "NO_PROXY": ",".join(exclusions),
            "no_proxy": ",".join(exclusions),
        }
        return self._exec_command(env, stdin=stdin, user=user)

    def _exec_command(
        self, env: dict[str, str], *, stdin: bool, user: str | None
    ) -> list[str]:
        assert self._container is not None
        return [
            "docker",
            "exec",
            *(("-i",) if stdin else ()),
            *(arg for key, value in env.items() for arg in ("--env", f"{key}={value}")),
            *(("--user", user) if user is not None else ()),
            "--workdir",
            self.config.workdir,
            self._container,
        ]

    def cleanup(self) -> None:
        if self.info.borrowed or self._container is None or self._stopped:
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
                check=False,
            )

    async def _run_host(self, *argv: str) -> ProgramResult:
        if self._host_backend is None:
            return await cli(*argv)
        return await self._host_backend.bind().run(list(argv), {})

    async def _communicate_host(
        self, *argv: str, input: bytes | None = None
    ) -> tuple[int, bytes, bytes]:
        if self._host_backend is None:
            return await _communicate(*argv, input=input)
        # Stage bytes through the provider filesystem, never its text command logs.
        temporary = f"/tmp/vf-io-{uuid.uuid4().hex}"
        command = f"{shlex.join(argv)} > {temporary}.out"
        try:
            if input is not None:
                await self._host_backend.bind().write(f"{temporary}.in", input)
                command += f" < {temporary}.in"
            result = await self._run_host("sh", "-c", command)
            data = await self._host_backend.bind().read(f"{temporary}.out")
            return result.exit_code, data, result.stderr.encode()
        finally:
            # Temporary-file cleanup is best-effort after command completion/failure.
            with contextlib.suppress(Exception):
                await run_shielded(
                    self._run_host("rm", "-f", f"{temporary}.in", f"{temporary}.out")
                )

    async def run(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        return await self._run_host(
            *self._exec_argv(runtime.process_env(env), user=runtime.user), *argv
        )

    async def exec(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        pidfile = f"/tmp/vf-process-{uuid.uuid4().hex}.pid"
        # Give the target its own process group when `setsid -w` is available so
        # terminate()/kill() reap its descendants while the CLI client remains
        # attached if setsid needs to fork. The inner shell records the
        # post-setsid PID before exec preserves it as the target PID.
        wrapper = (
            "if setsid -w true >/dev/null 2>&1; then "
            'exec setsid -w sh -c \'echo $$ > "$1"; shift; exec "$@"\' '
            'vf-process "$@"; '
            'fi; echo $$ > "$1"; shift; exec "$@"'
        )
        command = [
            *self._exec_argv(runtime.process_env(env), user=runtime.user, stdin=True),
            "sh",
            "-c",
            wrapper,
            "vf-process",
            pidfile,
            *argv,
        ]
        if self._host_backend is None:
            proc = SubprocessProcess(
                await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            )
        else:
            proc = await self._host_backend.bind().exec(command, {})
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (30 if self._host_backend is not None else 5)
        try:
            async with asyncio.timeout_at(deadline):
                while True:
                    # An exited client gets one final PID read: short commands may
                    # have written their PID and finished before the first poll.
                    returncode = await proc.poll()
                    ready = await self._run_host(
                        *self._exec_argv({}, user=runtime.user),
                        "cat",
                        pidfile,
                    )
                    if ready.exit_code == 0 and ready.stdout.strip().isdigit():
                        return ContainerProcess(
                            proc, runtime, int(ready.stdout.strip()), pidfile
                        )
                    if returncode is not None:
                        break
                    await asyncio.sleep(0.05)
        except TimeoutError as error:
            stderr = await run_shielded(_abort_process_startup(proc, runtime, pidfile))
            raise SandboxError(
                f"container streaming process failed to start: {stderr or 'PID unavailable'}"
            ) from error
        except BaseException:
            await run_shielded(_abort_process_startup(proc, runtime, pidfile))
            raise
        stderr = await run_shielded(_abort_process_startup(proc, runtime, pidfile))
        raise SandboxError(
            f"container client exited with code {returncode} before reporting its PID: "
            f"{stderr or ready.stderr.strip()}"
        )

    async def run_background(
        self, runtime: Runtime, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # The engine owns the process as a detached exec; no client stays attached.
        command = self._exec_argv(runtime.process_env(env), user=runtime.user)
        command.insert(2, "--detach")
        script = f"exec {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null"
        result = await self._run_host(*command, "sh", "-c", script)
        if result.exit_code != 0:
            raise SandboxError(
                f"docker background process failed: {result.stderr.strip()}"
            )

    async def read(
        self, runtime: Runtime, path: str, max_bytes: int | None = None
    ) -> bytes:
        argv = ["cat"] if max_bytes is None else ["head", "-c", str(max_bytes)]
        code, data, stderr = await self._communicate_host(
            *self._exec_argv({}, user=runtime.user), *argv, "--", path
        )
        if code != 0:
            raise SandboxError(
                f"read {path!r}: {stderr.decode(errors='replace').strip()}"
            )
        return data

    async def write(self, runtime: Runtime, path: str, data: bytes) -> None:
        parent = shlex.quote(str(PurePosixPath(path).parent))
        code, _, stderr = await self._communicate_host(
            *self._exec_argv({}, user=runtime.user, stdin=True),
            "sh",
            "-c",
            f"mkdir -p {parent} && cat > {shlex.quote(path)}",
            input=data,
        )
        if code != 0:
            raise SandboxError(
                f"write {path!r}: {stderr.decode(errors='replace').strip()}"
            )


async def _ensure_network_image() -> None:
    """Cache the iptables image, so the cut never needs package mirrors."""
    if (await cli("docker", "image", "inspect", _NETWORK_IMAGE)).exit_code == 0:
        return
    with tempfile.TemporaryDirectory(prefix="vf-network-") as directory:
        built = await cli(
            "docker",
            "build",
            "--tag",
            _NETWORK_IMAGE,
            "--file",
            "-",
            directory,
            input=b"FROM docker.io/library/alpine:3.22\nRUN apk add --no-cache iptables\n",
        )
    if built.exit_code != 0:
        raise SandboxError(f"docker network image build failed: {built.stderr.strip()}")
