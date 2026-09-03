"""Local Docker runtime with optional execution-time URL filtering."""

import array
import asyncio
import contextlib
import logging
import posixpath
import shlex
import socket
import subprocess
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeProcess,
    parse_gpu,
)
from verifiers.v1.runtimes.docker.egress import HOST_ALIAS, EgressProxy, NetworkPolicy
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

_SNAPSHOT_PREFIX = "docker-workspace:v1:"
_snapshot_store: tempfile.TemporaryDirectory[str] | None = None


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


def _snapshot_id(ref: str) -> str:
    if not ref.startswith(_SNAPSHOT_PREFIX):
        raise SandboxError(f"invalid Docker workspace snapshot reference: {ref!r}")
    raw = ref.removeprefix(_SNAPSHOT_PREFIX)
    try:
        snapshot_id = uuid.UUID(raw)
    except ValueError as e:
        raise SandboxError(
            f"invalid Docker workspace snapshot reference: {ref!r}"
        ) from e
    if snapshot_id.hex != raw:
        raise SandboxError(f"invalid Docker workspace snapshot reference: {ref!r}")
    return snapshot_id.hex


def _snapshot_path(ref: str, *, create: bool = False) -> Path:
    snapshot_id = _snapshot_id(ref)

    global _snapshot_store
    if _snapshot_store is None:
        if not create:
            raise SandboxError(f"Docker workspace snapshot is unavailable: {ref}")
        _snapshot_store = tempfile.TemporaryDirectory(prefix="vf-docker-snapshots-")
    return Path(_snapshot_store.name) / f"{snapshot_id}.tar"


async def _archive_workspace(container: str, workdir: str, path: Path) -> None:
    with path.open("wb") as archive:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "--user",
            "0",
            "--workdir",
            "/",
            container,
            "tar",
            "-C",
            workdir,
            "-cf",
            "-",
            ".",
            stdout=archive,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await proc.communicate()
        except BaseException:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            await run_shielded(proc.communicate())
            raise
    if proc.returncode != 0:
        raise SandboxError(
            f"docker workspace snapshot failed: {stderr.decode(errors='replace').strip()}"
        )


async def _extract_workspace(container: str, workdir: str, path: Path) -> None:
    with path.open("rb") as archive:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "--user",
            "0",
            "--workdir",
            "/",
            container,
            "tar",
            "-C",
            workdir,
            "-xf",
            "-",
            stdin=archive,
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
    if proc.returncode != 0:
        detail = (stderr or stdout).decode(errors="replace").strip()
        raise SandboxError(f"docker workspace restore failed: {detail}")


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
        restricted = self.network_restricted
        if restricted:
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
        env_args = [
            arg
            for key, value in self.env.items()
            for arg in ("--env", f"{key}={value}")
        ]
        run = await docker(
            "run",
            "--detach",
            *network,
            *limits,
            *env_args,
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
        if restricted:
            # Setup is trusted; colocated servers fetch their task from host interception
            # before the final framework routes are known.
            self._proxy = EgressProxy(
                NetworkPolicy(
                    NetworkPolicyConfig(), [HOST_ALIAS], allow_non_global=True
                )
            )
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
        # Keep numeric loopback container-local; the proxy maps this reserved name to
        # the Verifiers process's loopback for interception and host-local MCP routes.
        if self.network_restricted and host in ("127.0.0.1", "localhost"):
            return url.replace(host, HOST_ALIAS, 1)
        if (
            not self.network_restricted
            and sys.platform != "linux"
            and host in ("127.0.0.1", "localhost")
        ):
            return url.replace(host, "host.docker.internal", 1)
        return url

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
        proxy = f"http://verifiers:{self._proxy.token}@{host}:{self._proxy.port}"
        return {
            "HTTP_PROXY": proxy,
            "HTTPS_PROXY": proxy,
            "http_proxy": proxy,
            "https_proxy": proxy,
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
        }

    async def teardown(self) -> None:
        if self._proxy is not None:
            await self._proxy.stop()
        await super().teardown()

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        env = {**self.process_env(env), **(self._proxy_env() if self._cut else {})}
        env_args = [arg for k, v in env.items() for arg in ("--env", f"{k}={v}")]
        return await docker(
            "exec", *env_args, "--workdir", self.config.workdir, self._container, *argv
        )

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        assert self._container is not None
        env = {**self.process_env(env), **(self._proxy_env() if self._cut else {})}
        env_args = [
            arg for key, value in env.items() for arg in ("--env", f"{key}={value}")
        ]
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
        # Detached servers survive the cut, so they need the initially permissive proxy.
        env = {**self.process_env(env), **self._proxy_env()}
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

    def _snapshot_workdir(self) -> str:
        workdir = posixpath.normpath(self.config.workdir)
        if not workdir.startswith("/") or not workdir.lstrip("/"):
            raise SandboxError(
                "Docker workspace snapshots require an absolute, non-root workdir"
            )
        return workdir

    async def snapshot(self) -> str:
        """Stream the complete workspace into process-local host storage."""
        if self._container is None or self._stopped:
            raise SandboxError("cannot snapshot a Docker container that is not running")
        snapshot_id = uuid.uuid4().hex
        ref = f"{_SNAPSHOT_PREFIX}{snapshot_id}"
        path = _snapshot_path(ref, create=True)
        pending = path.with_suffix(".tmp")
        try:
            await _archive_workspace(
                self._container,
                self._snapshot_workdir(),
                pending,
            )
            pending.replace(path)
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
        logger.info("docker: snapshotted workspace %s -> %s", self._container, ref)
        return ref

    async def restore(self, ref: str) -> None:
        """Replace the workspace with a compatible Docker snapshot."""
        if self._container is None or self._stopped:
            raise SandboxError("cannot restore a Docker container that is not running")
        path = _snapshot_path(ref)
        if not path.is_file():
            raise SandboxError(f"Docker workspace snapshot is unavailable: {ref}")
        workdir = self._snapshot_workdir()
        nonce = uuid.uuid4().hex
        staging = f"{workdir}.vf-restore-{nonce}"
        backup = f"{workdir}.vf-backup-{nonce}"
        prepare = await docker(
            "exec",
            "--user",
            "0",
            "--workdir",
            "/",
            self._container,
            "mkdir",
            "--",
            staging,
        )
        if prepare.exit_code != 0:
            raise SandboxError(
                "docker workspace restore failed to create staging directory: "
                f"{(prepare.stderr or prepare.stdout).strip()}"
            )
        try:
            await _extract_workspace(self._container, staging, path)
        except BaseException:
            await run_shielded(self._remove_restore_staging(staging))
            raise

        swap = await run_shielded(
            docker(
                "exec",
                "--user",
                "0",
                "--workdir",
                "/",
                self._container,
                "sh",
                "-c",
                (
                    "work=$1 staging=$2 backup=$3; "
                    'if ! mv -- "$work" "$backup"; then '
                    'echo "failed to preserve existing workspace" >&2; exit 1; fi; '
                    'mv -- "$staging" "$work"; rc=$?; '
                    'if [ "$rc" -eq 0 ]; then rm -rf -- "$backup" || true; exit 0; fi; '
                    'if mv -- "$backup" "$work"; then '
                    'echo "failed to install snapshot; original workspace restored" >&2; '
                    'else echo "failed to install snapshot; original workspace retained at $backup" >&2; fi; '
                    'exit "$rc"'
                ),
                "vf-restore",
                workdir,
                staging,
                backup,
            )
        )
        if swap.exit_code != 0:
            await self._remove_restore_staging(staging)
            raise SandboxError(
                f"docker workspace restore failed to swap {workdir}: "
                f"{(swap.stderr or swap.stdout).strip()}"
            )
        logger.info("docker: restored workspace %s <- %s", self._container, ref)

    async def _remove_restore_staging(self, staging: str) -> None:
        cleanup = await docker(
            "exec",
            "--user",
            "0",
            "--workdir",
            "/",
            self._container,
            "rm",
            "-rf",
            "--",
            staging,
        )
        if cleanup.exit_code != 0:
            logger.warning(
                "docker: failed to remove restore staging %s: %s",
                staging,
                (cleanup.stderr or cleanup.stdout).strip(),
            )

    async def delete_snapshot(self, ref: str) -> None:
        """Delete a process-local Docker workspace snapshot, idempotently."""
        snapshot_id = _snapshot_id(ref)
        if _snapshot_store is not None:
            (Path(_snapshot_store.name) / f"{snapshot_id}.tar").unlink(missing_ok=True)

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
