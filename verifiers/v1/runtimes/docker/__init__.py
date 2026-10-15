"""Local Docker runtime: a bridged container with one published service port and a
relay for host callbacks. Docker enforces no network policy."""

from __future__ import annotations

import array
import contextlib
import json
import logging
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar, Literal
from urllib.parse import urlsplit

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    RuntimeBackend,
    parse_gpu,
)
from verifiers.v1.runtimes.container import ContainerBackend, ContainerConfig, cli
from verifiers.v1.runtimes.docker.callbacks import CallbackRelay, is_loopback_host
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.scope import run_scope

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from verifiers.v1.runtimes.modal import ModalConfig
    from verifiers.v1.runtimes.prime import PrimeConfig


class DockerConfig(ContainerConfig):
    type: Literal["docker"] = "docker"


class DockerRuntimeInfo(DockerConfig, BaseRuntimeInfo):
    pass


_PROXY_HOST = "host.docker.internal"
_PASS_LISTENER = r"""
import array, socket
control = socket.socket(socket.AF_UNIX)
control.connect("/run/vf/control.sock")
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
control.sendmsg([b"listener"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [listener.fileno()]))])
"""


class DockerBackend(ContainerBackend):
    supports_image: ClassVar[bool] = True
    supports_user: ClassVar[bool] = True

    def __init__(
        self,
        config: DockerConfig | PrimeConfig | ModalConfig,
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
        self._relay: CallbackRelay | None = None
        self._image_env: dict[str, str] = {}
        self._stopped = False
        # Helper containers carry the run label, including for borrowed containers.
        self._label_args = ["--label", f"verifiers.run={run_scope()}"]

    @classmethod
    @contextlib.asynccontextmanager
    async def attach(
        cls,
        config: DockerConfig | PrimeConfig | ModalConfig,
        container: str,
        *,
        service_url: str | None = None,
        host_backend: RuntimeBackend | None = None,
    ) -> AsyncIterator[DockerBackend]:
        """Borrow an existing container; its caller owns creation and removal."""
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
                backend._relay = CallbackRelay()
                if sys.platform == "linux":
                    await backend._relay.start(
                        listener=await backend._container_listener()
                    )
                else:
                    await backend._relay.start("127.0.0.1")
            yield backend
        finally:
            await backend.stop()

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
        # Engine names are global; cleanup must never target a caller's existing container.
        self._container = f"vf-{uuid.uuid4().hex}"
        options = ["--network", "bridge", "--publish", f"127.0.0.1::{SERVICE_PORT}"]
        if self.config.cpu is not None:
            options += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            options += ["--memory", f"{self.config.memory}g"]
        gpu_type, gpu_count = parse_gpu(self.config.gpu)
        if gpu_count:
            options += ["--gpus", str(gpu_count)]
        if sys.platform != "linux":
            options += ["--add-host", f"{_PROXY_HOST}:host-gateway"]
        env_args = [
            arg
            for key, value in self.env.items()
            for arg in ("--env", f"{key}={value}")
        ]
        run = await cli(
            "docker",
            "run",
            *self._label_args,
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
        published = await cli("docker", "port", self._container, f"{SERVICE_PORT}/tcp")
        host_port = (published.stdout.split() or [""])[0].rpartition(":")[2]
        if published.exit_code != 0 or not host_port.isdigit():
            detail = (published.stderr or published.stdout).strip()
            raise SandboxError(f"docker did not publish the service port: {detail}")
        self._service_url = f"http://127.0.0.1:{host_port}"
        # One listener handles every callback, without reserving the host service's
        # port inside the container.
        self._relay = CallbackRelay()
        if sys.platform == "linux":
            await self._relay.start(listener=await self._container_listener())
        else:
            await self._relay.start("127.0.0.1")
        logger.info(
            "docker: started container %s (image=%s)",
            self._container,
            self.config.image,
        )

    def host_url(self, url: str) -> str:
        if self._host_backend is not None:
            return self._host_backend.host_url(url)
        parts = urlsplit(url)
        if not is_loopback_host(parts.hostname or ""):
            return url
        assert self._relay is not None
        host_backend = "127.0.0.1" if sys.platform == "linux" else _PROXY_HOST
        return self._relay.callback_url(url, host_backend)

    async def expose(self, port: int) -> str:
        if self._host_backend is not None:
            return await self._host_backend.expose(port)
        if port != SERVICE_PORT or self._service_url is None:
            raise SandboxError(f"docker publishes only port {SERVICE_PORT}, not {port}")
        return self._service_url

    async def _container_listener(self) -> socket.socket:
        """Create the proxy listener inside the container netns, serviced here."""
        # Reuse the task image offline; only images without Python need the shared helper.
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
                *self._label_args,
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
                    f"docker callback listener failed: {helper.stderr.strip()}"
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
        if self._host_backend is not None:
            await self._host_backend.prepare_execution(routes)

    async def teardown(self) -> None:
        if self._relay is not None:
            await self._relay.stop()
        await super().teardown()

    def _exec(
        self, env: dict[str, str], *, stdin: bool = False, user: str | None = None
    ) -> list[str]:
        assert self._container is not None
        # Callback URLs must bypass any proxy the image or task configures.
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
            "docker",
            "exec",
            *(("-i",) if stdin else ()),
            *(arg for key, value in env.items() for arg in ("--env", f"{key}={value}")),
            *(("--user", user) if user is not None else ()),
            "--workdir",
            self.config.workdir,
            self._container,
        ]

    async def _run_background(
        self, runtime: Runtime, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # The engine owns the background server as a detached exec process.
        command = self._exec(runtime.process_env(env), user=runtime.user)
        command.insert(2, "--detach")
        script = f"exec {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null"
        result = await self._run_host(*command, "sh", "-c", script)
        if result.exit_code != 0:
            raise SandboxError(
                f"docker background process failed: {result.stderr.strip()}"
            )

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
