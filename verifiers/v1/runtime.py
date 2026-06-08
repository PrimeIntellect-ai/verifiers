from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import shlex
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol

from pydantic import Field

from .config import Config
from .types import JsonData

_INSTALL_CURL = (
    "{ command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; } "
    "|| { apt-get update -qq && apt-get install -y -qq curl ca-certificates; } "
    "|| apk add --no-cache curl ca-certificates"
)
_DOWNLOAD_UV = (
    "{ command -v curl >/dev/null 2>&1 && "
    "curl -LsSf https://astral.sh/uv/install.sh | sh; } "
    "|| { command -v wget >/dev/null 2>&1 && "
    "wget -qO- https://astral.sh/uv/install.sh | sh; }"
)
_ENSURE_UV = (
    'export PATH="$HOME/.local/bin:$PATH" UV_INSTALL_DIR="$HOME/.local/bin"; '
    "command -v uv >/dev/null 2>&1 "
    "|| pip install -q uv 2>/dev/null "
    f"|| {{ {_INSTALL_CURL}; {_DOWNLOAD_UV}; }}"
)
_SUBPROCESS_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
)


class RuntimeTunnel(Protocol):
    def sync_stop(self) -> None: ...


class CommandResult(Config):
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def exit_code(self) -> int:
        return self.returncode


class SubprocessRuntimeConfig(Config):
    type: Literal["subprocess"] = "subprocess"


class DockerRuntimeConfig(Config):
    type: Literal["docker"] = "docker"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    cpu_cores: float | None = None
    memory_gb: float | None = None
    gpu_count: int | None = None
    disk_gb: float | None = None


class PrimeRuntimeConfig(Config):
    type: Literal["prime"] = "prime"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    network_access: bool = True
    vm: bool = False
    guaranteed: bool = False
    region: str | None = None
    gpu_type: str | None = None
    timeout_minutes: int | Literal["auto"] = 360
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    gpu_count: int = 0
    disk_gb: float = 5.0


class ModalRuntimeConfig(Config):
    type: Literal["modal"] = "modal"
    image: str = "python:3.11-slim"


class DaytonaRuntimeConfig(Config):
    type: Literal["daytona"] = "daytona"
    image: str = "python:3.11-slim"


RuntimeConfig = Annotated[
    SubprocessRuntimeConfig
    | DockerRuntimeConfig
    | PrimeRuntimeConfig
    | ModalRuntimeConfig
    | DaytonaRuntimeConfig,
    Field(discriminator="type"),
]
RuntimeConfigValue = (
    SubprocessRuntimeConfig
    | DockerRuntimeConfig
    | PrimeRuntimeConfig
    | ModalRuntimeConfig
    | DaytonaRuntimeConfig
)
RUNTIME_CONFIG_TYPES = (
    SubprocessRuntimeConfig,
    DockerRuntimeConfig,
    PrimeRuntimeConfig,
    ModalRuntimeConfig,
    DaytonaRuntimeConfig,
)


class Runtime(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def expose(self, port: int) -> str: ...

    async def public_url(self, port: int) -> str | None:
        _ = port
        return None

    @abstractmethod
    async def run(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult: ...

    @abstractmethod
    async def read(self, path: str) -> bytes: ...

    @abstractmethod
    async def write(self, path: str, data: bytes) -> None: ...

    async def run_background(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        log: str | None = None,
    ) -> None:
        _ = command, cwd, env, log
        raise NotImplementedError(
            f"{type(self).__name__} does not support background commands."
        )

    async def run_uv_script(
        self,
        script: str | bytes,
        *,
        input: JsonData | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        token = uuid.uuid4().hex
        script_path = f"_vf_{token}.py"
        await self.write(
            script_path, script.encode() if isinstance(script, str) else script
        )
        command = f"{_ENSURE_UV}; uv run {shlex.quote(script_path)}"
        if input is not None:
            input_path = f"_vf_{token}.json"
            await self.write(input_path, json.dumps(input).encode())
            command = f"{command} {shlex.quote(input_path)}"
        return await self.run(["sh", "-c", command], env=env, timeout=timeout)

    async def __aenter__(self) -> "Runtime":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()


class RuntimeProvider(ABC):
    @abstractmethod
    def create_runtime(self) -> Runtime: ...


class SubprocessRuntimeProvider(RuntimeProvider):
    def __init__(self, config: SubprocessRuntimeConfig | None = None) -> None:
        self.config = config or SubprocessRuntimeConfig()

    def create_runtime(self) -> Runtime:
        return SubprocessRuntime(self.config)


class DockerRuntimeProvider(RuntimeProvider):
    def __init__(self, config: DockerRuntimeConfig) -> None:
        self.config = config

    def create_runtime(self) -> Runtime:
        return DockerRuntime(self.config)


class PrimeRuntimeProvider(RuntimeProvider):
    def __init__(self, config: PrimeRuntimeConfig) -> None:
        self.config = config

    def create_runtime(self) -> Runtime:
        return PrimeRuntime(self.config)


class ModalRuntimeProvider(RuntimeProvider):
    def __init__(self, config: ModalRuntimeConfig) -> None:
        self.config = config

    def create_runtime(self) -> Runtime:
        return ModalRuntime(self.config)


class DaytonaRuntimeProvider(RuntimeProvider):
    def __init__(self, config: DaytonaRuntimeConfig) -> None:
        self.config = config

    def create_runtime(self) -> Runtime:
        return DaytonaRuntime(self.config)


def make_runtime_provider(config: RuntimeConfigValue) -> RuntimeProvider:
    if isinstance(config, PrimeRuntimeConfig):
        return PrimeRuntimeProvider(config)
    if isinstance(config, DockerRuntimeConfig):
        return DockerRuntimeProvider(config)
    if isinstance(config, ModalRuntimeConfig):
        return ModalRuntimeProvider(config)
    if isinstance(config, DaytonaRuntimeConfig):
        return DaytonaRuntimeProvider(config)
    return SubprocessRuntimeProvider(config)


def resolve_runtime_config(
    *configs: RuntimeConfigValue | None,
) -> RuntimeConfigValue:
    resolved: RuntimeConfigValue | None = None
    for config in configs:
        if config is None:
            continue
        if resolved is None:
            resolved = config.model_copy()
            continue
        if type(config) is not type(resolved):
            raise ValueError(
                "Runtime config resolution requires a single provider type; got "
                f"{resolved.type!r} and {config.type!r}."
            )
        updates = {
            field: getattr(config, field)
            for field in config.model_fields_set
            if field != "type"
        }
        if updates:
            resolved = resolved.model_copy(update=updates)
    return resolved or SubprocessRuntimeConfig()


class SubprocessRuntime(Runtime):
    def __init__(self, config: SubprocessRuntimeConfig) -> None:
        self.config = config
        self.workdir: Path | None = None
        self.processes: list[asyncio.subprocess.Process] = []

    async def start(self) -> None:
        self.workdir = Path(tempfile.mkdtemp(prefix="vf-v1-", dir="/tmp"))

    async def stop(self) -> None:
        for process in self.processes:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
        for process in self.processes:
            if process.returncode is None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), timeout=5)
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
        self.processes = []
        if self.workdir is not None:
            await asyncio.to_thread(shutil.rmtree, self.workdir, True)
            self.workdir = None

    async def expose(self, port: int) -> str:
        return f"http://127.0.0.1:{port}"

    async def public_url(self, port: int) -> str | None:
        return f"http://127.0.0.1:{port}"

    async def run(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if not command:
            raise ValueError("Runtime.run requires a command.")
        full_env = {
            name: os.environ[name] for name in _SUBPROCESS_ENV if name in os.environ
        }
        full_env.update(env or {})
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self._path(cwd) if cwd is not None else self.workdir,
            env=full_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise
        return CommandResult(
            returncode=int(process.returncode or 0),
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    async def read(self, path: str) -> bytes:
        return await asyncio.to_thread(self._path(path).read_bytes)

    async def write(self, path: str, data: bytes) -> None:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def run_background(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        log: str | None = None,
    ) -> None:
        if not command:
            raise ValueError("Runtime.run_background requires a command.")
        full_env = {
            name: os.environ[name] for name in _SUBPROCESS_ENV if name in os.environ
        }
        full_env.update(env or {})
        stdout = stderr = asyncio.subprocess.DEVNULL
        log_file = None
        if log is not None:
            log_path = self._path(log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("ab")
            stdout = stderr = log_file
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self._path(cwd) if cwd is not None else self.workdir,
                env=full_env,
                stdout=stdout,
                stderr=stderr,
            )
        finally:
            if log_file is not None:
                log_file.close()
        self.processes.append(process)

    def _path(self, path: str | None) -> Path:
        if self.workdir is None:
            raise RuntimeError("Subprocess runtime has not started.")
        if path is None:
            return self.workdir
        value = Path(path)
        return value if value.is_absolute() else self.workdir / value


class DockerRuntime(Runtime):
    def __init__(self, config: DockerRuntimeConfig) -> None:
        self.config = config
        self.container: str | None = None

    async def start(self) -> None:
        try:
            version = await docker("version", "--format", "{{.Server.Version}}")
        except FileNotFoundError as exc:
            raise RuntimeError("docker runtime requires the docker CLI.") from exc
        if version.returncode != 0:
            detail = (version.stderr or version.stdout).strip()
            raise RuntimeError(f"Docker daemon is not reachable: {detail}")
        self.container = f"vf-v1-{uuid.uuid4().hex[:12]}"
        limits: list[str] = []
        if self.config.cpu_cores is not None:
            limits += ["--cpus", str(self.config.cpu_cores)]
        if self.config.memory_gb is not None:
            limits += ["--memory", f"{self.config.memory_gb}g"]
        if self.config.gpu_count:
            limits += ["--gpus", str(self.config.gpu_count)]
        result = await docker(
            "run",
            "--detach",
            "--network",
            "host",
            *limits,
            "--workdir",
            self.config.workdir,
            "--name",
            self.container,
            self.config.image,
            "sleep",
            "infinity",
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")

    async def stop(self) -> None:
        if self.container is None:
            return
        container, self.container = self.container, None
        with contextlib.suppress(Exception):
            await docker("rm", "--force", container)

    async def expose(self, port: int) -> str:
        return f"http://127.0.0.1:{port}"

    async def run(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        container = self._container()
        env_args = [
            arg
            for key, value in (env or {}).items()
            for arg in ("--env", f"{key}={value}")
        ]
        return await docker(
            "exec",
            *env_args,
            "--workdir",
            cwd or self.config.workdir,
            container,
            *command,
            timeout=timeout,
        )

    async def read(self, path: str) -> bytes:
        result = await self.run(["sh", "-c", f"base64 < {shlex.quote(path)}"])
        if result.returncode != 0:
            raise RuntimeError(f"read {path!r}: {result.stderr.strip()}")
        return base64.b64decode(result.stdout)

    async def write(self, path: str, data: bytes) -> None:
        container = self._container()
        parent = shlex.quote(str(PurePosixPath(path).parent))
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "--workdir",
            self.config.workdir,
            container,
            "sh",
            "-c",
            f"mkdir -p {parent} && cat > {shlex.quote(path)}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate(input=data)
        if process.returncode != 0:
            raise RuntimeError(
                f"write {path!r}: {stderr.decode(errors='replace').strip()}"
            )

    async def run_background(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        log: str | None = None,
    ) -> None:
        if not command:
            raise ValueError("Runtime.run_background requires a command.")
        container = self._container()
        env_args = [
            arg
            for key, value in (env or {}).items()
            for arg in ("--env", f"{key}={value}")
        ]
        log_path = shlex.quote(log or "/tmp/vf-background.log")
        result = await docker(
            "exec",
            "--detach",
            *env_args,
            "--workdir",
            cwd or self.config.workdir,
            container,
            "sh",
            "-c",
            f"exec {shlex.join(command)} >> {log_path} 2>&1",
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker background command failed: {result.stderr}")

    def _container(self) -> str:
        if self.container is None:
            raise RuntimeError("Docker runtime has not started.")
        return self.container


class PrimeRuntime(Runtime):
    def __init__(self, config: PrimeRuntimeConfig) -> None:
        self.config = config
        self.client = None
        self.sandbox_id: str | None = None
        self.tunnels: list[RuntimeTunnel] = []

    async def start(self) -> None:
        from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

        self.client = AsyncSandboxClient()
        timeout = (
            24 * 60
            if self.config.timeout_minutes == "auto"
            else self.config.timeout_minutes
        )
        sandbox = await self.client.create(
            CreateSandboxRequest(
                name="vf-v1-runtime",
                docker_image=self.config.image,
                cpu_cores=self.config.cpu_cores,
                memory_gb=self.config.memory_gb,
                disk_size_gb=self.config.disk_gb,
                gpu_count=self.config.gpu_count,
                timeout_minutes=timeout,
                network_access=self.config.network_access,
                vm=self.config.vm,
                guaranteed=self.config.guaranteed,
                gpu_type=self.config.gpu_type,
                region=self.config.region,
            )
        )
        self.sandbox_id = sandbox.id
        await self.client.wait_for_creation(self.sandbox_id)
        await self.client.run_background_job(
            self.sandbox_id,
            f"mkdir -p {shlex.quote(self.config.workdir)}",
        )

    async def stop(self) -> None:
        for tunnel in self.tunnels:
            with contextlib.suppress(Exception):
                tunnel.sync_stop()
        self.tunnels = []
        client, sandbox_id = self.client, self.sandbox_id
        self.client, self.sandbox_id = None, None
        if client is None:
            return
        if sandbox_id is not None:
            with contextlib.suppress(Exception):
                await client.delete(sandbox_id)
        with contextlib.suppress(Exception):
            await client.aclose()

    async def expose(self, port: int) -> str:
        from prime_tunnel import Tunnel

        tunnel = Tunnel(local_port=port)
        url = str(await tunnel.start()).rstrip("/")
        self.tunnels.append(tunnel)
        return url

    async def public_url(self, port: int) -> str | None:
        if self.client is None or self.sandbox_id is None:
            raise RuntimeError("Prime runtime has not started.")
        try:
            exposed = await self.client.expose(self.sandbox_id, port)
        except Exception as exc:
            raise RuntimeError(
                "Prime port exposure failed. Runtime-placed servers on Prime "
                "require sandbox port exposure; use a supported region/port or "
                "place the server in a host/subprocess runtime."
            ) from exc
        return str(exposed.url).rstrip("/")

    async def run(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        _ = timeout
        if self.client is None or self.sandbox_id is None:
            raise RuntimeError("Prime runtime has not started.")
        result = await self.client.run_background_job(
            self.sandbox_id,
            shlex.join(command),
            working_dir=cwd or self.config.workdir,
            env=env or {},
        )
        return CommandResult(
            returncode=int(result.exit_code or 0),
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def read(self, path: str) -> bytes:
        result = await self.run(["sh", "-c", f"base64 {shlex.quote(path)}"])
        if result.returncode != 0:
            raise RuntimeError(f"read {path!r}: {result.stderr.strip()}")
        return base64.b64decode(result.stdout)

    async def write(self, path: str, data: bytes) -> None:
        if self.client is None or self.sandbox_id is None:
            raise RuntimeError("Prime runtime has not started.")
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        parent = shlex.quote(str(PurePosixPath(target).parent))
        result = await self.run(["sh", "-c", f"mkdir -p {parent}"])
        if result.returncode != 0:
            raise RuntimeError(f"write {path!r}: {result.stderr.strip()}")
        try:
            await self.client.upload_bytes(
                self.sandbox_id,
                target,
                data,
                filename=PurePosixPath(target).name,
            )
        except Exception as exc:
            raise RuntimeError(f"write {path!r}: {exc}") from exc

    async def run_background(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        log: str | None = None,
    ) -> None:
        log_path = shlex.quote(log or "/tmp/vf-background.log")
        result = await self.run(
            [
                "sh",
                "-c",
                f"nohup {shlex.join(command)} >> {log_path} 2>&1 &",
            ],
            cwd=cwd,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"prime background command failed: {result.stderr}")


class ModalRuntime(Runtime):
    def __init__(self, config: ModalRuntimeConfig) -> None:
        self.config = config

    async def start(self) -> None:
        raise NotImplementedError("Modal runtime is not implemented yet.")

    async def stop(self) -> None:
        return None

    async def expose(self, port: int) -> str:
        _ = port
        raise NotImplementedError("Modal runtime is not implemented yet.")

    async def run(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        _ = command, cwd, env, timeout
        raise NotImplementedError("Modal runtime is not implemented yet.")

    async def read(self, path: str) -> bytes:
        _ = path
        raise NotImplementedError("Modal runtime is not implemented yet.")

    async def write(self, path: str, data: bytes) -> None:
        _ = path, data
        raise NotImplementedError("Modal runtime is not implemented yet.")


class DaytonaRuntime(Runtime):
    def __init__(self, config: DaytonaRuntimeConfig) -> None:
        self.config = config

    async def start(self) -> None:
        raise NotImplementedError("Daytona runtime is not implemented yet.")

    async def stop(self) -> None:
        return None

    async def expose(self, port: int) -> str:
        _ = port
        raise NotImplementedError("Daytona runtime is not implemented yet.")

    async def run(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        _ = command, cwd, env, timeout
        raise NotImplementedError("Daytona runtime is not implemented yet.")

    async def read(self, path: str) -> bytes:
        _ = path
        raise NotImplementedError("Daytona runtime is not implemented yet.")

    async def write(self, path: str, data: bytes) -> None:
        _ = path, data
        raise NotImplementedError("Daytona runtime is not implemented yet.")


async def docker(*args: str, timeout: float | None = None) -> CommandResult:
    process = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        raise
    return CommandResult(
        returncode=int(process.returncode or 0),
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )
