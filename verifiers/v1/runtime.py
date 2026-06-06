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

TrajectoryVisibility = Literal["visible", "hidden"]

_ENSURE_UV = "command -v uv >/dev/null 2>&1 || pip install -q uv"
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


class LocalRuntimeConfig(Config):
    type: Literal["local"] = "local"


class DockerRuntimeConfig(Config):
    type: Literal["docker"] = "docker"
    image: str = "python:3.11-slim"
    workdir: str = "/app"


class PrimeRuntimeConfig(Config):
    type: Literal["prime"] = "prime"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    network_access: bool = True


RuntimeConfig = Annotated[
    LocalRuntimeConfig | DockerRuntimeConfig | PrimeRuntimeConfig,
    Field(discriminator="type"),
]
RuntimeConfigValue = LocalRuntimeConfig | DockerRuntimeConfig | PrimeRuntimeConfig
RUNTIME_CONFIG_TYPES = (
    LocalRuntimeConfig,
    DockerRuntimeConfig,
    PrimeRuntimeConfig,
)


class RuntimeSession(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def expose(self, port: int) -> str: ...

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

    async def __aenter__(self) -> "RuntimeSession":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()


class RuntimeProvider(ABC):
    @abstractmethod
    def session(self) -> RuntimeSession: ...


class LocalRuntimeProvider(RuntimeProvider):
    def __init__(self, config: LocalRuntimeConfig | None = None) -> None:
        self.config = config or LocalRuntimeConfig()

    def session(self) -> RuntimeSession:
        return LocalRuntimeSession(self.config)


class DockerRuntimeProvider(RuntimeProvider):
    def __init__(self, config: DockerRuntimeConfig) -> None:
        self.config = config

    def session(self) -> RuntimeSession:
        return DockerRuntimeSession(self.config)


class PrimeRuntimeProvider(RuntimeProvider):
    def __init__(self, config: PrimeRuntimeConfig) -> None:
        self.config = config

    def session(self) -> RuntimeSession:
        return PrimeRuntimeSession(self.config)


def make_runtime_provider(config: RuntimeConfigValue) -> RuntimeProvider:
    if isinstance(config, PrimeRuntimeConfig):
        return PrimeRuntimeProvider(config)
    if isinstance(config, DockerRuntimeConfig):
        return DockerRuntimeProvider(config)
    return LocalRuntimeProvider(config)


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
    return resolved or LocalRuntimeConfig()


class LocalRuntimeSession(RuntimeSession):
    def __init__(self, config: LocalRuntimeConfig) -> None:
        self.config = config
        self.workdir: Path | None = None

    async def start(self) -> None:
        self.workdir = Path(tempfile.mkdtemp(prefix="vf-v1-", dir="/tmp"))

    async def stop(self) -> None:
        if self.workdir is not None:
            await asyncio.to_thread(shutil.rmtree, self.workdir, True)
            self.workdir = None

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
        if not command:
            raise ValueError("RuntimeSession.run requires a command.")
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

    def _path(self, path: str | None) -> Path:
        if self.workdir is None:
            raise RuntimeError("Local runtime has not started.")
        if path is None:
            return self.workdir
        value = Path(path)
        return value if value.is_absolute() else self.workdir / value


class DockerRuntimeSession(RuntimeSession):
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
        result = await docker(
            "run",
            "--detach",
            "--network",
            "host",
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
        result = await self.run(["cat", path])
        if result.returncode != 0:
            raise RuntimeError(f"read {path!r}: {result.stderr.strip()}")
        return result.stdout.encode()

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

    def _container(self) -> str:
        if self.container is None:
            raise RuntimeError("Docker runtime has not started.")
        return self.container


class PrimeRuntimeSession(RuntimeSession):
    def __init__(self, config: PrimeRuntimeConfig) -> None:
        self.config = config
        self.client = None
        self.sandbox_id: str | None = None
        self.tunnels: list[RuntimeTunnel] = []

    async def start(self) -> None:
        from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

        self.client = AsyncSandboxClient()
        sandbox = await self.client.create(
            CreateSandboxRequest(
                name="vf-v1-runtime",
                docker_image=self.config.image,
                cpu_cores=self.config.cpu_cores,
                memory_gb=self.config.memory_gb,
                network_access=self.config.network_access,
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
        encoded = base64.b64encode(data).decode()
        parent = shlex.quote(str(PurePosixPath(path).parent))
        result = await self.run(
            [
                "sh",
                "-c",
                f"mkdir -p {parent} && printf %s {encoded} | base64 -d > {shlex.quote(path)}",
            ]
        )
        if result.returncode != 0:
            raise RuntimeError(f"write {path!r}: {result.stderr.strip()}")


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
