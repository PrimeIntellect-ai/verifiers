"""Remote Daytona sandbox runtime.

`expose` (sandbox port -> public URL) uses Daytona's preview links, which only skip the
access token on a `public` sandbox. The reverse direction (a program in the sandbox
reaching a host service) is the shared host-side `Tunnel` (interception.tunnel), not the
runtime's concern. Daytona streams process output as text, which cannot carry the
length-prefixed binary ACP packets, so live processes are not supported.
"""

import asyncio
import contextlib
import logging
import math
import shlex
import uuid
from typing import ClassVar, Literal

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    parse_gpu,
)
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)


class DaytonaConfig(BaseConfig):
    type: Literal["daytona"] = "daytona"
    image: str = "python:3.11-slim"
    """Docker image to run; Daytona pulls it on first use."""
    workdir: str = "/app"
    network_access: bool = True
    public: bool = False
    """Make the sandbox's preview links reachable without an access token; required for
    `expose` (a tool server hosted in the sandbox)."""
    region: str | None = None
    """Target region to provision in (None = the organization's default)."""
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float = 1.0
    """CPU cores (Daytona allocates whole cores)."""
    memory: float = 2.0
    """Memory in GB (Daytona allocates whole GiB)."""
    gpu: str | None = None
    """GPU spec, e.g. "H100" or "H100:2" (a bare count = provider-chosen type)."""
    disk: float = 5.0
    """Disk in GB (Daytona allocates whole GiB)."""
    idle_timeout: float | None = 3600
    """Seconds without SDK interaction before the sandbox stops and self-deletes (None
    disables)."""


class DaytonaRuntimeInfo(DaytonaConfig, BaseRuntimeInfo):
    pass


class DaytonaRuntime(Runtime):
    is_local: ClassVar[bool] = False

    def __init__(self, config: DaytonaConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = DaytonaRuntimeInfo(**config.model_dump())
        self._daytona = None
        self._sandbox = None

    @property
    def published_port(self) -> int | None:
        return SERVICE_PORT

    async def start(self) -> None:
        try:
            from daytona import AsyncDaytona
            from daytona import DaytonaConfig as ClientConfig
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "DaytonaRuntime requires the Daytona SDK; install `verifiers[daytona]`."
            ) from e

        self._daytona = AsyncDaytona(ClientConfig(target=self.config.region))
        try:
            await run_shielded(self._create_sandbox())
            self.info.id = self._sandbox.id
            logger.info(
                "daytona: sandbox %s up (image=%s)", self.info.id, self.config.image
            )
            await self._sandbox.fs.create_folder(self.config.workdir, "755")
        except (
            Exception
        ) as e:  # provisioning failure is one rollout's problem, not the eval's
            raise SandboxError(f"daytona sandbox provisioning failed: {e}") from e

    async def _create_sandbox(self) -> None:
        """Create the sandbox and take ownership of the handle as one shielded step, so a
        cancellation landing mid-create can't leave a booted sandbox `teardown` never
        learns of."""
        from daytona import CreateSandboxFromImageParams, Resources

        gpu_type, gpu_count = parse_gpu(self.config.gpu)
        idle_minutes = (
            max(1, math.ceil(self.config.idle_timeout / 60))
            if self.config.idle_timeout is not None
            else 0
        )
        self._sandbox = await self._daytona.create(
            CreateSandboxFromImageParams(
                image=self.config.image,
                name=self.name,
                env_vars=self.env,
                public=self.config.public,
                network_block_all=not self.config.network_access,
                resources=Resources(
                    cpu=math.ceil(self.config.cpu),
                    memory=math.ceil(self.config.memory),
                    disk=math.ceil(self.config.disk),
                    gpu=gpu_count or None,
                    gpu_type=gpu_type,
                ),
                auto_stop_interval=idle_minutes,
                # A stopped sandbox is deleted right away: nothing here outlives its rollout.
                ephemeral=True,
            ),
            timeout=0,  # image pulls can take minutes; rollout cancellation bounds the wait
        )

    async def expose(self, port: int) -> str | None:
        if not self.config.public:
            raise SandboxError(
                "daytona preview links need an access token unless the sandbox is public; "
                "set `runtime.public=true` to host a tool server in a daytona sandbox"
            )
        try:
            link = await self._sandbox.get_preview_link(port)
        except Exception as e:
            raise SandboxError(f"daytona preview link failed (port {port}): {e}") from e
        logger.info("daytona: exposed sandbox port %d at %s", port, link.url)
        return link.url.rstrip("/")

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        # A session command keeps stdout and stderr apart (`exec` merges them) and, run
        # asynchronously and polled, isn't cut short by the SDK's HTTP timeout, so
        # rollout cancellation owns the execution timeout. The subshell keeps an `exit`
        # in argv from ending the session's shell instead of the command.
        from daytona import SessionExecuteRequest

        assignments = [f"{k}={v}" for k, v in self.process_env(env).items()]
        command = (
            f"(cd {shlex.quote(self.config.workdir)} && exec env "
            f"{shlex.join([*assignments, *argv])})"
        )
        session = f"vf-{uuid.uuid4().hex}"
        try:
            await self._sandbox.process.create_session(session)
            try:
                started = await self._sandbox.process.execute_session_command(
                    session, SessionExecuteRequest(command=command, run_async=True)
                )
                delay = 0.1
                while True:
                    status = await self._sandbox.process.get_session_command(
                        session, started.cmd_id
                    )
                    if status.exit_code is not None:
                        break
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 3)
                logs = await self._sandbox.process.get_session_command_logs(
                    session, started.cmd_id
                )
            finally:
                with contextlib.suppress(Exception):
                    await self._sandbox.process.delete_session(session)
        except (
            Exception
        ) as e:  # a sandbox/API failure is one rollout's problem, not the eval's
            raise SandboxError(f"daytona exec failed: {e}") from e
        return ProgramResult(
            exit_code=status.exit_code,
            stdout=logs.stdout or "",
            stderr=logs.stderr or "",
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # Launched through `exec` rather than a session: deleting a session kills the
        # processes it started, while an `exec` child backgrounded with `&` outlives the
        # call and runs until the sandbox is deleted in stop().
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 &"
        try:
            result = await self._sandbox.process.exec(
                inner, cwd=self.config.workdir, env=self.process_env(env)
            )
        except Exception as e:
            raise SandboxError(f"daytona background launch failed: {e}") from e
        if result.exit_code != 0:
            raise SandboxError(
                f"daytona background launch failed: {result.result.strip()}"
            )

    def _abs(self, path: str) -> str:
        # The filesystem API resolves relative paths against the sandbox's own working
        # directory, not ours.
        if path.startswith("/"):
            return path
        return f"{self.config.workdir.rstrip('/')}/{path}"

    async def _read(self, path: str, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None:
            return await super()._read(path, max_bytes)
        try:
            return await self._sandbox.fs.download_file(self._abs(path))
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}") from e

    async def write(self, path: str, data: bytes) -> None:
        try:
            await self._sandbox.fs.upload_file(data, self._abs(path))
        except Exception as e:
            raise SandboxError(f"write {path!r}: {e}") from e

    def cleanup(self) -> None:
        # Synchronous atexit backstop (the async client can't run once the loop is gone):
        # delete the sandbox via the sync client. Idempotent — the async `stop` deletes it
        # on the normal path, a second delete fails harmlessly.
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:  # keep info.id available after teardown
            from daytona import Daytona

            with contextlib.suppress(Exception):
                client = Daytona()
                client.delete(client.get(sandbox.id))

    async def teardown(self) -> None:
        # Best-effort, idempotent teardown: delete the sandbox (the costly resource). Runs
        # via `stop`, shielded from cancellation. `_sandbox` — the atexit backstop's key —
        # is consumed only after the delete attempt, so a loop death mid-await still
        # leaves the backstop something to delete.
        sandbox = self._sandbox
        if sandbox is None:
            return
        try:
            await self._daytona.delete(sandbox)
        except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
            logger.warning("daytona: failed to delete sandbox %s: %s", self.info.id, e)
        finally:
            self._sandbox = None
            with contextlib.suppress(Exception):
                await self._daytona.close()
