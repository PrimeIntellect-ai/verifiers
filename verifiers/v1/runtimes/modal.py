"""Remote Modal sandbox runtime: run the program in a Modal sandbox, reached via a tunnel.

`expose` (sandbox port -> public internet) uses Modal's own forwarding — a port named via
`encrypted_ports` at `Sandbox.create`, read back from `sandbox.tunnels()` — so a host-side
harness/framework can reach a tool/user server hosted in the sandbox. The reverse direction (a
program in the sandbox reaching a host service) is the shared host-side `host_endpoint` tunnel,
not the runtime's concern.
"""

import asyncio
import contextlib
import logging
import shlex
from pathlib import PurePosixPath
from typing import ClassVar, Literal

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import SERVICE_PORT, ProgramResult, Runtime
from verifiers.v1.runtimes.limiters import creation_limiter

logger = logging.getLogger(__name__)


# Shared Modal app every rollout's sandbox attaches to (created on first lookup).
_APP_NAME = "verifiers-v1"


class ModalConfig(BaseConfig):
    type: Literal["modal"] = "modal"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    network_access: bool = True
    region: str | None = None
    """Region to provision in (None = provider-chosen)."""
    # TaskResources, in Modal's native units (also settable per-task via Task.resources, with
    # precedence cli/toml > task > this default).
    cpu: float = 1.0
    """CPU cores."""
    memory: float = 2.0
    """Memory in GB."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "A100:2"."""
    disk: float = 5.0
    """Disk in GB. Modal sandboxes have no disk knob, so this is accepted (so a task can
    declare it without a warning) but not enforced."""
    creates_per_sec: float | None = 40.0
    """Pace sandbox creation to this many per second, enforced host-wide across every
    env-server worker process (None/<= 0 disables it)."""


class ModalRuntime(Runtime):
    """Runs the program in a Modal sandbox; the server is reached via a tunnel."""

    is_local: ClassVar[bool] = False

    def __init__(self, config: ModalConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self._sandbox = None
        self._sandbox_id: str | None = None

    @property
    def descriptor(self) -> str | None:
        return self._sandbox_id

    @property
    def published_port(self) -> int | None:
        return SERVICE_PORT

    async def start(self) -> None:
        try:
            import modal
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "ModalRuntime requires the Modal SDK; install `verifiers[modal]`."
            ) from e

        try:
            app = await modal.App.lookup.aio(_APP_NAME, create_if_missing=True)
            async with (
                creation_limiter(self.config.creates_per_sec, "modal-sandbox")
                or contextlib.nullcontext()
            ):
                self._sandbox = await modal.Sandbox.create.aio(
                    "sleep",
                    "infinity",  # keep-alive entrypoint; the harness runs via `exec`
                    app=app,
                    name=self.name,
                    # Clear the image's ENTRYPOINT so `sleep infinity` runs as the command
                    # rather than as args to it — otherwise an image with its own entrypoint
                    # (e.g. SWE task images) never starts the keep-alive and the sandbox dies.
                    image=modal.Image.from_registry(self.config.image).entrypoint([]),
                    workdir=self.config.workdir,
                    cpu=self.config.cpu,
                    memory=int(self.config.memory * 1024),  # Modal memory is MB
                    gpu=self.config.gpu,
                    region=self.config.region,
                    block_network=not self.config.network_access,
                    timeout=24 * 60 * 60,  # Maximum lifetime of any sandbox.
                    encrypted_ports=[SERVICE_PORT],
                )
            self._sandbox_id = self._sandbox.object_id
            logger.info(
                "modal: sandbox %s up (image=%s)", self._sandbox_id, self.config.image
            )
            await self._sandbox.filesystem.make_directory.aio(self.config.workdir)
        except (
            Exception
        ) as e:  # provisioning failure is one rollout's problem, not the eval's
            raise SandboxError(f"modal sandbox provisioning failed: {e}") from e

    async def expose(self, port: int) -> str | None:
        # Publish a server hosted IN the sandbox: Modal forwards `port` (named via
        # `encrypted_ports` at creation) to a public URL, read back from `sandbox.tunnels()`.
        # Only the pre-declared SERVICE_PORT is forwarded; any other port has no tunnel.
        if self._sandbox is None:
            return None
        try:
            tunnels = await self._sandbox.tunnels.aio()
        except Exception as e:
            raise SandboxError(f"modal tunnels unavailable (port {port}): {e}") from e
        tunnel = tunnels.get(port)
        return str(tunnel.url).rstrip("/") if tunnel else None

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        try:
            proc = await self._sandbox.exec.aio(
                *argv, workdir=self.config.workdir, env=env
            )
            # Drain both pipes concurrently so a large stderr can't deadlock stdout.
            stdout, stderr = await asyncio.gather(
                proc.stdout.read.aio(), proc.stderr.read.aio()
            )
            await proc.wait.aio()
        except (
            Exception
        ) as e:  # a sandbox/API failure is one rollout's problem, not the eval's
            raise SandboxError(f"modal exec failed: {e}") from e
        return ProgramResult(
            exit_code=proc.returncode or 0,
            stdout=stdout or "",
            stderr=stderr or "",
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # `&` backgrounds inside the sandbox; the job returns immediately, the process
        # lives until the sandbox is terminated in stop().
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise SandboxError(
                f"modal background launch failed: {result.stderr.strip()}"
            )

    def _abs(self, path: str) -> str:
        # Modal's filesystem API only accepts absolute remote paths; resolve a relative
        # one against the workdir (the cwd run/run_background use).
        if path.startswith("/"):
            return path
        return f"{self.config.workdir.rstrip('/')}/{path}"

    async def read(self, path: str) -> bytes:
        try:
            return await self._sandbox.filesystem.read_bytes.aio(self._abs(path))
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}") from e

    async def write(self, path: str, data: bytes) -> None:
        # Create the parent first (Modal's write does not mkdir).
        target = self._abs(path)
        try:
            await self._sandbox.filesystem.make_directory.aio(
                str(PurePosixPath(target).parent)
            )
            await self._sandbox.filesystem.write_bytes.aio(data, target)
        except Exception as e:
            raise SandboxError(f"write {path!r}: {e}") from e

    def cleanup(self) -> None:
        # Synchronous atexit backstop (the async API can't run once the loop is gone): terminate
        # the sandbox via Modal's sync API so the costly resource isn't left to its max-lifetime.
        # Idempotent — the async `stop` handles the normal path, and a second terminate is a no-op.
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:  # `_sandbox_id` kept so descriptor survives teardown
            with contextlib.suppress(Exception):
                sandbox.terminate()

    async def stop(self) -> None:
        # Best-effort, idempotent teardown on the normal path: terminate the sandbox (the costly
        # resource) via the async API. Runs from the rollout's `finally`, so it fires on success,
        # error, and cancellation; `_sandbox` is nulled as the idempotency guard (atexit no-ops).
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is None:
            return
        try:
            await sandbox.terminate.aio()
        except Exception as e:
            logger.warning(
                "modal: failed to terminate sandbox %s: %s", self._sandbox_id, e
            )
