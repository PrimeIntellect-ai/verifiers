"""Remote Modal sandbox runtime: run the program in a Modal sandbox, server via tunnel.

The program runs in a remote sandbox and reaches the host interception server over a
tunnel — the host-side `prime_tunnel`, since Modal's own port forwarding goes the other
way (it publishes a sandbox port, not a host one).
"""

import asyncio
import contextlib
import logging
import shlex
from pathlib import PurePosixPath
from typing import Literal

from pydantic_config import BaseConfig

from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes.base import ProgramResult, Runtime
from verifiers.v1.runtimes.limiters import _TUNNEL_LIMITER, creation_limiter

logger = logging.getLogger(__name__)


# Modal's max sandbox lifetime (24h in seconds); "auto" timeout requests it. Sandboxes are
# created with a `sleep infinity` entrypoint so they stay alive for `exec` until terminated.
_MAX_TIMEOUT_SECONDS = 24 * 60 * 60
# Shared Modal app every rollout's sandbox attaches to (created on first lookup).
_APP_NAME = "verifiers-v1"


class ModalConfig(BaseConfig):
    type: Literal["modal"] = "modal"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    network_access: bool = True
    region: str | None = None
    """Region to provision in (None = provider-chosen)."""
    timeout: int | Literal["auto"] = 21600
    """Max sandbox lifetime in seconds (default 6h; or "auto" = the highest Modal supports,
    24h). A hard backstop: the sandbox self-terminates even if local cleanup is skipped."""
    # Resources, in Modal's native units (also settable per-task via Task.resources, with
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
    creates_per_sec: float | None = 5.0
    """Pace sandbox creation to this many per second, enforced host-wide across every
    env-server worker process (None/<= 0 disables it) — Modal's per-workspace limit is 5/s."""


class ModalRuntime(Runtime):
    """Runs the program in a Modal sandbox; the server is reached via a tunnel."""

    def __init__(self, config: ModalConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self._sandbox = None
        self._sandbox_id: str | None = None
        self._tunnels: list = []

    @property
    def descriptor(self) -> str | None:
        return self._sandbox_id

    async def start(self) -> None:
        import modal

        timeout = (
            _MAX_TIMEOUT_SECONDS
            if self.config.timeout == "auto"
            else self.config.timeout
        )
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
                    image=modal.Image.from_registry(self.config.image),
                    workdir=self.config.workdir,
                    cpu=self.config.cpu,
                    memory=int(self.config.memory * 1024),  # Modal memory is MB
                    gpu=self.config.gpu,
                    region=self.config.region,
                    block_network=not self.config.network_access,
                    timeout=timeout,
                )
            self._sandbox_id = self._sandbox.object_id
            logger.info(
                "modal: sandbox %s up (image=%s)", self._sandbox_id, self.config.image
            )
            await self._sandbox.filesystem.make_directory.aio(self.config.workdir)
        except (
            Exception
        ) as e:  # provisioning failure is one rollout's problem, not the eval's
            raise ProgramError(f"modal sandbox provisioning failed: {e}") from e

    async def expose(self, port: int) -> str:
        # The sandbox is remote, so reach a host port via a tunnel (one per port). Modal's
        # own forwarding publishes a sandbox port, not a host one, so use the host-side
        # `prime_tunnel` here.
        from prime_tunnel import Tunnel

        tunnel = Tunnel(local_port=port)
        try:
            async with (
                _TUNNEL_LIMITER
            ):  # shared prime_tunnel rate (512/min, runtime-independent)
                url = str(await tunnel.start()).rstrip("/")
        except Exception as e:
            raise ProgramError(f"modal tunnel failed (host port {port}): {e}") from e
        self._tunnels.append(tunnel)
        logger.info("modal: tunnel up at %s (host port %d)", url, port)
        return url

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
            raise ProgramError(f"modal exec failed: {e}") from e
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
            raise ProgramError(
                f"modal background launch failed: {result.stderr.strip()}"
            )

    async def read(self, path: str) -> bytes:
        try:
            return await self._sandbox.filesystem.read_bytes.aio(path)
        except Exception as e:
            raise ProgramError(f"read {path!r}: {e}") from e

    async def write(self, path: str, data: bytes) -> None:
        # Resolve a relative path against the workdir and create its parent first (Modal's
        # write does not mkdir).
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        try:
            await self._sandbox.filesystem.make_directory.aio(
                str(PurePosixPath(target).parent)
            )
            await self._sandbox.filesystem.write_bytes.aio(data, target)
        except Exception as e:
            raise ProgramError(f"write {path!r}: {e}") from e

    def cleanup(self) -> None:
        # Synchronous atexit backstop (the async API can't run once the loop is gone): stop
        # the already-sync tunnels and terminate the sandbox via Modal's sync API, so the
        # costly resource isn't left to its max-lifetime. Idempotent — the async `stop`
        # handles the normal path, and a second terminate is a no-op.
        for tunnel in self._tunnels:
            with contextlib.suppress(Exception):
                tunnel.sync_stop()
        self._tunnels = []
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:  # `_sandbox_id` kept so descriptor survives teardown
            with contextlib.suppress(Exception):
                sandbox.terminate()

    async def stop(self) -> None:
        # Best-effort, idempotent teardown on the normal path: tunnels first, then the
        # sandbox (the costly resource) via the async API. Runs from the rollout's
        # `finally`, so it fires on success, error, and cancellation; `_sandbox` is nulled
        # as the idempotency guard (the atexit `cleanup` then no-ops).
        for tunnel in self._tunnels:
            with contextlib.suppress(Exception):
                tunnel.sync_stop()
        self._tunnels = []
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is None:
            return
        try:
            await sandbox.terminate.aio()
        except Exception as e:
            logger.warning(
                "modal: failed to terminate sandbox %s: %s", self._sandbox_id, e
            )
