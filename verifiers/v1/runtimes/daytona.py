"""Remote Daytona sandbox runtime: run the program in a Daytona sandbox, server via tunnel.

The program runs in a remote sandbox and reaches the host interception server over a
tunnel — the host-side `prime_tunnel`, since Daytona's preview links go the other way
(they publish a sandbox port, not a host one). `public_url` uses those preview links
natively, so a tool server in its own Daytona sandbox needs no host middleman.
"""

import contextlib
import logging
import math
import shlex
import uuid
from typing import Literal

from pydantic_config import BaseConfig

from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes.base import ProgramResult, Runtime, parse_gpu
from verifiers.v1.runtimes.limiters import _TUNNEL_LIMITER, creation_limiter

logger = logging.getLogger(__name__)


# "auto" timeout requests the same ceiling as modal/prime (24h). Daytona has no absolute
# lifetime knob, so the backstop is inactivity-based: see `DaytonaConfig.timeout`.
_MAX_TIMEOUT_SECONDS = 24 * 60 * 60
# Sandbox creation cap — covers a cold registry pull of the task image (the SDK default
# of 60s is calibrated for warm snapshots, not arbitrary images).
_CREATE_TIMEOUT_SECONDS = 300


class DaytonaConfig(BaseConfig):
    type: Literal["daytona"] = "daytona"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    network_access: bool = True
    public: bool = False
    """Make the sandbox's preview ports publicly reachable without signing. Not needed
    for `public_url`: on a private sandbox (the default) it mints a *signed* preview
    URL instead (valid for the 24h signing max; dead once the sandbox is deleted)."""
    region: str | None = None
    """Daytona target region, e.g. "us" or "eu" (None = the account default)."""
    timeout: int | Literal["auto"] = 21600
    """Backstop sandbox lifetime in seconds (default 6h; or "auto" = 24h, matching
    modal/prime). Daytona's knob is inactivity-based rather than absolute: the sandbox
    auto-stops after this long without activity and is deleted on stop, so a leaked
    sandbox still tears itself down even if local cleanup is skipped."""
    # Resources, in Modal's units (also settable per-task via Task.resources, with
    # precedence cli/toml > task > this default). Mapped to Daytona's API in `start`.
    cpu: float = 1.0
    """CPU cores."""
    memory: float = 2.0
    """Memory in GB."""
    gpu: str | None = None
    """GPU spec, e.g. "H100" or "H100:2" (type[:count]). Daytona GPU sandboxes must be
    delete-on-stop ("ephemeral"), which the always-on `auto_delete_interval=0` covers."""
    disk: float = 5.0
    """Disk in GB."""
    creates_per_sec: float | None = 5.0
    """Pace sandbox creation to this many per second, enforced host-wide across every
    env-server worker process (None/<= 0 disables it). Daytona's creation limit is
    org-specific (300-600/min on self-serve tiers, higher or custom on dedicated
    plans), so the default takes the self-serve floor (300/min) — raise it to match
    your org, or disable it for orgs with custom limits."""
    connection_pool_maxsize: int = 250
    """Max connections in the SDK client's HTTP pool, per process (the SDK default).
    Raise it for massive runs driving many parallel sandboxes from one process."""


class DaytonaRuntime(Runtime):
    """Runs the program in a Daytona sandbox; the server is reached via a tunnel."""

    def __init__(self, config: DaytonaConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self._daytona = None
        self._sandbox = None
        self._sandbox_id: str | None = None
        self._tunnels: list = []

    @property
    def descriptor(self) -> str | None:
        return self._sandbox_id

    async def start(self) -> None:
        from daytona import (
            AsyncDaytona,
            CreateSandboxFromImageParams,
            GpuType,
            Resources,
        )
        from daytona import DaytonaConfig as SDKConfig

        timeout = (
            _MAX_TIMEOUT_SECONDS
            if self.config.timeout == "auto"
            else self.config.timeout
        )
        # Map the resources onto Daytona's API (whole units, split GPU; memory/disk are
        # already GB). Auth comes from the environment (DAYTONA_API_KEY / DAYTONA_API_URL).
        gpu_type, gpu_count = parse_gpu(self.config.gpu)
        try:
            self._daytona = AsyncDaytona(
                SDKConfig(
                    target=self.config.region,
                    connection_pool_maxsize=self.config.connection_pool_maxsize,
                )
            )
            async with (
                creation_limiter(self.config.creates_per_sec, "daytona-sandbox")
                or contextlib.nullcontext()
            ):
                self._sandbox = await self._daytona.create(
                    CreateSandboxFromImageParams(
                        name=self.name,
                        image=self.config.image,
                        resources=Resources(
                            cpu=math.ceil(self.config.cpu),
                            memory=math.ceil(self.config.memory),
                            disk=math.ceil(self.config.disk),
                            gpu=gpu_count or None,
                            gpu_type=GpuType(gpu_type) if gpu_type else None,
                        ),
                        public=self.config.public,
                        network_block_all=not self.config.network_access,
                        # The lifetime backstop: auto-stop after `timeout` of
                        # inactivity, then delete rather than archive. (The SDK's
                        # `ephemeral=True` is an alias for exactly this setting, so
                        # no separate flag — and it covers GPU sandboxes, which must
                        # be delete-on-stop.)
                        auto_stop_interval=max(1, timeout // 60),
                        auto_delete_interval=0,
                    ),
                    timeout=_CREATE_TIMEOUT_SECONDS,
                )
            self._sandbox_id = self._sandbox.id
            logger.info(
                "daytona: sandbox %s up (image=%s)", self._sandbox_id, self.config.image
            )
            # fs.create_folder has `-p` semantics, is idempotent, and raises on a real
            # failure (a shell mkdir's exit code would be silently ignored).
            await self._sandbox.fs.create_folder(self.config.workdir, "755")
        except (
            Exception
        ) as e:  # provisioning failure is one rollout's problem, not the eval's
            raise ProgramError(f"daytona sandbox provisioning failed: {e}") from e

    async def expose(self, port: int) -> str:
        # The sandbox is remote, so reach a host port via a tunnel (one per port).
        # Daytona's preview links publish a sandbox port, not a host one, so use the
        # host-side `prime_tunnel` here (see `public_url` for the other direction).
        from prime_tunnel import Tunnel

        tunnel = Tunnel(local_port=port)
        try:
            async with (
                _TUNNEL_LIMITER
            ):  # shared prime_tunnel rate (512/min, runtime-independent)
                url = str(await tunnel.start()).rstrip("/")
        except Exception as e:
            raise ProgramError(f"daytona tunnel failed (host port {port}): {e}") from e
        self._tunnels.append(tunnel)
        logger.info("daytona: tunnel up at %s (host port %d)", url, port)
        return url

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        # Daytona's exec returns a single combined output stream, so recover the
        # stdout/stderr split the contract requires in-band: stderr is redirected to a
        # file during the run and emitted after a unique marker, then the two halves
        # are partitioned locally. The program's exit code is preserved across the
        # bookkeeping commands.
        marker = f"__vf_stderr_{uuid.uuid4().hex[:12]}__"
        err = shlex.quote(f"/tmp/.{marker}")
        command = (
            f"{{ {shlex.join(argv)} ; }} 2>{err}; __vf_ec=$?; "
            f"printf '\\n%s\\n' {shlex.quote(marker)}; cat {err}; rm -f {err}; "
            f"exit $__vf_ec"
        )
        try:
            response = await self._sandbox.process.exec(
                command, cwd=self.config.workdir, env=env
            )
        except (
            Exception
        ) as e:  # a sandbox/API failure is one rollout's problem, not the eval's
            raise ProgramError(f"daytona exec failed: {e}") from e
        output = response.result or ""
        stdout, sep, stderr = output.partition(f"\n{marker}\n")
        if not sep:  # marker lost (the wrapper never ran) — surface it all as stdout
            stdout, stderr = output, ""
        return ProgramResult(
            exit_code=response.exit_code if response.exit_code is not None else 0,
            stdout=stdout,
            stderr=stderr,
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # `&` backgrounds inside the sandbox; the job returns immediately, the process
        # lives until the sandbox is deleted in stop().
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise ProgramError(
                f"daytona background launch failed: {result.stderr.strip()}"
            )

    async def public_url(self, port: int) -> str | None:
        # Native preview links → a public HTTPS URL reachable from anywhere (incl.
        # another sandbox), so a tool in its own Daytona sandbox needs no host
        # middleman/tunnel. A plain preview URL is only unauthenticated on a public
        # sandbox, so on a private one (the default) mint a *signed* URL instead:
        # same reachability, no token header. Signed for the 24h signing max
        # unconditionally — `timeout` is an inactivity backstop, so an active sandbox
        # can legitimately outlive it and a shorter signature would go stale
        # mid-rollout; the link stops resolving when the sandbox is deleted anyway.
        try:
            if self.config.public:
                preview = await self._sandbox.get_preview_link(port)
            else:
                preview = await self._sandbox.create_signed_preview_url(
                    port, expires_in_seconds=_MAX_TIMEOUT_SECONDS
                )
        except Exception as e:  # surface daytona's exposure constraints actionably
            raise ProgramError(
                f"daytona port exposure failed (port {port}): {e}"
            ) from e
        logger.info("daytona: exposed sandbox port %d at %s", port, preview.url)
        return str(preview.url).rstrip("/")

    async def read(self, path: str) -> bytes:
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        try:
            data = await self._sandbox.fs.download_file(target)
        except Exception as e:
            raise ProgramError(f"read {path!r}: {e}") from e
        if data is None:
            raise ProgramError(f"read {path!r}: no data returned")
        return data

    async def write(self, path: str, data: bytes) -> None:
        # The upload does NOT run in the workdir (the fs API resolves relative paths
        # against $HOME), so resolve a relative path against it explicitly; the upload
        # itself creates any missing parent directories.
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        try:
            await self._sandbox.fs.upload_file(data, target)
        except Exception as e:
            raise ProgramError(f"write {path!r}: {e}") from e

    def cleanup(self) -> None:
        # Synchronous atexit backstop (the async client can't run once the loop is
        # gone): stop the already-sync tunnels and delete the sandbox via the sync
        # client, so the costly resource isn't left to its inactivity backstop.
        # Keyed off `_sandbox_id`, which is never nulled — a `stop` cancelled
        # mid-delete still gets cleaned here; a duplicate delete just 404s
        # (suppressed).
        for tunnel in self._tunnels:
            with contextlib.suppress(Exception):
                tunnel.sync_stop()
        self._tunnels = []
        self._daytona = None
        self._sandbox = None
        if self._sandbox_id is None:
            return
        from daytona import Daytona

        with contextlib.suppress(Exception):
            client = Daytona()
            client.delete(client.get(self._sandbox_id))

    async def stop(self) -> None:
        # Best-effort, idempotent teardown on the normal path: tunnels first, then the
        # sandbox (the costly resource) via the async API. Runs from the rollout's
        # `finally`, so it fires on success, error, and cancellation; `_daytona` is
        # nulled as the idempotency guard, and if cancellation cuts the delete short,
        # the atexit `cleanup` finishes it by id.
        for tunnel in self._tunnels:
            with contextlib.suppress(Exception):
                tunnel.sync_stop()
        self._tunnels = []
        client, self._daytona = self._daytona, None
        sandbox, self._sandbox = self._sandbox, None
        if client is None:
            return
        if sandbox is not None:  # `_sandbox_id` kept so descriptor survives teardown
            try:
                await client.delete(sandbox)
            except Exception as e:
                logger.warning(
                    "daytona: failed to delete sandbox %s: %s", self._sandbox_id, e
                )
        with contextlib.suppress(Exception):
            await client.close()
