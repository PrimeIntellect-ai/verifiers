"""Remote Prime sandbox runtime: run the program in a sandbox, reached via native port exposure.

`expose` (sandbox port -> public URL) uses the SDK's native exposure (`client.expose`), so a
host-side harness/framework can reach a tool/user server hosted in the sandbox. The reverse
direction (a program in the sandbox reaching a host service) is the shared host-side
`host_endpoint` tunnel, not the runtime's concern.
"""

import base64
import contextlib
import logging
import shlex
from pathlib import PurePosixPath
from typing import ClassVar, Literal

from pydantic_config import BaseConfig

from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes.base import SERVICE_PORT, ProgramResult, Runtime, parse_gpu
from verifiers.v1.runtimes.limiters import creation_limiter

logger = logging.getLogger(__name__)


# "auto" timeout requests the max prime allows (24h).
_MAX_TIMEOUT_SECONDS = 24 * 60 * 60


class PrimeConfig(BaseConfig):
    type: Literal["prime"] = "prime"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    network_access: bool = True
    vm: bool = False
    """Run as a micro-VM rather than a container (kernel features / stronger isolation)."""
    guaranteed: bool = False
    """Request guaranteed (vs best-effort) capacity."""
    region: str | None = None
    """Region to provision in (None = provider-chosen)."""
    labels: list[str] = []
    """Labels attached to the sandbox and its tunnels — e.g. to group every resource a run
    creates. When unset, the eval defaults them to the run's uuid (see `run_eval`)."""
    timeout: int | Literal["auto"] = 21600
    """Max sandbox lifetime in seconds (default 6h; or "auto" = the highest prime
    supports). A hard backstop: the sandbox self-terminates even if local cleanup is
    skipped."""
    # Resources, in Modal's units (also settable per-task via Task.resources, with
    # precedence cli/toml > task > this default). Mapped to prime's API in `start`.
    cpu: float = 1.0
    """CPU cores."""
    memory: float = 2.0
    """Memory in GB."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "A100:2" (a bare count = provider-chosen type)."""
    disk: float = 5.0
    """Disk in GB."""
    creates_per_min: int | None = None
    """Pace sandbox creation to this many per minute, enforced host-wide across every
    env-server worker process (None/<= 0 disables it). (Tunnel creation is limited separately
    and globally — see limiters.TUNNEL_LIMITER.)"""


class PrimeRuntime(Runtime):
    """Runs the program in a Prime sandbox; the server is reached via a tunnel."""

    is_local: ClassVar[bool] = False

    def __init__(self, config: PrimeConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self._client = None
        self._sandbox_id: str | None = None

    @property
    def descriptor(self) -> str | None:
        return self._sandbox_id

    @property
    def published_port(self) -> int | None:
        return SERVICE_PORT

    async def start(self) -> None:
        from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

        self._client = AsyncSandboxClient()
        timeout = (
            _MAX_TIMEOUT_SECONDS
            if self.config.timeout == "auto"
            else self.config.timeout
        )
        # Map the resources onto prime's API (minutes, split GPU; memory/disk are already
        # GB). gpu_type/region are only sent when set (else provider-chosen).
        gpu_type, gpu_count = parse_gpu(self.config.gpu)
        options = {
            "cpu_cores": self.config.cpu,
            "memory_gb": self.config.memory,
            "disk_size_gb": self.config.disk,
            "gpu_count": gpu_count,
            "timeout_minutes": timeout // 60,
            "gpu_type": gpu_type,
            "region": self.config.region,
        }
        try:
            async with (
                creation_limiter(
                    (self.config.creates_per_min or 0) / 60, "prime-sandbox"
                )
                or contextlib.nullcontext()
            ):
                sandbox = await self._client.create(
                    CreateSandboxRequest(
                        name=self.name,
                        labels=self.config.labels,
                        docker_image=self.config.image,
                        network_access=self.config.network_access,
                        vm=self.config.vm,
                        guaranteed=self.config.guaranteed,
                        **{k: v for k, v in options.items() if v is not None},
                    )
                )
            self._sandbox_id = sandbox.id
            await self._client.wait_for_creation(self._sandbox_id)
            logger.info(
                "prime: sandbox %s up (image=%s)", self._sandbox_id, self.config.image
            )
            await self._client.run_background_job(
                self._sandbox_id, f"mkdir -p {shlex.quote(self.config.workdir)}"
            )
        except (
            Exception
        ) as e:  # provisioning failure is one rollout's problem, not the eval's
            raise ProgramError(f"prime sandbox provisioning failed: {e}") from e

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        try:
            result = await self._client.run_background_job(
                self._sandbox_id,
                shlex.join(argv),
                working_dir=self.config.workdir,
                env=env,
                timeout=(
                    _MAX_TIMEOUT_SECONDS
                    if self.config.timeout == "auto"
                    else self.config.timeout
                ),
            )
        except (
            Exception
        ) as e:  # a sandbox/API failure is one rollout's problem, not the eval's
            raise ProgramError(f"prime exec failed: {e}") from e
        return ProgramResult(
            exit_code=result.exit_code or 0,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def expose(self, port: int) -> str | None:
        # Publish a server hosted IN the sandbox via the SDK's native port exposure → a public
        # HTTPS URL reachable from anywhere (incl. another sandbox). The exposure is removed when
        # the sandbox is deleted in stop(), so a tool in its own prime sandbox needs no host tunnel.
        # TODO: `client.expose` currently only works in a default-region sandbox (port <= 9000), so
        # a tool/user-sim in its own prime sandbox needs the region pinned. Fix once prime supports
        # port exposure in any region (then drop the e2e `skip_if_unexposable` guard).
        try:
            exposed = await self._client.expose(self._sandbox_id, port)
        except Exception as e:  # surface prime's exposure constraints actionably
            raise ProgramError(
                "prime port exposure failed — `client.expose` currently needs a default-region "
                "sandbox and port <= 9000; pin `tools.runtime.region` to a supported "
                f"region, or use a host/docker tools.runtime instead. ({e})"
            ) from e
        logger.info("prime: exposed sandbox port %d at %s", port, exposed.url)
        return exposed.url.rstrip("/")

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # `&` backgrounds inside the sandbox; the job returns immediately, the process
        # lives until the sandbox is deleted in stop().
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise ProgramError(
                f"prime background launch failed: {result.stderr.strip()}"
            )

    async def read(self, path: str) -> bytes:
        result = await self.run(["sh", "-c", f"base64 {shlex.quote(path)}"], {})
        if result.exit_code != 0:
            raise ProgramError(f"read {path!r}: {result.stderr.strip()}")
        return base64.b64decode(result.stdout)

    async def write(self, path: str, data: bytes) -> None:
        # Upload via the gateway (multipart) — never inline the bytes on the command line
        # (a large file, e.g. a task tarball, overflows the exec command-length limit and
        # fails with ENAMETOOLONG). The upload does NOT run in the workdir, so resolve a
        # relative path against it (and mkdir its parent) — otherwise the sidecar writes
        # it somewhere unwritable ("Operation not permitted").
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        await self.run(
            ["sh", "-c", f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))}"],
            {},
        )
        try:
            await self._client.upload_bytes(
                self._sandbox_id, target, data, filename=PurePosixPath(target).name
            )
        except Exception as e:
            raise ProgramError(f"write {path!r}: {e}") from e

    def cleanup(self) -> None:
        # Synchronous atexit backstop (the async client can't run once the loop is gone): delete
        # the sandbox via the sync client, so the costly resource isn't left to its max-lifetime.
        # Idempotent — the async `stop` deletes it on the normal path, a second delete 404s.
        if self._sandbox_id is not None:
            from prime_sandboxes import SandboxClient
            from prime_sandboxes.core import APIClient

            with contextlib.suppress(Exception):
                SandboxClient(APIClient()).delete(self._sandbox_id)

    async def stop(self) -> None:
        # Best-effort, idempotent teardown: delete the sandbox (the costly resource). Runs from the
        # rollout's `finally`, so it fires on success, error, and cancellation.
        client, self._client = self._client, None  # `_client` is the idempotency guard
        if client is None:
            return
        if (
            self._sandbox_id is not None
        ):  # kept (not nulled) so descriptor survives teardown
            try:
                await client.delete(self._sandbox_id)
            except Exception as e:
                logger.warning(
                    "prime: failed to delete sandbox %s: %s", self._sandbox_id, e
                )
        with contextlib.suppress(Exception):
            await client.aclose()
