"""Remote Prime VM sandbox runtime."""

import asyncio
import contextlib
import io
import logging
import math
import shlex
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

from prime_sandboxes.models import validate_egress_lists
from pydantic import Field, model_validator

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeProcess,
    parse_gpu,
)
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.prime import ensure_prime_auth
from verifiers.v1.utils.scope import run_scope

logger = logging.getLogger(__name__)

EFFECTIVELY_UNBOUNDED_SECONDS = 30 * 24 * 60 * 60
"""Safety deadline for APIs that require a finite bound. Normal execution remains
bounded by idle detection or rollout cancellation; 30 days is above any real run."""
# The SDK reads finished jobs' output through a bounded pool whose deadline includes time
# spent queued; its defaults (20 reads, 45s) expire most reads when thousands of rollouts
# finish a job at once (e.g. every harness's setup at a 3k-rollout start).
_OUTPUT_READS = 100
"""Concurrent output reads on the shared client."""
_OUTPUT_DEADLINE_SECONDS = 300
"""Deadline for one read of a finished job's output, queue time included."""
_OUTPUT_RETRIES = 10
"""Re-reads of a finished job's output that the SDK still failed to fetch, before the
exec is reported as failed."""


BASE_LABELS: list[str] = []


@dataclass
class _SharedClient:
    client: Any
    leases: int = 0


# One `AsyncSandboxClient` per event loop, leased by every live runtime on it. The SDK
# coalesces concurrent status polls per client into batched requests (up to 100 ids
# each), so sharing a client turns N runtimes' creation/job polls into a few batch
# calls. Keyed by loop (not plain process-global) so a client's tasks stay on the loop
# that created it.
_shared_clients: dict[asyncio.AbstractEventLoop, _SharedClient] = {}


def set_base_sandbox_labels(labels: list[str]) -> None:
    """Set process-wide base labels attached to every Prime sandbox, extended by each
    runtime's ``PrimeConfig.labels``. Call it in the process that creates the sandboxes
    (env-server workers set it via their setup hook) — e.g. a trainer stamps its run
    name so every sandbox of a run is findable on the platform."""
    global BASE_LABELS
    BASE_LABELS = list(labels)


class PrimeConfig(NetworkPolicyConfig):
    type: Literal["prime"] = "prime"
    image: str = "python:3.11-slim"
    """Docker image to run. Any pullable ref works: on the first use of an image, the
    platform auto-builds the VM image that the sandbox needs from it (~10 minutes)
    and caches the result, so later sandboxes on the same ref start in
    seconds."""
    workdir: str | None = None
    """Working directory override; None uses the task's workdir, or /app."""
    region: str | None = None
    """Region to provision in (None = provider-chosen)."""
    labels: list[str] = Field(default_factory=list)
    """Labels attached to the sandbox, extending any process-wide base labels (see ``set_base_sandbox_labels``)."""
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float = 1.0
    """CPU cores."""
    memory: float = 2.0
    """Memory in GB."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "A100:2" (a bare count = provider-chosen type)."""
    disk: float = 5.0
    """Disk in GB."""
    idle_timeout: float | None = 3600
    """Seconds of inactivity before the sandbox self-deletes (None disables)."""
    creates_per_min: int | None = None
    """Pace sandbox creation to this many per minute, enforced run-wide across every
    env-server worker process (None/<= 0 disables it). (Tunnel creation is limited separately
    — see interception.tunnel.prime.tunnel_limiter.)"""

    @model_validator(mode="after")
    def _validate_egress(self) -> "PrimeConfig":
        if not self.network_restricted:
            return self
        if not self.allow:
            return self
        validate_egress_lists(
            None if self.allow == ["*"] else self.allow,
            self.block or None,
        )
        return self


class PrimeRuntimeInfo(PrimeConfig, BaseRuntimeInfo):
    image_cached: bool | None = None
    """Whether the platform already had the image at create (None until then). False means
    a first-use auto-build ran while this sandbox waited to start."""


class PrimeProcess(RuntimeProcess):
    def __init__(self, process) -> None:
        self._process = process
        self.stdout: AsyncIterator[bytes] = process.stdout
        self.stderr: AsyncIterator[bytes] = process.stderr

    async def write(self, data: bytes) -> None:
        await self._process.write_stdin(data)

    async def wait(self) -> int:
        return await self._process.wait()

    async def poll(self) -> int | None:
        return self._process.returncode

    async def terminate(self) -> None:
        await self._process.terminate()

    async def kill(self) -> None:
        await self._process.kill()


class PrimeRuntime(Runtime):
    is_local: ClassVar[bool] = False

    def __init__(self, config: PrimeConfig, name: str | None = None) -> None:
        ensure_prime_auth()
        super().__init__(name)
        self.config = config.model_copy(update={"workdir": config.workdir or "/app"})
        self.info = PrimeRuntimeInfo(**self.config.model_dump())
        self._client = None

    @property
    def supports_live_processes(self) -> bool:
        return True

    async def start(self) -> None:
        from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

        loop = asyncio.get_running_loop()
        shared = _shared_clients.get(loop)
        if shared is None:
            shared = _shared_clients[loop] = _SharedClient(
                AsyncSandboxClient(background_job_output_concurrency=_OUTPUT_READS)
            )
        shared.leases += 1
        self._client = shared.client
        # Map the resources onto prime's API (minutes, split GPU; memory/disk are already
        # GB). gpu_type/region are only sent when set (else provider-chosen).
        gpu_type, gpu_count = parse_gpu(self.config.gpu)
        # prime's idle timeout is in whole minutes; convert from the seconds config surface
        # (raised to the SDK's 1-minute minimum).
        idle_minutes = (
            max(1, math.ceil(self.config.idle_timeout / 60))
            if self.config.idle_timeout is not None
            else None
        )
        options = {
            "cpu_cores": self.config.cpu,
            "memory_gb": self.config.memory,
            "disk_size_gb": self.config.disk,
            "gpu_count": gpu_count,
            "timeout_minutes": -1,
            "idle_timeout_minutes": idle_minutes,
            "gpu_type": gpu_type,
            "region": self.config.region,
        }
        scope = run_scope()
        labels = [*BASE_LABELS, *self.config.labels, scope]
        try:
            async with (
                creation_limiter(
                    (self.config.creates_per_min or 0) / 60,
                    "prime-sandbox",
                    scope,
                )
                or contextlib.nullcontext()
            ):
                # Shielded through the id capture: a cancel that aborts the POST
                # mid-flight leaves the platform creating a sandbox this side
                # never learned the id of — teardown() then cannot delete it
                async def create_and_capture_id():
                    sandbox = await self._client.create(
                        CreateSandboxRequest(
                            name=self.name,
                            labels=list(dict.fromkeys(labels)),
                            docker_image=self.config.image,
                            environment_vars=self.env,
                            **{k: v for k, v in options.items() if v is not None},
                        )
                    )
                    self.info.id = sandbox.id
                    return sandbox

                sandbox = await run_shielded(create_and_capture_id())
            # The create response says whether the platform already has the image:
            # `pending_image_build_id` set means a first-use auto-build is running and the
            # sandbox stays PENDING until it finishes (`wait_for_creation` gives that phase
            # its own budget, separate from the normal boot attempts).
            self.info.image_cached = sandbox.pending_image_build_id is None
            if not self.info.image_cached:
                logger.warning(
                    "prime: image %s isn't cached on the platform - auto-building it "
                    "(sandbox %s waits for the build; first use of an image can take "
                    "~10 minutes, later runs start in seconds)",
                    self.config.image,
                    self.info.id,
                )
            await self._client.wait_for_creation(self.info.id, max_attempts=180)
            logger.info(
                "prime: sandbox %s up (image=%s)", self.info.id, self.config.image
            )
            await self._client.execute_command(
                self.info.id, f"mkdir -p {shlex.quote(self.config.workdir)}"
            )
        except (
            Exception
        ) as e:  # provisioning failure is one rollout's problem, not the eval's
            raise SandboxError(f"prime sandbox provisioning failed: {e}") from e

    async def prepare_execution(self, routes: list[str] | None) -> None:
        """Apply the host policy after setup and wait until the platform enforces it."""
        if not self.network_restricted:
            return
        try:
            if routes is None:
                policy = {"allow": ["*"]}
            else:
                hosts = list(
                    dict.fromkeys(
                        h for h in (urlsplit(route).hostname for route in routes) if h
                    )
                )
                if self.config.allow == ["*"]:
                    policy = {"deny": self.config.block}
                else:
                    entries = list(dict.fromkeys([*hosts, *self.config.allow]))
                    validate_egress_lists(entries, None)
                    policy = {"allow": entries} if entries else {"deny": ["*"]}
            status = await self._client.set_network(self.info.id, **policy)
            try:
                async with asyncio.timeout(60):
                    delay = 0.1
                    while not status.applied:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 3)
                        status = await self._client.get_network(self.info.id)
            except TimeoutError as e:
                raise SandboxError(
                    "prime egress policy was not applied within 60s on sandbox "
                    f"{self.info.id}; refusing to start the agent unrestricted"
                ) from e
        except SandboxError:
            raise
        except Exception as e:
            raise SandboxError(f"prime egress policy failed: {e}") from e
        logger.info(
            "prime: egress policy applied on sandbox %s (allow=%s block=%s)",
            self.info.id,
            policy.get("allow"),
            policy.get("deny"),
        )

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        try:
            # Poll directly so rollout cancellation owns the execution timeout.
            job = await self._client.start_background_job(
                self.info.id,
                shlex.join(argv),
                working_dir=self.config.workdir,
                env=self.process_env(env),
            )
            delay = 0.1
            output_retries = 0
            missing = None
            while True:
                result = await self._client.get_background_job(
                    self.info.id, job, timeout=_OUTPUT_DEADLINE_SECONDS
                )
                # Under load the SDK can see a job finish yet miss its output (its bounded
                # output reads expire while queued) and reports that as `*_error`. The job
                # is done, so asking again re-reads the output.
                missing = result.stdout_error or result.stderr_error
                if result.completed and (
                    missing is None or output_retries == _OUTPUT_RETRIES
                ):
                    break
                if result.completed:
                    output_retries += 1
                await asyncio.sleep(delay)
                delay = min(delay * 2, 3)
        except (
            Exception
        ) as e:  # a sandbox/API failure is one rollout's problem, not the eval's
            raise SandboxError(f"prime exec failed: {e}") from e
        if missing is not None:
            raise SandboxError(f"prime exec output unavailable: {missing}")
        if result.exit_code is None:
            raise SandboxError("prime exec completed without an exit code")
        return ProgramResult(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        try:
            process = await self._client.open_process(
                self.info.id,
                shlex.join(argv),
                working_dir=self.config.workdir,
                env=self.process_env(env),
            )
        except Exception as e:
            raise SandboxError(f"prime live process failed to start: {e}") from e
        return PrimeProcess(process)

    async def expose(self, port: int) -> str:
        raise SandboxError(
            "Prime VM sandboxes do not support port exposure; colocate the service "
            "with its consumer, or use a docker or modal service runtime"
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        command = f"exec {shlex.join(argv)} > {shlex.quote(log)} 2>&1"
        try:
            await self._client.start_background_job(
                self.info.id,
                command,
                working_dir=self.config.workdir,
                env=self.process_env(env),
            )
        except Exception as e:
            raise SandboxError(f"prime background launch failed: {e}") from e

    async def _read(self, path: str, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None:
            try:
                # Stream binary output: execute_command buffers base64 text for the
                # entire file. Bound the source read and the host buffer independently.
                process = await self._client.open_process(
                    self.info.id,
                    f"head -c {max_bytes} -- {shlex.quote(path)}",
                    working_dir=self.config.workdir,
                    env=self.process_env({}),
                )
                async with contextlib.aclosing(process):
                    with io.BytesIO() as data:
                        async for chunk in process.stdout:
                            if data.tell() + len(chunk) > max_bytes:
                                raise SandboxError(
                                    "read stream exceeded its byte limit"
                                )
                            data.write(chunk)
                        stderr = b""
                        async for chunk in process.stderr:
                            stderr = (stderr + chunk)[-500:]
                        if await process.wait():
                            raise SandboxError(stderr.decode(errors="replace").strip())
                        return data.getvalue()
            except Exception as exc:
                raise SandboxError(f"read {path!r}: {exc}") from exc
        # Avoid background-job log limits and base64 overhead by downloading binary data directly.
        # The temporary file is removed on every exit, and its byte read stays off the event loop.
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                download = Path(directory) / "download"
                await self._client.download_file(self.info.id, target, str(download))
                return await asyncio.to_thread(download.read_bytes)
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}") from e

    async def write(self, path: str, data: bytes) -> None:
        # The gateway creates missing parents and uploads binary data without command-line
        # limits. Resolve relative paths here because uploads do not use this runtime's workdir.
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        try:
            await self._client.upload_bytes(
                self.info.id, target, data, filename=PurePosixPath(target).name
            )
        except Exception as e:
            raise SandboxError(f"write {path!r}: {e}") from e

    def cleanup(self) -> None:
        # Synchronous atexit backstop (the async client can't run once the loop is gone): delete
        # the sandbox via the sync client, so the costly resource isn't left to its max-lifetime.
        # Idempotent — the async `stop` deletes it on the normal path, a second delete 404s.
        if self.info.id is not None:
            from prime_sandboxes import SandboxClient
            from prime_sandboxes.core import APIClient

            with contextlib.suppress(Exception):
                SandboxClient(APIClient()).delete(self.info.id)

    async def teardown(self) -> None:
        # Best-effort, idempotent teardown: delete the sandbox (the costly resource). Runs via
        # `stop`, shielded from cancellation, so it fires on success, error, and Ctrl-C.
        client, self._client = self._client, None  # `_client` is the idempotency guard
        if client is None:
            return
        try:
            if self.info.id is not None:  # keep info.id available after teardown
                await client.delete(self.info.id)
        except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
            logger.warning("prime: failed to delete sandbox %s: %s", self.info.id, e)
        finally:
            loop = asyncio.get_running_loop()
            shared = _shared_clients[loop]
            shared.leases -= 1
            if not shared.leases:  # last runtime on this loop closes the client
                del _shared_clients[loop]
                with contextlib.suppress(Exception):
                    await client.aclose()
