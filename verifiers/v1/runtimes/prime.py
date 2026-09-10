"""Remote Prime sandbox runtime.

`expose` (sandbox port -> public URL) uses the SDK's native exposure (`client.expose`), so a
host-side harness/framework can reach a tool server hosted in the sandbox. The reverse
direction (a program in the sandbox reaching a host service) is the shared host-side
`Tunnel` (interception.tunnel), not the runtime's concern.
"""

import asyncio
import base64
import contextlib
import functools
import logging
import math
import shlex
import tempfile
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from prime_sandboxes import (
    APIError,
    APITimeoutError,
    AsyncSandboxProcess,
    CommandTimeoutError,
    DownloadTimeoutError,
    PaymentRequiredError,
    SandboxFileNotFoundError,
    SandboxImagePullError,
    SandboxTimeoutError,
    UnauthorizedError,
    UploadTimeoutError,
)
from prime_sandboxes.models import validate_egress_lists
from pydantic import Field, model_validator

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError, sandbox_fault_code
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeProcess,
    parse_gpu,
)
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.prime import ensure_prime_auth

logger = logging.getLogger(__name__)

EFFECTIVELY_UNBOUNDED_SECONDS = 30 * 24 * 60 * 60
"""Safety deadline for APIs that require a finite bound. Normal execution remains
bounded by idle detection or rollout cancellation; 30 days is above any real run."""


BASE_LABELS: list[str] = []

EGRESS_APPLY_TIMEOUT: float = 60
"""Seconds `prepare_execution` waits for the platform to report an egress policy applied."""

_SDK_TIMEOUTS = (
    APITimeoutError,
    CommandTimeoutError,
    DownloadTimeoutError,
    UploadTimeoutError,
    SandboxTimeoutError,
)
_RPC_FAULTS = {
    Code.NOT_FOUND: "not_found",
    Code.DEADLINE_EXCEEDED: "timeout",
    Code.UNAVAILABLE: "unavailable",
    Code.RESOURCE_EXHAUSTED: "unavailable",
    Code.PERMISSION_DENIED: "denied",
    Code.UNAUTHENTICATED: "denied",
}


def _chain(e: BaseException) -> Iterator[BaseException]:
    """`e` and the failures it wraps, outermost first. SDK gap: the base `APIError` names the
    HTTP status only in its text (`"HTTP 503: ..."`) and is raised inside the `httpx` handler
    without `from`, so the typed status is reached through its `__context__` here — the one
    place that hop is taken, instead of parsing the message."""
    current: BaseException | None = e
    while current is not None:
        yield current
        if isinstance(current, APIError) and current.__cause__ is None:
            current = current.__context__
        else:
            current = current.__cause__


def _fault_code(e: BaseException) -> str | None:
    """The `SandboxError.code` an SDK failure identifies: its exception type, the Connect RPC
    code or `httpx` status it wraps, or a Python-level fault (`sandbox_fault_code`)."""
    for current in _chain(e):
        if isinstance(current, SandboxFileNotFoundError):
            return "not_found"
        if isinstance(current, (UnauthorizedError, PaymentRequiredError)):
            return "denied"
        if isinstance(current, _SDK_TIMEOUTS):
            return "timeout"
        if isinstance(current, SandboxImagePullError):
            return "provisioning"
        if isinstance(current, ConnectError) and current.code in _RPC_FAULTS:
            return _RPC_FAULTS[current.code]
        if (code := sandbox_fault_code(current)) is not None:
            return code
    return None


def _cancelled() -> bool:
    """Whether the current task is being cancelled — what tells a swallowed cancel from a
    fault. connectrpc turns the `CancelledError` raised inside an RPC into
    `ConnectError(CANCELED, "Request was cancelled")` and the SDK re-wraps that as `APIError`,
    so a cancelled task comes back from a sandbox call with an ordinary error while
    `Task.cancelling()` stays set. Only that count is consulted, never the error's chain: an
    `asyncio.timeout` deadline cancels the task the same way and meets the same rewrite, and
    only the scope owning the deadline tells the two apart — from the `CancelledError` it is
    handed while the count is still set. Past the scope the task is uncancelled, and a CANCELED
    code under an ordinary error is a deadline that already fired (or another task's cancel,
    batched in by the SDK's coalesced polls), not this task's cancel."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


@contextlib.contextmanager
def _faults(what: str, *, fallback: str | None = None) -> Iterator[None]:
    """Wrap one SDK call. Its failure is re-raised as the task's cancellation when that is what
    the SDK reported (`_cancelled`) — a `SandboxError` there would let a caller that retries on
    sandbox faults swallow the cancel and keep working — else as a `SandboxError` carrying the
    typed fault code (`fallback` when nothing typed says). Sits directly around the call, inside
    any `asyncio.timeout` scope over it: the cancel must be re-raised while the task still
    counts as cancelling, where the scope turns its own deadline into `TimeoutError` and lets
    an external cancel through. Past the scope the swallowed deadline is an ordinary error, and
    would surface as the cancel of a task nobody cancelled."""
    try:
        yield
    except SandboxError:
        raise
    except Exception as e:
        if _cancelled():
            raise asyncio.CancelledError() from e
        raise SandboxError(f"{what}: {e}", code=_fault_code(e) or fallback) from e


def _cancel_aware(can_reconnect: Callable[..., bool]) -> Callable[..., bool]:
    """`AsyncSandboxProcess._can_reconnect` with the pump's own cancellation terminal."""

    @functools.wraps(can_reconnect)
    def wrapper(self: Any, reconnects: int, error: BaseException | None) -> bool:
        return not _cancelled() and can_reconnect(self, reconnects, error)

    return wrapper


# The SDK's live-process stream pump (`AsyncSandboxProcess._pump`, a task of its own) meets the
# same swallow: cancelled (by `aclose`, or by the loop's shutdown cancelling every task), its
# stream raises `ConnectError(CANCELED)` rather than `CancelledError`, and `_can_reconnect` — a
# deny-list of definitive codes — calls that recoverable and re-attaches ("live process stream
# dropped (Request was cancelled); re-attaching 1/5"), so a cancelled pump streams on until the
# remote process exits and `asyncio.run` never returns. Until the SDK owns the cancel, the
# predicate is wrapped here so the pump ends instead; guarded, so an SDK that renames it is
# left alone.
_sdk_can_reconnect = getattr(AsyncSandboxProcess, "_can_reconnect", None)
if _sdk_can_reconnect is not None and not hasattr(_sdk_can_reconnect, "__wrapped__"):
    AsyncSandboxProcess._can_reconnect = _cancel_aware(_sdk_can_reconnect)


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
    platform auto-builds what the sandbox needs from it (a VM image for `vm` sandboxes,
    ~10 minutes) and caches the result, so later sandboxes on the same ref start in
    seconds."""
    workdir: str = "/app"
    vm: bool = True
    """Run as a micro-VM rather than a container (kernel features / stronger isolation)."""
    guaranteed: bool = False
    """Request guaranteed (vs best-effort) capacity."""
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
    """Pace sandbox creation to this many per minute, enforced user-wide across every
    env-server worker process (None/<= 0 disables it). (Tunnel creation is limited separately
    and globally — see interception.tunnel.prime.TUNNEL_LIMITER.)"""

    @model_validator(mode="after")
    def _validate_egress(self) -> "PrimeConfig":
        if not self.network_restricted:
            return self
        if not self.vm:
            raise ValueError(
                "Prime allow/block egress lists require a VM sandbox (vm=true)"
            )
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

    async def terminate(self) -> None:
        await self._process.terminate()

    async def kill(self) -> None:
        await self._process.kill()


class PrimeRuntime(Runtime):
    is_local: ClassVar[bool] = False

    def __init__(self, config: PrimeConfig, name: str | None = None) -> None:
        ensure_prime_auth()
        super().__init__(name)
        self.config = config
        self.info = PrimeRuntimeInfo(**config.model_dump())
        self._client = None

    @property
    def supports_live_processes(self) -> bool:
        return self.config.vm

    @property
    def published_port(self) -> int | None:
        return SERVICE_PORT

    async def start(self) -> None:
        from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

        loop = asyncio.get_running_loop()
        shared = _shared_clients.get(loop)
        if shared is None:
            shared = _shared_clients[loop] = _SharedClient(AsyncSandboxClient())
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
            # -1 is prime's convention for no lifetime limit; containers with an
            # idle timeout must carry a finite lifetime as a safety fallback (which
            # must exceed the idle timeout)
            "timeout_minutes": (
                -1
                if self.config.vm or idle_minutes is None
                else max(EFFECTIVELY_UNBOUNDED_SECONDS // 60, idle_minutes + 1)
            ),
            "idle_timeout_minutes": idle_minutes,
            "gpu_type": gpu_type,
            "region": self.config.region,
        }
        # provisioning failure is one rollout's problem, not the eval's
        with _faults("prime sandbox provisioning failed", fallback="provisioning"):
            async with (
                creation_limiter(
                    (self.config.creates_per_min or 0) / 60, "prime-sandbox"
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
                            labels=list(
                                dict.fromkeys([*BASE_LABELS, *self.config.labels])
                            ),
                            docker_image=self.config.image,
                            vm=self.config.vm,
                            guaranteed=self.config.guaranteed,
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

    async def prepare_execution(self, routes: list[str] | None) -> None:
        """Apply the host policy after setup and wait until the platform enforces it."""
        if not self.network_restricted:
            return
        with _faults("prime egress policy failed"):
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
            async with asyncio.timeout(EGRESS_APPLY_TIMEOUT):
                delay = 0.1
                while not status.applied:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 3)
                    # Inside the deadline's scope, so its expiry lands below as `TimeoutError`.
                    with _faults("prime egress policy failed"):
                        status = await self._client.get_network(self.info.id)
        except TimeoutError as e:
            raise SandboxError(
                "prime egress policy was not applied within "
                f"{EGRESS_APPLY_TIMEOUT:g}s on sandbox {self.info.id}; refusing to start "
                "the agent unrestricted",
                code="timeout",
            ) from e
        logger.info(
            "prime: egress policy applied on sandbox %s (allow=%s block=%s)",
            self.info.id,
            policy.get("allow"),
            policy.get("deny"),
        )

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        # a sandbox/API failure is one rollout's problem, not the eval's
        with _faults("prime exec failed"):
            # Poll directly so rollout cancellation owns the execution timeout.
            job = await self._client.start_background_job(
                self.info.id,
                shlex.join(argv),
                working_dir=self.config.workdir,
                env=self.process_env(env),
            )
            delay = 0.1
            while True:
                result = await self._client.get_background_job(self.info.id, job)
                if result.completed:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, 3)
        return ProgramResult(
            exit_code=result.exit_code or 0,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        if not self.config.vm:
            raise SandboxError(
                "persistent harness sessions on Prime require a VM sandbox; "
                "set runtime.prime.vm=true"
            )
        with _faults("prime live process failed to start"):
            process = await self._client.open_process(
                self.info.id,
                shlex.join(argv),
                working_dir=self.config.workdir,
                env=self.process_env(env),
            )
        return PrimeProcess(process)

    async def expose(self, port: int) -> str | None:
        # Publish a server hosted IN the sandbox via the SDK's native port exposure → a public
        # HTTPS URL. Removed when the sandbox is deleted in stop(), so a tool in its own prime
        # sandbox needs no host tunnel. Port exposure is region-gated: many regions (incl. the
        # backend default, which lands in us-central) 400 it; `us` supports it. TODO: re-enable the
        # prime cases in the e2e `skip_if_unexposable` guard once prime exposes ports in any region.
        # surface prime's exposure constraints actionably
        with _faults(
            "prime port exposure failed — port exposure isn't supported in this sandbox's "
            "region; pin `tools.runtime.region` to a region that supports it (e.g. `us`), or "
            "use a colocated / docker / modal tools.runtime instead"
        ):
            exposed = await self._client.expose(self.info.id, port)
        logger.info("prime: exposed sandbox port %d at %s", port, exposed.url)
        return exposed.url.rstrip("/")

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        command = f"exec {shlex.join(argv)} > {shlex.quote(log)} 2>&1"
        with _faults("prime background launch failed"):
            await self._client.start_background_job(
                self.info.id,
                command,
                working_dir=self.config.workdir,
                env=self.process_env(env),
            )

    async def _read(self, path: str, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None and self.config.vm:
            with _faults(f"read {path!r}"):
                # VM execute_command uses bash and returns the complete output stream.
                result = await self._client.execute_command(
                    self.info.id,
                    f"set -o pipefail; head -c {max_bytes} -- {shlex.quote(path)} | base64",
                    working_dir=self.config.workdir,
                    env=self.process_env({}),
                    timeout=EFFECTIVELY_UNBOUNDED_SECONDS,
                )
            if result.exit_code:
                raise SandboxError(f"read {path!r}: {result.stderr.strip()[-500:]}")
            return base64.b64decode(result.stdout)
        if max_bytes is not None:
            return await super()._read(path, max_bytes)
        # Avoid background-job log limits and base64 overhead by downloading binary data directly.
        # The temporary file is removed on every exit, and its byte read stays off the event loop.
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        with _faults(f"read {path!r}"), tempfile.TemporaryDirectory() as directory:
            download = Path(directory) / "download"
            await self._client.download_file(self.info.id, target, str(download))
            return await asyncio.to_thread(download.read_bytes)

    async def write(self, path: str, data: bytes) -> None:
        # The gateway creates missing parents and uploads binary data without command-line
        # limits. Resolve relative paths here because uploads do not use this runtime's workdir.
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        with _faults(f"write {path!r}"):
            await self._client.upload_bytes(
                self.info.id, target, data, filename=PurePosixPath(target).name
            )

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
