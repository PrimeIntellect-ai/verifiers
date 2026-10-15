"""Remote Prime VM sandbox runtime."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import math
import re
import shlex
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal, TypeVar
from urllib.parse import urlsplit

from prime_sandboxes.models import validate_egress_lists
from pydantic import Field, model_validator

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import (
    SandboxError,
    SandboxErrorKind,
    http_status,
    missing_file_kind,
    sandbox_error_kind,
)
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
    parse_gpu,
)
from verifiers.v1.runtimes.limiters import CreationBacklogFull, creation_limiter
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.aio import cancel_requested, run_shielded
from verifiers.v1.utils.prime import ensure_prime_auth
from verifiers.v1.utils.retries import retrying
from verifiers.v1.utils.scope import run_scope

logger = logging.getLogger(__name__)
T = TypeVar("T")

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

_LIVENESS_INTERVAL_SECONDS = 60
"""How often a running job is checked for a dead wrapper. The SDK's wrapper writes the
exit file; if a command inside the box kills the wrapper (`pkill -f`, `killall sh`),
the file never appears and polling alone would wait forever."""
_LIVENESS_MISSES = 2
"""Consecutive checks that must find the job dead before it is reported dead, so the
moment between the job's output closing and its exit file appearing is never one."""

_BOOT_WAIT_ATTEMPTS = 180
"""Creation-wait budget for a box booting from its image, in the SDK's polls (~355 s)."""
_RESTORE_WAIT_ATTEMPTS = 420
"""Creation-wait budget for a box restored from a checkpoint (~14 min): a restore
copies the checkpoint's data first, and a large box's takes longer than a boot."""
_MISSING_CWD = re.compile(r"cwd '([^']+)' does not exist")
"""The platform's refusal of a command whose working directory is not on the box."""

_CHECKPOINT_DEADLINE_SECONDS = 900
"""Wait for a checkpoint to become restorable. An incremental checkpoint takes seconds;
the first of a box holding several GB can take minutes."""

_CHECKPOINT_DELETE_DEADLINE_SECONDS = 300
"""How long a checkpoint delete keeps retrying the platform's 409. It refuses while a
box started from the checkpoint, or the checkpoint's source box, still runs, and boxes
stop asynchronously after their delete."""
_CHECKPOINT_DELETE_MAX_DELAY_SECONDS = 30
"""Longest wait between two tries of a refused checkpoint delete."""

_TRANSFER_PART_BYTES = 128 * 1024 * 1024
"""Largest single upload or download request. One gateway request of about 2 GB always
fails, and one of 512 MB fails when the gateway drops the connection under load; a
larger file crosses in parts of this size, joined (write) or cut (read) in the box."""
_TRANSFER_ATTEMPTS = 3
"""Tries of one transfer request that failed in transit (a timeout, a dropped
connection, a gateway 5xx). A refused request, a missing file, or a lost box is final."""
_TRANSFER_BACKOFF_SECONDS = 5
"""Wait before a transfer request's second try; it doubles for each later try."""
_TRANSFER_RETRIES = {
    "attempts": _TRANSFER_ATTEMPTS,
    "initial": _TRANSFER_BACKOFF_SECONDS,
}
"""`_repeat`'s budget for one upload or download request: every one is idempotent (an
upload overwrites, a download rewrites its file)."""


BASE_LABELS: list[str] = []


class _JobDied(Exception):
    """A background job's wrapper died before it wrote the job's exit status."""


_API_ATTEMPTS = 4
"""Tries of a repeatable sandbox API call (a job poll, a checkpoint delete) that failed
in transit."""


def _in_transit(error: BaseException) -> bool:
    """Whether a sandbox API call failed in transit (a gateway error, a timeout, a
    dropped connection: `sandbox_error_kind` "io"), so a repeat can ride it out. A lost
    box, a refusal, or a missing file is final."""
    return sandbox_error_kind(error) == "io"


async def _repeat(call, *, attempts: int = _API_ATTEMPTS, initial: float = 1.0):
    """Run an idempotent sandbox API call, repeating it while it fails in transit."""
    async for attempt in retrying(
        when=_in_transit,
        retries=attempts - 1,
        initial=initial,
        label="prime sandbox API call",
    ):
        with attempt:
            return await call()


HELD_KINDS = frozenset({"io", "quota"})
"""Failure kinds an outage hold rides out (`PrimeConfig.outage_budget_s`)."""
HOLD_START_S, HOLD_MAX_S = 1.0, 30.0
"""Seconds a held call waits before it is sent again (doubling), and the longest wait."""
NOT_PLACED = "sandbox_not_placed"
"""The gateway's error code (the `error` field of its 503 body) for a box the routing
holds no placement for: transient while a box comes up; on a box that has answered an
exec, the box is lost and never routed again."""
NOTICE_S = 60.0
"""Seconds between two warnings about held calls; the holds between log at debug."""
_noticed = 0.0
_clock = time.monotonic
"""The clock an outage hold's budget runs on."""


def _notice(message: str, *args: object) -> None:
    global _noticed
    now = time.monotonic()
    level = logging.WARNING if now - _noticed >= NOTICE_S else logging.DEBUG
    if level == logging.WARNING:
        _noticed = now
    logger.log(level, message, *args)


def placement_lost(error: BaseException) -> bool:
    """Whether a failure (or one it chains) is the gateway's 503 naming `NOT_PLACED`:
    from its JSON body when it has a response, else its text."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if http_status(current) == 503:
            body = None
            try:
                body = response.json() if response is not None else None
            except ValueError:
                body = None
            if isinstance(body, dict):
                if body.get("error") == NOT_PLACED:
                    return True
            elif NOT_PLACED in str(current):
                return True
        current = current.__cause__ or current.__context__
    return False


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

_discards: set[asyncio.Task] = set()
"""Background deletes of abandoned checkpoints, held so they are not collected."""


def set_base_sandbox_labels(labels: list[str]) -> None:
    """Set process-wide base labels attached to every Prime sandbox, extended by each
    runtime's ``PrimeConfig.labels``. Call it in the process that creates the sandboxes
    (env-server workers set it via their setup hook) — e.g. a trainer stamps its run
    name so every sandbox of a run is findable on the platform."""
    global BASE_LABELS
    BASE_LABELS = list(labels)


def run_labels() -> list[str]:
    """The labels every Prime sandbox and tunnel this process opens carries, whatever its
    config adds: the base labels and the run scope. ``runtimes.sweep`` deletes by them."""
    return list(dict.fromkeys([*BASE_LABELS, run_scope()]))


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
    """Disk in GB. Unused with ``checkpoint``: a sandbox keeps its checkpoint's disk."""
    checkpoint: str | None = None
    """A DURABLE filesystem checkpoint to start from instead of ``image`` (see the SDK's
    ``checkpoint`` and ``wait_for_checkpoint``)."""
    idle_timeout: float | None = 3600
    """Seconds of inactivity before the sandbox self-deletes (None disables)."""
    transfer_timeout: int = Field(600, gt=0)
    """Seconds one upload or download request may take (the SDK's default is 300). A
    file larger than one request part (128 MiB) takes one request per part. Many
    concurrent transfers share the gateway's bandwidth: 512 MB requests have taken over
    600 s while twenty boxes transferred at once."""
    creates_per_min: int | None = None
    """Pace sandbox creation to this many per minute, enforced run-wide across every
    env-server worker process (None/<= 0 disables it). (Tunnel creation is limited separately
    — see interception.tunnel.prime.tunnel_limiter.)"""
    creates_backlog_s: float | None = Field(None, gt=0)
    """Longest a create may wait for its slot under `creates_per_min` before it fails
    (a `quota` SandboxError) rather than stall behind the backlog; None waits it out."""
    create_timeout_s: float | None = Field(None, gt=0)
    """Seconds the whole provisioning may take (the wait for a creation slot, the create,
    the boot, the workdir) before it fails (`io`) and the late box is deleted; None
    leaves each phase its own budget."""
    outage_budget_s: float = Field(0, ge=0)
    """Seconds an idempotent platform call (the create, whose every send carries one
    idempotency key so a repeat never makes a second box; the boot wait, a job poll, an
    upload or download, the teardown's delete) is held and sent again through a provider outage (an `io` or
    `quota` failure: no route to the box, the API or file gateway dropping connections,
    a 5xx, a 429) before the error reaches the caller: minutes-long outages are ridden
    out instead of failing after a few tries. The waits go 1 s doubling to 30 s, with a
    warning a minute. 0 (the default) keeps the few quick tries alone. A command is never
    sent again (its reply may be lost after it ran), and a box whose placement is lost
    is gone at once (`box_lost`), never held."""

    @model_validator(mode="after")
    def _validate_egress(self) -> PrimeConfig:
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


class PrimeBackend(RuntimeBackend):
    supports_user: ClassVar[bool] = True
    supports_expose: ClassVar[bool] = False
    is_local: ClassVar[bool] = False

    def __init__(self, config: PrimeConfig, name: str | None = None) -> None:
        ensure_prime_auth()
        super().__init__(name)
        self.config = config.model_copy(update={"workdir": config.workdir or "/app"})
        self.info = PrimeRuntimeInfo(**self.config.model_dump())
        self._client = None
        self.placed = False
        """The box has answered an exec: a placement lost from now on is the box lost,
        not an outage."""

    @property
    def supports_live_processes(self) -> bool:
        return True

    def _kind(
        self, error: BaseException, default: SandboxErrorKind = "unknown"
    ) -> SandboxErrorKind:
        """`sandbox_error_kind`, except that a box which answered once and whose
        placement the gateway has lost (`NOT_PLACED`) is `box_lost`, not an outage."""
        if self.placed and placement_lost(error):
            return "box_lost"
        return sandbox_error_kind(error, default)

    async def _held(
        self,
        what: str,
        call: Callable[[], Awaitable[T]],
        *,
        attempts: int = _API_ATTEMPTS,
        initial: float = 1.0,
    ) -> T:
        """`call()`, an idempotent platform call: repeated while it fails in transit
        (`_repeat`), then held through an outage for up to `outage_budget_s`
        (`HOLD_START_S` doubling to `HOLD_MAX_S` between sends). A box that answered
        once and whose placement the gateway has lost (`NOT_PLACED`) is gone at once:
        a `box_lost` SandboxError, never held. A cancellation the SDK swallowed is never
        held through."""

        async def placed_call() -> T:
            try:
                return await call()
            except Exception as e:
                if self.placed and placement_lost(e):
                    raise SandboxError(
                        f"prime {what}: sandbox {self.info.id} lost its placement "
                        f"({NOT_PLACED}): {e}",
                        kind="box_lost",
                    ) from e
                raise

        deadline = _clock() + self.config.outage_budget_s
        delay = HOLD_START_S
        while True:
            try:
                return await _repeat(placed_call, attempts=attempts, initial=initial)
            except Exception as e:
                if (
                    sandbox_error_kind(e) not in HELD_KINDS
                    or cancel_requested()
                    or _clock() + delay > deadline
                ):
                    raise
                _notice(
                    "prime: %s on sandbox %s held %gs through a platform outage: %s",
                    what,
                    self.info.id,
                    delay,
                    e,
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, HOLD_MAX_S)

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
            "gpu_count": gpu_count,
            "timeout_minutes": -1,
            "idle_timeout_minutes": idle_minutes,
            "gpu_type": gpu_type,
            "region": self.config.region,
        }
        if self.config.checkpoint:
            source = {"checkpoint_id": self.config.checkpoint}
        else:
            source = {
                "docker_image": self.config.image,
                "disk_size_gb": self.config.disk,
            }
        scope = run_scope()
        labels = [*BASE_LABELS, *self.config.labels, scope]

        # Every send of this create carries one idempotency key: a send repeated after
        # its reply was lost (the platform may have made the box) gets that box back
        # instead of a second one nobody would delete.
        idempotency_key = uuid.uuid4().hex

        # Shielded through the id capture: a cancel that aborts the POST mid-flight
        # leaves the platform creating a sandbox this side never learned the id of —
        # teardown() then cannot delete it
        async def create_and_capture_id():
            sandbox = await self._client.create(
                CreateSandboxRequest(
                    name=self.name,
                    labels=list(dict.fromkeys(labels)),
                    idempotency_key=idempotency_key,
                    **source,
                    environment_vars=self.env,
                    **{k: v for k, v in options.items() if v is not None},
                )
            )
            self.info.id = sandbox.id
            return sandbox

        bound: asyncio.Timeout | None = None
        try:
            async with asyncio.timeout(self.config.create_timeout_s) as bound:
                async with (
                    creation_limiter(
                        (self.config.creates_per_min or 0) / 60,
                        "prime-sandbox",
                        scope,
                        self.config.creates_backlog_s,
                    )
                    or contextlib.nullcontext()
                ):
                    sandbox = await self._held(
                        "create",
                        lambda: run_shielded(create_and_capture_id()),
                        attempts=1,
                    )
                # The create response says whether the platform already has the
                # image: `pending_image_build_id` set means a first-use auto-build is
                # running and the sandbox stays PENDING until it finishes
                # (`wait_for_creation` gives that phase its own budget, separate from
                # the normal boot attempts).
                self.info.image_cached = sandbox.pending_image_build_id is None
                if not self.info.image_cached:
                    logger.warning(
                        "prime: image %s isn't cached on the platform - auto-building "
                        "it (sandbox %s waits for the build; first use of an image can "
                        "take ~10 minutes, later runs start in seconds)",
                        self.config.image,
                        self.info.id,
                    )
                await self._boot()
                logger.info(
                    "prime: sandbox %s up (image=%s)", self.info.id, self.config.image
                )
                await self._held(
                    "workdir",
                    lambda: self._client.execute_command(
                        self.info.id,
                        f"mkdir -p {shlex.quote(self.config.workdir)}",
                        working_dir="/",
                    ),
                )
                self.placed = True
        except CreationBacklogFull as e:
            raise SandboxError(
                f"prime sandbox provisioning failed: {e}", kind="quota"
            ) from e
        except (
            Exception
        ) as e:  # provisioning failure is one rollout's problem, not the eval's
            if bound is not None and bound.expired():
                # The SDK may answer the cancellation with an error of its own.
                raise SandboxError(
                    "prime sandbox provisioning failed: no box within "
                    f"{self.config.create_timeout_s:g}s: {e}",
                    kind="io",
                ) from e
            raise SandboxError(
                f"prime sandbox provisioning failed: {e}", kind=sandbox_error_kind(e)
            ) from e

    async def _boot(self) -> None:
        """Wait for the created box to boot. The SDK's readiness probe runs in the
        image's WORKDIR, so a box restored from a checkpoint whose filesystem lacks it
        (a run on the box deleted it) never passes the probe: the directory is made
        again from / and the wait runs once more."""

        def wait() -> Any:
            return self._client.wait_for_creation(
                self.info.id,
                max_attempts=_RESTORE_WAIT_ATTEMPTS
                if self.config.checkpoint
                else _BOOT_WAIT_ATTEMPTS,
            )

        try:
            await self._held("boot", wait, attempts=1)
        except Exception as e:
            missing = _MISSING_CWD.search(str(e)) if self.config.checkpoint else None
            if missing is None:
                raise
            logger.warning(
                "prime: sandbox %s restored from %s lacks its image's working "
                "directory %s; making it again",
                self.info.id,
                self.config.checkpoint,
                missing.group(1),
            )
            await self._held(
                "workdir",
                lambda: self._client.execute_command(
                    self.info.id,
                    f"mkdir -p {shlex.quote(missing.group(1))}",
                    working_dir="/",
                ),
            )
            await self._held("boot", wait, attempts=1)

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
                    f"{self.info.id}; refusing to start the agent unrestricted",
                    kind=sandbox_error_kind(e),
                ) from e
        except SandboxError:
            raise
        except Exception as e:
            raise SandboxError(
                f"prime egress policy failed: {e}", kind=sandbox_error_kind(e)
            ) from e
        logger.info(
            "prime: egress policy applied on sandbox %s (allow=%s block=%s)",
            self.info.id,
            policy.get("allow"),
            policy.get("deny"),
        )

    async def _run(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        try:
            # Poll directly so rollout cancellation owns the execution timeout.
            job = await self._client.start_background_job(
                self.info.id,
                shlex.join(argv),
                working_dir=self.config.workdir,
                env=runtime.process_env(env),
                user=runtime.user,
            )
            delay = 0.1
            output_retries = 0
            missing = None
            loop = asyncio.get_running_loop()
            next_check = loop.time() + _LIVENESS_INTERVAL_SECONDS
            misses = 0
            while True:
                result = await self._held(
                    "job poll",
                    lambda: self._client.get_background_job(
                        self.info.id, job, timeout=_OUTPUT_DEADLINE_SECONDS
                    ),
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
                elif loop.time() >= next_check:
                    next_check = loop.time() + _LIVENESS_INTERVAL_SECONDS
                    alive = await self._job_alive(job, runtime.user)
                    misses = 0 if alive is not False else misses + 1
                    if misses >= _LIVENESS_MISSES:
                        raise _JobDied(
                            f"job {job.job_id} died without an exit status: nothing "
                            "holds its output open and no exit file was written (a "
                            "command in the box likely killed the job's wrapper)"
                        )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 3)
        except _JobDied as e:
            raise SandboxError(f"prime exec failed: {e}", kind="exec") from None
        except (
            Exception
        ) as e:  # a sandbox/API failure is one rollout's problem, not the eval's
            raise SandboxError(
                f"prime exec failed: {e}",
                kind=self._kind(e, "exec"),
            ) from e
        if missing is not None:
            raise SandboxError(f"prime exec output unavailable: {missing}", kind="io")
        if result.exit_code is None:
            raise SandboxError("prime exec completed without an exit code", kind="exec")
        return ProgramResult(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def _job_alive(self, job: Any, user: str | None) -> bool | None:
        """Whether a job without an exit file still runs: some process holds its output
        open, or the exit file has appeared since. None when the check itself failed.
        Runs as the job's user, who can read the job's own file descriptors."""
        log = shlex.quote(job.stdout_log_file)
        command = (
            f"if [ -e {shlex.quote(job.exit_file)} ] || "
            f"ls -l /proc/[0-9]*/fd 2>/dev/null | grep -qF -- ' -> '{log}; "
            "then echo alive; else echo dead; fi"
        )
        try:
            result = await self._client.execute_command(
                self.info.id, command, timeout=60, user=user
            )
        except Exception as e:  # noqa: BLE001 - an unknown liveness is not a death
            logger.debug("prime: liveness check of job %s failed: %s", job.job_id, e)
            return None
        verdict = (result.stdout or "").strip()
        return {"alive": True, "dead": False}.get(verdict)

    async def _open_process(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        try:
            process = await self._client.open_process(
                self.info.id,
                shlex.join(argv),
                working_dir=self.config.workdir,
                env=runtime.process_env(env),
                user=runtime.user,
            )
        except Exception as e:
            raise SandboxError(
                f"prime live process failed to start: {e}",
                kind=self._kind(e, "exec"),
            ) from e
        return PrimeProcess(process)

    @property
    def restored(self) -> bool:
        # the same test start() uses to pick the checkpoint over the image
        return bool(self.config.checkpoint)

    async def checkpoint(self) -> str:
        try:
            # Shielded through the id capture, as in start(): a cancel that aborts the
            # POST mid-flight would leave a durable checkpoint nobody has the id of.
            requested = await run_shielded(self._client.checkpoint(self.info.id))
        except Exception as e:
            raise SandboxError(
                f"prime checkpoint failed: {e}", kind=self._kind(e)
            ) from e
        try:
            durable = await self._client.wait_for_checkpoint(
                requested.id, timeout_seconds=_CHECKPOINT_DEADLINE_SECONDS
            )
        except BaseException as e:
            # Never handed out (failed or cancelled): delete it rather than leak it.
            self._discard_checkpoint(requested.id)
            if isinstance(e, Exception):
                raise SandboxError(
                    f"prime checkpoint failed: {e}", kind=self._kind(e)
                ) from e
            raise
        return durable.id

    def _discard_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint no caller got the id of, in the background: a delete
        waits out the platform's 409 while the checkpoint is still being created."""

        async def discard() -> None:
            try:
                # On a client of its own: a teardown meanwhile closes the shared one.
                await self.delete_checkpoint(checkpoint_id, own_client=True)
            except Exception as e:  # noqa: BLE001 - best effort; the platform expires it
                logger.warning(
                    "prime: failed to delete abandoned checkpoint %s: %s",
                    checkpoint_id,
                    e,
                )

        task = asyncio.get_running_loop().create_task(discard())
        _discards.add(task)
        task.add_done_callback(_discards.discard)

    async def delete_checkpoint(
        self, checkpoint_id: str, *, own_client: bool = False
    ) -> None:
        """`DELETE /sandbox/checkpoints/{id}`: an empty body or a 404 is success; a 409
        (the checkpoint is in use or still being created) is retried with backoff for up
        to `_CHECKPOINT_DELETE_DEADLINE_SECONDS`. Uses the live client when this runtime
        holds one, else (or with `own_client`) a short-lived client of its own."""
        async with contextlib.AsyncExitStack() as stack:
            client = None if own_client else self._client
            if client is None:
                from prime_sandboxes import AsyncSandboxClient

                client = await stack.enter_async_context(AsyncSandboxClient())
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _CHECKPOINT_DELETE_DEADLINE_SECONDS
            delay = 1.0
            while True:
                try:
                    await _repeat(
                        lambda: client.client.request(
                            "DELETE", f"/sandbox/checkpoints/{checkpoint_id}"
                        )
                    )
                    return
                except json.JSONDecodeError:
                    return  # the platform answers a delete with an empty body
                except Exception as e:
                    status = http_status(e)
                    if status == 404:
                        return
                    if status != 409 or loop.time() + delay > deadline:
                        raise SandboxError(
                            f"prime checkpoint delete failed: {e}",
                            kind=self._kind(e),
                        ) from e
                    logger.info(
                        "prime: checkpoint %s not deletable yet, retrying in %.0fs: %s",
                        checkpoint_id,
                        delay,
                        e,
                    )
                await asyncio.sleep(delay)
                delay = min(delay * 2, _CHECKPOINT_DELETE_MAX_DELAY_SECONDS)

    async def expose(self, port: int) -> str:
        raise SandboxError(
            "Prime VM sandboxes do not support port exposure; colocate the service "
            "with its consumer, or use a docker or modal service runtime"
        )

    async def _run_background(
        self, runtime: Runtime, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        command = f"exec {shlex.join(argv)} > {shlex.quote(log)} 2>&1"
        try:
            await self._client.start_background_job(
                self.info.id,
                command,
                working_dir=self.config.workdir,
                env=runtime.process_env(env),
                user=runtime.user,
            )
        except Exception as e:
            raise SandboxError(
                f"prime background launch failed: {e}",
                kind=self._kind(e, "exec"),
            ) from e

    async def _read(
        self, runtime: Runtime, path: str, max_bytes: int | None = None
    ) -> bytes:
        # The file API acts as the default user; a user view reads through exec.
        if max_bytes is not None or runtime.user is not None:
            reader = "cat" if max_bytes is None else f"head -c {max_bytes}"
            try:
                # Stream binary output: execute_command buffers base64 text for the
                # entire file. Bound the source read and the host buffer independently.
                process = await self._client.open_process(
                    self.info.id,
                    f"{reader} -- {shlex.quote(path)}",
                    working_dir=self.config.workdir,
                    env=runtime.process_env({}),
                    user=runtime.user,
                )
                async with contextlib.aclosing(process):
                    with io.BytesIO() as data:
                        async for chunk in process.stdout:
                            if (
                                max_bytes is not None
                                and data.tell() + len(chunk) > max_bytes
                            ):
                                raise SandboxError(
                                    "read stream exceeded its byte limit"
                                )
                            data.write(chunk)
                        stderr = b""
                        async for chunk in process.stderr:
                            stderr = (stderr + chunk)[-500:]
                        if await process.wait():
                            detail = stderr.decode(errors="replace").strip()
                            raise SandboxError(
                                detail, kind=missing_file_kind(detail, "io")
                            )
                        return data.getvalue()
            except Exception as exc:
                raise SandboxError(
                    f"read {path!r}: {exc}", kind=self._kind(exc, "io")
                ) from exc
        # Avoid background-job log limits and base64 overhead by downloading binary data directly.
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        try:
            size = await self._file_size(target)
            if size <= _TRANSFER_PART_BYTES:
                return await self._download(target)
            return await self._read_parts(target, size)
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}", kind=self._kind(e, "io")) from e

    async def _write(self, runtime: Runtime, path: str, data: bytes) -> None:
        # The gateway creates missing parents and uploads binary data without command-line
        # limits. Resolve relative paths here because uploads do not use this runtime's workdir.
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        if runtime.user is None:
            try:
                if len(data) <= _TRANSFER_PART_BYTES:
                    await self._upload(target, data)
                else:
                    await self._write_parts(target, data)
            except Exception as e:
                raise SandboxError(
                    f"write {path!r}: {e}", kind=self._kind(e, "io")
                ) from e
            return
        # Uploads act as the default user: stage the bytes, then copy them in as the user.
        root = runtime.with_user(None)
        staged = f"/tmp/vf-write-{uuid.uuid4().hex}"
        try:
            await root.write(staged, data)
            await root.run(["chmod", "644", staged], {})
            copy = 'mkdir -p "$(dirname "$1")" && cat -- "$2" > "$1"'
            result = await runtime.run(["sh", "-c", copy, "sh", target, staged], {})
        finally:
            await root.run(["rm", "-f", staged], {})
        if result.exit_code != 0:
            raise SandboxError(
                f"write {path!r}: {result.stderr.strip()[-500:]}", kind="io"
            )

    async def _file_size(self, target: str) -> int:
        """`target`'s size in bytes, which picks one download or several. A missing
        file fails here, before any transfer."""
        result = await self._held(
            "size",
            lambda: self._client.execute_command(
                self.info.id, f"wc -c < {shlex.quote(target)}", timeout=60
            ),
            **_TRANSFER_RETRIES,
        )
        if result.exit_code:
            detail = (result.stderr or "").strip()[-500:]
            raise SandboxError(detail, kind=missing_file_kind(detail, "io"))
        return int(result.stdout.strip())

    async def _upload(self, target: str, data: bytes) -> None:
        await self._held(
            "upload",
            lambda: self._client.upload_bytes(
                self.info.id,
                target,
                data,
                filename=PurePosixPath(target).name,
                timeout=self.config.transfer_timeout,
            ),
            **_TRANSFER_RETRIES,
        )

    async def _download(self, target: str) -> bytes:
        # The temporary file is removed on every exit, and its byte read stays off the event loop.
        with tempfile.TemporaryDirectory() as directory:
            download = Path(directory) / "download"
            await self._held(
                "download",
                lambda: self._client.download_file(
                    self.info.id,
                    target,
                    str(download),
                    timeout=self.config.transfer_timeout,
                ),
                **_TRANSFER_RETRIES,
            )
            return await asyncio.to_thread(download.read_bytes)

    async def _write_parts(self, target: str, data: bytes) -> None:
        """Upload `data` in `_TRANSFER_PART_BYTES` parts beside `target`, then join them
        in the box and move the result onto `target` once its size checks out, so a
        failed write never leaves a truncated `target`. Each part is removed as soon as
        it is appended, so the join needs one part's room beyond the file's own. The
        parts are removed on every exit."""
        stem = f"{target}.vf-part-{uuid.uuid4().hex[:12]}"
        size = _TRANSFER_PART_BYTES
        parts = [f"{stem}.{n:05d}" for n in range(math.ceil(len(data) / size))]
        root = self.bind()  # uploads land as the default user
        try:
            for n, part in enumerate(parts):
                await self._upload(part, data[n * size : (n + 1) * size])
            joined = await root.run(
                [
                    "sh",
                    "-c",
                    (
                        'out=$1 size=$2 tmp=$3; shift 3; : > "$tmp" || exit 1;'
                        ' for p; do cat -- "$p" >> "$tmp" && rm -f -- "$p" || exit 1; done;'
                        ' [ "$(wc -c < "$tmp")" -eq "$size" ] && mv -f -- "$tmp" "$out"'
                    ),
                    "sh",
                    target,
                    str(len(data)),
                    f"{stem}.tmp",
                    *parts,
                ],
                {},
            )
            if joined.exit_code:
                raise SandboxError(
                    f"joining {len(parts)} uploaded parts failed "
                    f"({joined.exit_code}): {joined.stderr.strip()[-500:]}",
                    kind="io",
                )
        finally:
            with contextlib.suppress(Exception):
                await root.run(["sh", "-c", 'rm -f -- "$1".*', "sh", stem], {})

    async def _read_parts(self, target: str, size: int) -> bytes:
        """Download `target`'s first `size` bytes as `_TRANSFER_PART_BYTES` parts cut
        into `/tmp` in the box one at a time (the target's directory may be read-only),
        so the box holds at most one extra part (a failed read removes it). A file that
        grows while it is read (a log) gives its first `size` bytes; one that shrinks
        fails the read."""
        stem = f"/tmp/vf-read-{uuid.uuid4().hex[:12]}"
        root = self.bind()
        chunks: list[bytes] = []
        try:
            for n, offset in enumerate(range(0, size, _TRANSFER_PART_BYTES)):
                part = f"{stem}.{n:05d}"
                cut = await root.run(
                    [
                        "sh",
                        "-c",
                        'tail -c "+$2" -- "$1" | head -c "$3" > "$4"',
                        "sh",
                        target,
                        str(offset + 1),
                        str(min(_TRANSFER_PART_BYTES, size - offset)),
                        part,
                    ],
                    {},
                )
                if cut.exit_code:
                    raise SandboxError(
                        f"cutting part {n} failed ({cut.exit_code}): "
                        f"{cut.stderr.strip()[-500:]}",
                        kind="io",
                    )
                chunks.append(await self._download(part))
                await root.run(["rm", "-f", "--", part], {})
        except BaseException:
            with contextlib.suppress(Exception):
                await root.run(["sh", "-c", 'rm -f -- "$1".*', "sh", stem], {})
            raise
        data = b"".join(chunks)
        if len(data) != size:
            raise SandboxError(
                f"{target} changed while it was read ({size} bytes became {len(data)})",
                kind="io",
            )
        return data

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
                box = self.info.id
                await self._held("delete", lambda: client.delete(box))
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
