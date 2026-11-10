"""Remote Prime VM sandbox runtime."""

from __future__ import annotations

import asyncio
import contextlib
import io
import math
import shlex
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar
from urllib.parse import urlsplit

from prime_sandboxes.models import validate_egress_lists

from verifiers.v1.configs.runtime import NetworkPolicyConfig, PrimeConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    Program,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
    Snapshot,
    parse_gpu,
)
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.artifacts import validate_runtime_mounts
from verifiers.v1.utils.format import format_time
from verifiers.v1.utils.logging import get_logger
from verifiers.v1.utils.prime import ensure_prime_auth
from verifiers.v1.utils.scope import run_scope

logger = get_logger(__name__)

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
_CHECKPOINT_DURABLE_SECONDS = 300
_PAUSE_SETTLE_SECONDS = 120
"""How long a pause or a resume gets to take effect; they take seconds."""
_DEAD_STATUSES = frozenset({"ERROR", "TERMINATED", "TIMEOUT"})
_STATUS_POLL_SECONDS = 0.3
_FORK_ATTEMPTS = 3
"""Creates of a box from a memory snapshot before its restore counts as failed."""
_RESUME_RETRY_SECONDS = 600
"""How long a checkpoint gets to become restorable (`DURABLE`): the platform uploads it
after the request returns, and only a durable one restores."""
_PUBLIC_DOMAIN = "sandbox-v2-prod.pinfra.io"
"""Domain of the public URLs `https://<port>-<sandbox-id>.<domain>` that a LAN network's
`internet` rule publishes ports at."""
_EXPOSE_TAG = "vf-exposed"


BASE_LABELS: list[str] = []


class _StatusPoller:
    """Batched status lookups for the boxes that wait for a pause or a resume to take
    effect: one request serves up to 100 boxes, so a save of every rollout of a run
    costs a few requests per poll instead of one per box."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.waiting: dict[str, asyncio.Future] = {}
        self.task: asyncio.Task | None = None

    async def next(self, sandbox_id: str) -> Any:
        """The box's status at the next poll (one waiter per box)."""
        future = asyncio.get_running_loop().create_future()
        self.waiting[sandbox_id] = future
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())
        return await future

    async def _run(self) -> None:
        while self.waiting:
            await asyncio.sleep(_STATUS_POLL_SECONDS)
            waiting, self.waiting = self.waiting, {}
            ids = list(waiting)
            await asyncio.gather(
                *(
                    self._poll({i: waiting[i] for i in ids[start : start + 100]})
                    for start in range(0, len(ids), 100)
                )
            )

    async def _poll(self, waiting: dict[str, asyncio.Future]) -> None:
        try:
            response = await self.client.get_sandbox_statuses(list(waiting))
        except Exception as error:  # noqa: BLE001 - each waiter reports it
            results: dict[str, Any] = dict.fromkeys(waiting, error)
        else:
            results = {status.sandbox_id: status for status in response.statuses}
            results.update(
                (item.sandbox_id, SandboxError(f"{item.code}: {item.message}"))
                for item in response.errors
            )
        for sandbox_id, future in waiting.items():
            result = results.get(sandbox_id, SandboxError("no status returned"))
            if future.done():
                continue
            if isinstance(result, Exception):
                future.set_exception(result)
            else:
                future.set_result(result)


@dataclass
class _SharedClient:
    client: Any
    leases: int = 0
    poller: _StatusPoller | None = None


# One `AsyncSandboxClient` per event loop, leased by every live runtime on it. The SDK
# coalesces concurrent status polls per client into batched requests (up to 100 ids
# each), so sharing a client turns N runtimes' creation/job polls into a few batch
# calls. Keyed by loop (not plain process-global) so a client's tasks stay on the loop
# that created it.
_shared_clients: dict[asyncio.AbstractEventLoop, _SharedClient] = {}


def _lease_client() -> Any:
    from prime_sandboxes import AsyncSandboxClient

    loop = asyncio.get_running_loop()
    shared = _shared_clients.get(loop)
    if shared is None:
        shared = _shared_clients[loop] = _SharedClient(
            AsyncSandboxClient(background_job_output_concurrency=_OUTPUT_READS)
        )
    shared.leases += 1
    return shared.client


async def _release_client(client: Any) -> None:
    loop = asyncio.get_running_loop()
    shared = _shared_clients[loop]
    shared.leases -= 1
    if not shared.leases:  # the last lease on this loop closes the client
        del _shared_clients[loop]
        with contextlib.suppress(Exception):
            await client.aclose()


@asynccontextmanager
async def leased_client() -> AsyncIterator[Any]:
    """This event loop's shared client, for calls that belong to no one sandbox, such as
    LAN networks."""
    client = _lease_client()
    try:
        yield client
    finally:
        await _release_client(client)


def set_base_sandbox_labels(labels: list[str]) -> None:
    """Set process-wide base labels attached to every Prime sandbox, extended by each
    runtime's ``PrimeConfig.labels``. Call it in the process that creates the sandboxes
    (a host that spawns worker processes sets it in each one) — e.g. a trainer stamps its run
    name so every sandbox of a run is findable on the platform."""
    global BASE_LABELS
    BASE_LABELS = list(labels)


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
    SUPPORTS_TASK_IMAGES: ClassVar[bool] = True
    SUPPORTS_USER_SWITCHING: ClassVar[bool] = True
    SUPPORTS_COMPOSE: ClassVar[bool] = True
    SUPPORTS_PORT_EXPOSURE: ClassVar[bool] = True
    SUPPORTS_MEMORY_SNAPSHOTS: ClassVar[bool] = True

    def __init__(self, config: PrimeConfig, name: str | None = None) -> None:
        ensure_prime_auth()
        super().__init__(name)
        self.user: str | None = None
        self._workdir_override = config.workdir
        self.config = config.model_copy(update={"workdir": config.workdir or "/app"})
        self.info = PrimeRuntimeInfo(**self.config.model_dump())
        self._client = None
        # A LAN network (id, tags) to join as the sandbox boots: prime-sandboxes joins it
        # once it is running, within `wait_for_creation` on the creating client.
        self.lan: tuple[str, list[str]] | None = None
        # Exposure state: a LAN network created on the first `expose`, separate from
        # `lan`, and the ports its `internet` rule publishes.
        self._expose_network_id: str | None = None
        self._exposed_ports: list[int] = []
        self._expose_lock = asyncio.Lock()
        self._awake = asyncio.Event()
        """Cleared while a memory snapshot stops the box: the platform refuses every
        call into a stopped box, so calls wait for it."""
        self._awake.set()

    @property
    def published_port(self) -> int:
        # Any port can be exposed, but the public URL reaches only servers bound to
        # 0.0.0.0, which `launch` arranges for a fixed published port.
        return SERVICE_PORT

    def bind(self, *, env: Mapping[str, str] | None = None) -> Runtime:
        return super().bind(env=env).with_user(self.user)

    async def start(self) -> None:
        from prime_sandboxes import (
            APIError,
            CreateSandboxRequest,
            SandboxNotRunningError,
        )

        started = time.perf_counter()
        self._client = _lease_client()
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
        if self.config.mounts:
            options["volumes"] = [
                {"mount_path": target, **mount.model_dump()}
                for target, mount in self.config.mounts.items()
            ]
        memory = bool(self.config.checkpoint and self.config.restore_memory)
        if memory:
            # The box continues the snapshot's own processes: the platform fixes its
            # shape, environment and volumes to the paused box's and refuses others.
            source = {"checkpoint_id": self.config.checkpoint, "restore_memory": True}
            options = {"timeout_minutes": -1, "idle_timeout_minutes": idle_minutes}
        elif self.config.checkpoint:
            source = {"checkpoint_id": self.config.checkpoint}
        else:
            source = {
                "docker_image": self.config.image,
                "disk_size_gb": self.config.disk,
            }
        if self.lan is not None:
            options["lan_network_id"], options["lan_tags"] = self.lan
        scope = run_scope()
        labels = [*BASE_LABELS, *self.config.labels, scope]
        try:
            attempts = _FORK_ATTEMPTS if memory else 1
            for attempt in range(attempts):
                async with (
                    creation_limiter(
                        (self.config.creates_per_min or 0) / 60,
                        "prime-sandbox",
                        scope,
                    )
                    or contextlib.nullcontext()
                ):
                    for name in dict.fromkeys(
                        m.name for m in self.config.mounts.values()
                    ):
                        try:
                            await self._client.volumes.create(name)
                        except APIError as error:
                            # Create returns 409 when this scoped name already exists,
                            # including when another runtime creates it concurrently.
                            if not str(error).startswith("HTTP 409:"):
                                raise

                    # Shielded through the id capture: a cancel that aborts the POST
                    # mid-flight leaves the platform creating a sandbox this side
                    # never learned the id of — teardown() then cannot delete it
                    async def create_and_capture_id():
                        sandbox = await self._client.create(
                            CreateSandboxRequest(
                                name=self.name,
                                labels=list(dict.fromkeys(labels)),
                                **({} if memory else {"environment_vars": self.env}),
                                **source,
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
                        "Building image %s on the platform for prime runtime %s - the "
                        "first use of an image can take ~10 minutes, later runs start "
                        "in seconds",
                        self.config.image,
                        self.info.id,
                    )
                try:
                    await self._client.wait_for_creation(self.info.id, max_attempts=900)
                    break
                except SandboxNotRunningError as error:
                    # A box that continues an old memory snapshot also continues its
                    # idle clock: the platform can end it as idle before it runs.
                    if attempt + 1 == attempts:
                        raise
                    logger.warning(
                        "Prime runtime %s ended before it ran from memory snapshot "
                        "%s; creating it again (attempt %d/%d) - %s",
                        self.info.id,
                        self.config.checkpoint,
                        attempt + 2,
                        attempts,
                        error,
                    )
            logger.debug(
                "Started prime runtime %s in %s image=%s%s",
                self.info.id,
                format_time(time.perf_counter() - started),
                self.config.image,
                f" checkpoint={self.config.checkpoint}"
                if self.config.checkpoint
                else "",
            )
            if memory:
                # The paused box was already prepared; its processes run again.
                return
            await self._client.execute_command(
                self.info.id, f"mkdir -p {shlex.quote(self.config.workdir)}"
            )
            # Imported image filesystems may not map the VM's generated hostname.
            await self._client.execute_command(
                self.info.id,
                'getent hosts "$(cat /proc/sys/kernel/hostname)" >/dev/null || '
                'printf "127.0.1.1 %s\\n" "$(cat /proc/sys/kernel/hostname)" >> /etc/hosts',
            )
            await validate_runtime_mounts(self.bind(), [])
        except (
            Exception
        ) as e:  # provisioning failure is one rollout's problem, not the eval's
            raise SandboxError(f"prime sandbox provisioning failed: {e}") from e

    async def snapshot(self) -> Snapshot:
        # A pause is the platform's memory snapshot: it stops the box, and the resume
        # continues it. The platform uploads the snapshot after the pause. Until it is
        # durable the snapshot is stray: the platform refuses to delete it while this
        # box lives, so teardown frees it after.
        async def create() -> Snapshot:
            # Shielded as one: a box left paused never runs its rollout again.
            self._awake.clear()
            try:
                paused = await self._client.pause(self.info.id)
                snapshot = Snapshot(id=paused.checkpoint_id)
                self.stray_snapshots.append(snapshot)
                try:
                    await self._settle("PAUSED", "PAUSE_FAILED")
                except BaseException:
                    self.stray_snapshots.remove(snapshot)
                    raise
                await self._resume()
            finally:
                self._awake.set()
            return snapshot

        try:
            return await run_shielded(create())
        except Exception as e:
            raise SandboxError(f"prime checkpoint failed: {e}") from e

    async def _resume(self) -> None:
        """Resume the paused box. The pause freed its memory on its node, which another
        box may have taken: the platform then refuses the resume until there is room
        again, so ask until it runs or `_RESUME_RETRY_SECONDS` passed."""
        deadline = time.monotonic() + _RESUME_RETRY_SECONDS
        delay = 1.0
        while True:
            try:
                # An earlier request may have taken effect after it was given up on.
                if (await self._client.get(self.info.id)).status == "RUNNING":
                    return
                await self._client.resume(self.info.id)
                await self._settle("RUNNING", "RESUME_FAILED")
                return
            except Exception as e:
                if time.monotonic() > deadline:
                    raise SandboxError(
                        f"prime sandbox {self.info.id} did not resume after its "
                        f"memory snapshot: {e}"
                    ) from e
                logger.warning(
                    "prime: sandbox %s did not resume, asking again in %.0fs: %s",
                    self.info.id,
                    delay,
                    e,
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 15)

    def _stopped_by_snapshot(self, error: Exception) -> bool:
        """Whether a call failed because a memory snapshot had stopped the box: the
        call is safe to repeat once the box runs again."""
        return not self._awake.is_set() or "sandbox is paused" in str(error).lower()

    async def _settle(self, status: str, failure: str) -> None:
        """Wait until the platform reports `status` for this box after a pause or a
        resume (both return once accepted)."""
        deadline = time.monotonic() + _PAUSE_SETTLE_SECONDS
        shared = _shared_clients[asyncio.get_running_loop()]
        if shared.poller is None:
            shared.poller = _StatusPoller(shared.client)
        while True:
            box = await shared.poller.next(self.info.id)
            if box.status == status:
                return
            if box.error_type == failure or box.status in _DEAD_STATUSES:
                raise SandboxError(
                    f"prime sandbox {self.info.id} did not reach {status}: "
                    f"{box.status} {box.error_type or ''} {box.error_message or ''}"
                )
            if time.monotonic() > deadline:
                raise SandboxError(
                    f"prime sandbox {self.info.id} did not reach {status} within "
                    f"{_PAUSE_SETTLE_SECONDS}s"
                )

    async def wait_snapshot(self, snapshot: Snapshot) -> int | None:
        started = time.monotonic()
        try:
            durable = await self._client.wait_for_checkpoint(
                snapshot.id, timeout_seconds=_CHECKPOINT_DURABLE_SECONDS
            )
        except Exception as e:
            raise SandboxError(f"prime checkpoint {snapshot.id}: {e}") from e
        self.stray_snapshots.remove(snapshot)
        logger.debug(
            "prime: checkpoint %s durable after %.1fs on sandbox %s",
            snapshot.id,
            time.monotonic() - started,
            self.info.id,
        )
        return durable.stored_bytes

    @classmethod
    async def discard(cls, snapshots: list[Snapshot]) -> list[str]:
        # The platform refuses only the newest checkpoint of a running box (the parent
        # of its live state); older ones go while it runs, their data kept as long as a
        # newer one needs it. Usually called after teardown released `_client`.
        deleted = []
        async with leased_client() as client:
            for snapshot in snapshots:
                try:
                    await client.delete_checkpoint(snapshot.id)
                except Exception as e:  # noqa: BLE001 - provider cleanup is best-effort
                    if "HTTP 404" in str(e):
                        # Already gone: an owner that asks again must not ask forever.
                        deleted.append(snapshot.id)
                        continue
                    logger.warning(
                        "prime: failed to delete checkpoint %s: %s", snapshot.id, e
                    )
                else:
                    deleted.append(snapshot.id)
        return deleted

    @classmethod
    def restore_config(cls, config: PrimeConfig, snapshot: Snapshot) -> PrimeConfig:
        return config.model_copy(
            update={"checkpoint": snapshot.id, "restore_memory": True}
        )

    async def prepare_execution(
        self, routes: list[str] | None, network: NetworkPolicyConfig | None = None
    ) -> None:
        """Apply the host policy, or `network`, after setup and wait until the platform
        enforces it."""
        if not self.network_restricted:
            return
        config = network or self.config
        try:
            if routes is None:
                policy = {"allow": ["*"]}
            else:
                hosts = list(
                    dict.fromkeys(
                        h for h in (urlsplit(route).hostname for route in routes) if h
                    )
                )
                if config.allow == ["*"]:
                    policy = {"deny": config.block}
                else:
                    entries = list(dict.fromkeys([*hosts, *config.allow]))
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
        logger.debug(
            "Applied egress policy to prime runtime %s allow=%s block=%s",
            self.info.id,
            policy.get("allow"),
            policy.get("deny"),
        )

    async def run(
        self,
        runtime: Runtime,
        argv: list[str],
        env: dict[str, str],
        *,
        program: Program | None = None,
    ) -> ProgramResult:
        await self._awake.wait()
        try:
            # Poll directly so rollout cancellation owns the execution timeout.
            job = await self._client.start_background_job(
                self.info.id,
                shlex.join(argv),
                working_dir=self.config.workdir,
                env=runtime.process_env(env),
                user=runtime.user,
            )
        except (
            Exception
        ) as e:  # a sandbox/API failure is one rollout's problem, not the eval's
            raise SandboxError(f"prime exec failed: {e}") from e
        if program is not None:
            program.job = job.model_dump()
        try:
            return await self._wait_job(job)
        finally:
            if program is not None:
                program.job = None

    async def attach(self, runtime: Runtime, job: dict[str, Any]) -> ProgramResult:
        from prime_sandboxes import BackgroundJob

        # The job's files and process came with the snapshot; only the box is new.
        return await self._wait_job(
            BackgroundJob(**{**job, "sandbox_id": self.info.id})
        )

    async def _wait_job(self, job: Any) -> ProgramResult:
        try:
            delay = 0.1
            output_retries = 0
            missing = None
            while True:
                await self._awake.wait()
                try:
                    result = await self._client.get_background_job(
                        self.info.id, job, timeout=_OUTPUT_DEADLINE_SECONDS
                    )
                except Exception as e:
                    if not self._stopped_by_snapshot(e):
                        raise
                    continue
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

    async def exec(
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
            raise SandboxError(f"prime streaming process failed to start: {e}") from e
        return PrimeProcess(process)

    async def expose(self, port: int) -> str:
        """Publish `port` through a LAN network whose `internet` rule exposes it on this
        sandbox. The URL has no Prime authentication."""
        from prime_sandboxes import LanRule

        async with self._expose_lock:
            if port not in self._exposed_ports:
                ports = [*self._exposed_ports, port]
                rules = [
                    LanRule(from_="internet", to=f"tag:{_EXPOSE_TAG}", ports=ports)
                ]
                try:
                    if self._expose_network_id is None:
                        # Shielded through the id capture, as in `start()`: a cancel
                        # mid-POST would leave a network teardown cannot delete.
                        async def create_and_capture_id() -> None:
                            network = await self._client.create_lan_network(
                                f"{self.name}-exposed", rules=rules
                            )
                            self._expose_network_id = network.id

                        await run_shielded(create_and_capture_id())
                    else:
                        await self._client.set_lan_rules(self._expose_network_id, rules)
                    # Joins (an idempotent PUT, so a retry re-adds a failed join) and
                    # waits until the platform has pushed the membership and rules.
                    await self._client._join_lan(
                        self._expose_network_id, self.info.id, [_EXPOSE_TAG]
                    )
                except Exception as e:
                    raise SandboxError(f"prime expose port {port} failed: {e}") from e
                self._exposed_ports = ports
        return f"https://{port}-{self.info.id}.{_PUBLIC_DOMAIN}"

    async def run_background(
        self, runtime: Runtime, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        await self._awake.wait()
        # A server-owned job runs without a client stream for its whole life.
        command = f"exec {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null"
        try:
            await self._client.start_background_job(
                self.info.id,
                command,
                working_dir=self.config.workdir,
                env=runtime.process_env(env),
                user=runtime.user,
            )
        except Exception as e:
            raise SandboxError(f"prime background launch failed: {e}") from e

    async def read(
        self, runtime: Runtime, path: str, max_bytes: int | None = None
    ) -> bytes:
        await self._awake.wait()
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

    async def write(self, runtime: Runtime, path: str, data: bytes) -> None:
        await self._awake.wait()
        # The gateway creates missing parents and uploads binary data without command-line
        # limits. Resolve relative paths here because uploads do not use this runtime's workdir.
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        if runtime.user is None:
            try:
                await self._client.upload_bytes(
                    self.info.id, target, data, filename=PurePosixPath(target).name
                )
            except Exception as e:
                raise SandboxError(f"write {path!r}: {e}") from e
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
            raise SandboxError(f"write {path!r}: {result.stderr.strip()[-500:]}")

    def cleanup(self) -> None:
        # Synchronous atexit backstop (the async client can't run once the loop is gone): delete
        # the sandbox via the sync client, so the costly resource isn't left to its max-lifetime.
        # Idempotent — the async `stop` deletes it on the normal path, a second delete 404s.
        if self.info.id is not None:
            from prime_sandboxes import SandboxClient
            from prime_sandboxes.core import APIClient

            client = SandboxClient(APIClient())
            with contextlib.suppress(Exception):
                client.delete(self.info.id)
            if self._expose_network_id is not None:
                with contextlib.suppress(Exception):
                    client.delete_lan_network(self._expose_network_id)

    async def teardown(self) -> None:
        # Best-effort, idempotent teardown: delete the sandbox (the costly resource). Runs via
        # `stop`, shielded from cancellation, so it fires on success, error, and Ctrl-C.
        client, self._client = self._client, None  # `_client` is the idempotency guard
        if client is None:
            return
        try:
            try:
                if self.info.id is not None:  # keep info.id available after teardown
                    await client.delete(self.info.id)
            except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
                logger.warning(
                    "Failed to delete prime runtime %s - %s: %s",
                    self.info.id,
                    type(e).__name__,
                    e,
                )
            if self._expose_network_id is not None:
                try:
                    await client.delete_lan_network(self._expose_network_id)
                    self._expose_network_id = None
                except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
                    logger.warning(
                        "Failed to delete LAN network %s - %s: %s",
                        self._expose_network_id,
                        type(e).__name__,
                        e,
                    )
        finally:
            await _release_client(client)
