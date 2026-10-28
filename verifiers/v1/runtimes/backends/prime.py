"""Remote Prime VM sandbox runtime."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import math
import shlex
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar
from urllib.parse import urlsplit

from prime_sandboxes.models import validate_egress_lists

from verifiers.v1.configs.runtime import NetworkPolicyConfig, PrimeConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
    parse_gpu,
)
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.runtimes.runtime import Runtime
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
_OPERATION_HEARTBEAT_SECONDS = 30
"""Interval between metadata-only records for a stalled Prime operation."""


@dataclass
class PrimeOperationContext:
    """Metadata for one Prime operation, also exposed to attribution analysis."""

    rollout_id: str
    operation_id: str
    operation: str
    phase: str
    started_at: float = field(default_factory=time.monotonic, repr=False)
    sdk_wait: float = 0.0
    sdk_started_at: float | None = field(default=None, repr=False)
    poll_count: int = 0
    output_retries: int = 0
    last_stream_activity: float | None = field(default=None, repr=False)
    terminal_error_type: str | None = None

    async def sdk(self, awaitable: Awaitable[Any], *, phase: str | None = None) -> Any:
        """Await an SDK/stream operation and account only for time inside that await."""
        if phase is not None:
            self.phase = phase
        self.sdk_started_at = time.monotonic()
        try:
            return await awaitable
        finally:
            self.sdk_wait += time.monotonic() - self.sdk_started_at
            self.sdk_started_at = None

    def record(self, event: str) -> dict[str, Any]:
        """A payload-free record suitable for structured logs and attribution."""
        now = time.monotonic()
        sdk_wait = self.sdk_wait + (
            now - self.sdk_started_at if self.sdk_started_at is not None else 0.0
        )
        stream_age = (
            max(0.0, now - self.last_stream_activity)
            if self.last_stream_activity is not None
            else None
        )
        return {
            "event": event,
            "rollout_id": self.rollout_id,
            "operation_id": self.operation_id,
            "operation": self.operation,
            "phase": self.phase,
            "elapsed_s": round(max(0.0, now - self.started_at), 3),
            "sdk_wait_s": round(sdk_wait, 3),
            "poll_count": self.poll_count,
            "output_retries": self.output_retries,
            "last_stream_activity_age_s": (
                round(stream_age, 3) if stream_age is not None else None
            ),
            "terminal_error_type": self.terminal_error_type,
        }


_current_operation: ContextVar[PrimeOperationContext | None] = ContextVar(
    "prime_operation", default=None
)


def current_prime_operation() -> PrimeOperationContext | None:
    """Return the active operation's metadata for later attribution analysis."""
    return _current_operation.get()


def _log_operation(context: PrimeOperationContext, event: str) -> None:
    logger.info(
        "prime_operation %s",
        json.dumps(context.record(event), sort_keys=True, separators=(",", ":")),
    )


async def _operation_heartbeat(context: PrimeOperationContext) -> None:
    while True:
        await asyncio.sleep(_OPERATION_HEARTBEAT_SECONDS)
        _log_operation(context, "heartbeat")


@asynccontextmanager
async def _operation(
    rollout_id: str, operation: str, *, stream: bool = False
) -> AsyncIterator[PrimeOperationContext]:
    context = PrimeOperationContext(
        rollout_id=rollout_id,
        operation_id=uuid.uuid4().hex,
        operation=operation,
        phase=operation,
    )
    if stream:
        context.last_stream_activity = context.started_at
    token = _current_operation.set(context)
    _log_operation(context, "start")
    heartbeat = asyncio.create_task(_operation_heartbeat(context))
    try:
        yield context
    except BaseException as exc:
        # The type identifies the terminal path without reproducing error messages,
        # which can contain provider commands or output.
        context.terminal_error_type = type(exc).__name__
        raise
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await heartbeat
        _log_operation(context, "end")
        _current_operation.reset(token)


async def _stream_chunks(
    stream: AsyncIterator[bytes], rollout_id: str, stream_name: str
) -> AsyncIterator[bytes]:
    """Instrument a stream as one operation, never as one record per chunk."""
    async with _operation(rollout_id, f"{stream_name}_stream", stream=True) as context:
        iterator = stream.__aiter__()
        while True:
            try:
                chunk = await context.sdk(anext(iterator), phase=f"{stream_name}_read")
            except StopAsyncIteration:
                return
            context.last_stream_activity = time.monotonic()
            context.phase = f"{stream_name}_yield"
            yield chunk


async def _wait_for_process(process: Any, rollout_id: str) -> int:
    async with _operation(rollout_id, "process_wait") as context:
        return await context.sdk(process.wait())


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
    (env-server workers set it via their setup hook) — e.g. a trainer stamps its run
    name so every sandbox of a run is findable on the platform."""
    global BASE_LABELS
    BASE_LABELS = list(labels)


class PrimeRuntimeInfo(PrimeConfig, BaseRuntimeInfo):
    image_cached: bool | None = None
    """Whether the platform already had the image at create (None until then). False means
    a first-use auto-build ran while this sandbox waited to start."""


class PrimeProcess(RuntimeProcess):
    def __init__(self, process, rollout_id: str) -> None:
        self._process = process
        self._rollout_id = rollout_id
        self.stdout: AsyncIterator[bytes] = _stream_chunks(
            process.stdout, rollout_id, "stdout"
        )
        self.stderr: AsyncIterator[bytes] = _stream_chunks(
            process.stderr, rollout_id, "stderr"
        )

    async def write(self, data: bytes) -> None:
        await self._process.write_stdin(data)

    async def wait(self) -> int:
        return await _wait_for_process(self._process, self._rollout_id)

    async def poll(self) -> int | None:
        return self._process.returncode

    async def terminate(self) -> None:
        await self._process.terminate()

    async def kill(self) -> None:
        await self._process.kill()


class PrimeBackend(RuntimeBackend):
    SUPPORTS_TASK_IMAGES: ClassVar[bool] = True
    SUPPORTS_COMPOSE: ClassVar[bool] = True
    SUPPORTS_USER_SWITCHING: ClassVar[bool] = True

    def __init__(self, config: PrimeConfig, name: str | None = None) -> None:
        ensure_prime_auth()
        super().__init__(name)
        self.config = config.model_copy(update={"workdir": config.workdir or "/app"})
        self.info = PrimeRuntimeInfo(**self.config.model_dump())
        self._client = None
        # A LAN network (id, tags) to join as the sandbox boots: prime-sandboxes joins it
        # once it is running, within `wait_for_creation` on the creating client.
        self.lan: tuple[str, list[str]] | None = None

    async def start(self) -> None:
        from prime_sandboxes import CreateSandboxRequest

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
        if self.config.checkpoint:
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
                            **source,
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
        logger.info(
            "prime: egress policy applied on sandbox %s (allow=%s block=%s)",
            self.info.id,
            policy.get("allow"),
            policy.get("deny"),
        )

    async def _start_background_job(
        self, runtime: Runtime, command: str, env: dict[str, str]
    ) -> Any:
        async with _operation(self.name, "start_background_job") as context:
            return await context.sdk(
                self._client.start_background_job(
                    self.info.id,
                    command,
                    working_dir=self.config.workdir,
                    env=runtime.process_env(env),
                    user=runtime.user,
                )
            )

    async def _open_process(
        self, runtime: Runtime, command: str, env: dict[str, str]
    ) -> Any:
        async with _operation(self.name, "open_process") as context:
            return await context.sdk(
                self._client.open_process(
                    self.info.id,
                    command,
                    working_dir=self.config.workdir,
                    env=runtime.process_env(env),
                    user=runtime.user,
                )
            )

    async def run(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        try:
            # Poll directly so rollout cancellation owns the execution timeout.
            job = await self._start_background_job(runtime, shlex.join(argv), env)
            delay = 0.1
            missing = None
            async with _operation(self.name, "poll_output") as context:
                while True:
                    context.poll_count += 1
                    result = await context.sdk(
                        self._client.get_background_job(
                            self.info.id, job, timeout=_OUTPUT_DEADLINE_SECONDS
                        ),
                        phase=("output_read" if context.output_retries else "polling"),
                    )
                    context.phase = "output_read" if result.completed else "polling"
                    # Under load the SDK can see a job finish yet miss its output (its bounded
                    # output reads expire while queued) and reports that as `*_error`. The job
                    # is done, so asking again re-reads the output.
                    missing = result.stdout_error or result.stderr_error
                    if result.completed and (
                        missing is None or context.output_retries == _OUTPUT_RETRIES
                    ):
                        break
                    if result.completed:
                        context.output_retries += 1
                    context.phase = "poll_backoff"
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 3)
                if missing is not None or result.exit_code is None:
                    # The existing validation below raises after this context closes.
                    context.terminal_error_type = "SandboxError"
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
            process = await self._open_process(runtime, shlex.join(argv), env)
        except Exception as e:
            raise SandboxError(f"prime streaming process failed to start: {e}") from e
        return PrimeProcess(process, self.name)

    async def expose(self, port: int) -> str:
        raise SandboxError(
            "the prime runtime cannot expose a port; colocate the service with its "
            "consumer"
        )

    async def run_background(
        self, runtime: Runtime, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # A server-owned job runs without a client stream for its whole life.
        command = f"exec {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null"
        try:
            await self._start_background_job(runtime, command, env)
        except Exception as e:
            raise SandboxError(f"prime background launch failed: {e}") from e

    async def read(
        self, runtime: Runtime, path: str, max_bytes: int | None = None
    ) -> bytes:
        # The file API acts as the default user; a user view reads through exec.
        if max_bytes is not None or runtime.user is not None:
            reader = "cat" if max_bytes is None else f"head -c {max_bytes}"
            try:
                # Stream binary output: execute_command buffers base64 text for the
                # entire file. Bound the source read and the host buffer independently.
                process = await self._open_process(
                    runtime, f"{reader} -- {shlex.quote(path)}", {}
                )
                async with contextlib.aclosing(process):
                    with io.BytesIO() as data:
                        async for chunk in _stream_chunks(
                            process.stdout, self.name, "stdout"
                        ):
                            if (
                                max_bytes is not None
                                and data.tell() + len(chunk) > max_bytes
                            ):
                                raise SandboxError(
                                    "read stream exceeded its byte limit"
                                )
                            data.write(chunk)
                        stderr = b""
                        async for chunk in _stream_chunks(
                            process.stderr, self.name, "stderr"
                        ):
                            stderr = (stderr + chunk)[-500:]
                        if await _wait_for_process(process, self.name):
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
            await _release_client(client)
