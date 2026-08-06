"""Remote Prime sandbox runtime.

`expose` (sandbox port -> public URL) uses the SDK's native exposure (`client.expose`), so a
host-side harness/framework can reach a tool server hosted in the sandbox. The reverse
direction (a program in the sandbox reaching a host service) is the shared host-side
`Tunnel` (interception.tunnel), not the runtime's concern.
"""

import asyncio
import contextlib
import logging
import math
import shlex
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path, PurePosixPath
from typing import ClassVar, Literal
from urllib.parse import urlsplit

from prime_sandboxes.models import validate_egress_lists
from pydantic import Field, model_validator

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    NetworkPolicyConfig,
    ProgramResult,
    Runtime,
    RuntimeProcess,
    parse_gpu,
)
from verifiers.v1.runtimes.limiters import creation_limiter

logger = logging.getLogger(__name__)

MAX_LIFETIME = 24 * 60 * 60
"""Prime's fixed cap (seconds) on any sandbox's total lifetime."""


class PrimeConfig(NetworkPolicyConfig):
    type: Literal["prime"] = "prime"
    image: str = "python:3.11-slim"
    """Docker image to run. Any pullable ref works: on the first use of an image, the
    platform auto-builds what the sandbox needs from it (a VM image for `vm` sandboxes,
    ~10 minutes) and caches the result, so later sandboxes on the same ref start in
    seconds."""
    workdir: str = "/app"
    vm: bool = False
    """Run as a micro-VM rather than a container (kernel features / stronger isolation)."""
    guaranteed: bool = False
    """Request guaranteed (vs best-effort) capacity."""
    region: str | None = None
    """Region to provision in (None = provider-chosen)."""
    labels: list[str] = Field(default_factory=list)
    """Labels attached to the sandbox."""
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float = 1.0
    """CPU cores."""
    memory: float = 2.0
    """Memory in GB."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "A100:2" (a bare count = provider-chosen type)."""
    disk: float = 5.0
    """Disk in GB."""
    lifetime_timeout: float = Field(default=MAX_LIFETIME, gt=0, le=MAX_LIFETIME)
    """Maximum sandbox lifetime in seconds."""
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

    @model_validator(mode="after")
    def _validate_idle_timeout(self) -> "PrimeConfig":
        if (
            not self.vm
            and self.idle_timeout is not None
            and self.idle_timeout > self.lifetime_timeout
        ):
            raise ValueError(
                f"idle_timeout ({self.idle_timeout}s) must not exceed the "
                f"lifetime_timeout ({self.lifetime_timeout}s)"
            )
        return self


class PrimeRuntimeInfo(PrimeConfig, BaseRuntimeInfo):
    image_cached: bool | None = None
    """Whether the platform already had the image at create (None until then). False means
    a first-use auto-build ran while this sandbox waited to start."""


class PrimeProcess(RuntimeProcess):
    def __init__(self, process, transport=None) -> None:
        self._process = process
        self._transport = transport
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

    async def aclose(self) -> None:
        # The SDK handle owns the command-session stream and its pump task; the
        # transport owns this process's dedicated connection. Both must go even
        # when the remote exit event never arrived — an abandoned stream pins a
        # gateway HTTP/2 stream slot until the sandbox is deleted.
        try:
            await self._process.aclose()
        finally:
            if self._transport is not None:
                with contextlib.suppress(Exception):
                    await self._transport.aclose()


class PrimeRuntime(Runtime):
    is_local: ClassVar[bool] = False

    def __init__(self, config: PrimeConfig, name: str | None = None) -> None:
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

        self._client = AsyncSandboxClient()
        # Map the resources onto prime's API (minutes, split GPU; memory/disk are already
        # GB). gpu_type/region are only sent when set (else provider-chosen).
        gpu_type, gpu_count = parse_gpu(self.config.gpu)
        # prime's idle timeout is in whole minutes; convert from the seconds config surface
        # (floored to the SDK's 1-minute minimum). VM sandboxes don't support an idle timeout
        # (the API 422s on it), so it's dropped there rather than failing every VM rollout.
        idle_minutes = (
            max(1, math.ceil(self.config.idle_timeout / 60))
            if self.config.idle_timeout is not None and not self.config.vm
            else None
        )
        options = {
            "cpu_cores": self.config.cpu,
            "memory_gb": self.config.memory,
            "disk_size_gb": self.config.disk,
            "gpu_count": gpu_count,
            "timeout_minutes": max(1, math.ceil(self.config.lifetime_timeout / 60)),
            "idle_timeout_minutes": idle_minutes,
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
                        vm=self.config.vm,
                        guaranteed=self.config.guaranteed,
                        **{k: v for k, v in options.items() if v is not None},
                    )
                )
            self.info.id = sandbox.id
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
            await self._client.wait_for_creation(self.info.id)
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
            result = await self._client.run_background_job(
                self.info.id,
                shlex.join(argv),
                timeout=self.config.lifetime_timeout,
                working_dir=self.config.workdir,
                env=env,
                poll_interval=1,
            )
        except (
            Exception
        ) as e:  # a sandbox/API failure is one rollout's problem, not the eval's
            raise SandboxError(f"prime exec failed: {e}") from e
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
        try:
            process, transport = await self._open_process_on_own_transport(
                shlex.join(argv), env
            )
        except Exception as e:
            raise SandboxError(f"prime live process failed to start: {e}") from e
        return PrimeProcess(process, transport)

    async def _open_process_on_own_transport(self, command: str, env: dict[str, str]):
        """Start a live process whose RPCs ride one dedicated HTTP connection.

        The SDK routes every command-session RPC over pyqwest's shared default
        transport, and the gateway caps one HTTP/2 connection at 100 concurrent
        streams. Each live process pins one stream for its whole life, so
        accumulated sessions starve stdin/signal RPCs into their 30s deadline
        ("deadline_exceeded") and queue new process streams for minutes under
        the 24h stream timeout. One transport per process keeps a process
        competing only with itself. Rebuilds the SDK's open_process wiring from
        prime-sandboxes internals (pinned >=0.2.35); drop this once the SDK
        isolates its own command-session transports."""
        from connectrpc.client import ConnectClient
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError
        from prime_sandboxes.core import APIError
        from prime_sandboxes.process import AsyncSandboxProcess
        from prime_sandboxes.rpc_command_session import (
            COMMAND_SESSION_SEND_INPUT_RPC_METHOD,
            COMMAND_SESSION_SEND_SIGNAL_RPC_METHOD,
            COMMAND_SESSION_START_RPC_METHOD,
            build_command_session_send_input_request,
            build_command_session_send_signal_request,
            build_command_session_start_request,
        )
        from prime_sandboxes.sandbox import (
            _LIVE_PROCESS_TIMEOUT_MS,
            _PROCESS_INPUT_TIMEOUT_MS,
            _PROCESS_SIGNAL_TIMEOUT_MS,
        )
        from pyqwest import Client as HTTPClient
        from pyqwest import HTTPTransport

        client = self._client
        sandbox_id = self.info.id
        try:
            # pyqwest >= 0.7 leaves system CA trust off for bare transports; a
            # bare HTTPTransport() there fails TLS with "UnknownIssuer".
            transport = HTTPTransport(tls_include_system_certs=True)
        except TypeError:  # pyqwest 0.6.x trusts system CAs by default
            transport = HTTPTransport()
        http_client = HTTPClient(transport=transport)

        async def authed_target() -> tuple[str, dict[str, str]]:
            auth = await client._auth_cache.get_or_refresh(sandbox_id)
            gateway_url = auth["gateway_url"].rstrip("/")
            return (
                f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}",
                {"Authorization": f"Bearer {auth['token']}"},
            )

        async def control_rpc(request, method, timeout_ms: int, operation: str) -> None:
            # Mirrors AsyncSandboxClient._execute_process_control_rpc, including
            # its error strings, but sends over this process's own transport.
            reauthed = False
            while True:
                base_url, headers = await authed_target()
                rpc_client = ConnectClient(base_url, http_client=http_client)
                try:
                    await rpc_client.execute_unary(
                        request=request,
                        method=method,
                        headers=headers,
                        timeout_ms=timeout_ms,
                    )
                    return
                except ConnectError as error:
                    if (
                        error.code == Code.UNAUTHENTICATED
                        and await client._should_retry_401(sandbox_id, reauthed)
                    ):
                        reauthed = True
                        continue
                    raise APIError(
                        f"process {operation} RPC failed ({error.code.value}): {error.message}"
                    ) from error
                finally:
                    await rpc_client.close()

        async def write_stdin(pid: int, data: bytes) -> None:
            await control_rpc(
                build_command_session_send_input_request(pid, data),
                COMMAND_SESSION_SEND_INPUT_RPC_METHOD,
                _PROCESS_INPUT_TIMEOUT_MS,
                "stdin",
            )

        async def send_signal(pid: int, signal: Literal["terminate", "kill"]) -> None:
            await control_rpc(
                build_command_session_send_signal_request(pid, signal),
                COMMAND_SESSION_SEND_SIGNAL_RPC_METHOD,
                _PROCESS_SIGNAL_TIMEOUT_MS,
                "signal",
            )

        try:
            base_url, headers = await authed_target()
            stream_client = ConnectClient(base_url, http_client=http_client)
            stream = stream_client.execute_server_stream(
                request=build_command_session_start_request(
                    command, self.config.workdir, env, stdin=True
                ),
                method=COMMAND_SESSION_START_RPC_METHOD,
                headers=headers,
                timeout_ms=_LIVE_PROCESS_TIMEOUT_MS,
            )
            process = await AsyncSandboxProcess._create(
                stream_client, stream, write_stdin, send_signal
            )
        except BaseException:
            with contextlib.suppress(Exception):
                await transport.aclose()
            raise
        return process, transport

    async def expose(self, port: int) -> str | None:
        # Publish a server hosted IN the sandbox via the SDK's native port exposure → a public
        # HTTPS URL. Removed when the sandbox is deleted in stop(), so a tool in its own prime
        # sandbox needs no host tunnel. Port exposure is region-gated: many regions (incl. the
        # backend default, which lands in us-central) 400 it; `us` supports it. TODO: re-enable the
        # prime cases in the e2e `skip_if_unexposable` guard once prime exposes ports in any region.
        try:
            exposed = await self._client.expose(self.info.id, port)
        except Exception as e:  # surface prime's exposure constraints actionably
            raise SandboxError(
                "prime port exposure failed — port exposure isn't supported in this sandbox's "
                "region; pin `tools.runtime.region` to a region that supports it (e.g. `us`), or "
                f"use a colocated / docker / modal tools.runtime instead. ({e})"
            ) from e
        logger.info("prime: exposed sandbox port %d at %s", port, exposed.url)
        return exposed.url.rstrip("/")

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        command = f"exec {shlex.join(argv)} > {shlex.quote(log)} 2>&1"
        try:
            await self._client.start_background_job(
                self.info.id,
                command,
                working_dir=self.config.workdir,
                env=env,
            )
        except Exception as e:
            raise SandboxError(f"prime background launch failed: {e}") from e

    async def _read(self, path: str) -> bytes:
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
        try:
            await self._client.execute_command(
                self.info.id,
                f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))}",
            )
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
        if self.info.id is not None:  # keep info.id available after teardown
            try:
                await client.delete(self.info.id)
            except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
                logger.warning(
                    "prime: failed to delete sandbox %s: %s", self.info.id, e
                )
        with contextlib.suppress(Exception):
            await client.aclose()
