"""The runtime contract: provision execution, run the program, tear down.

A runtime decides WHERE the program runs and HOW it reaches the host interception
server. Concrete runtimes live alongside this base; harnesses and the Environment
depend only on this contract, so they stay runtime-agnostic.
"""

import asyncio
import atexit
import contextlib
import hashlib
import logging
import shlex
import uuid
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)

logger = logging.getLogger(__name__)

# Ensure `uv` is available to run our PEP 723 scripts (the harness + tool servers): use it
# if present, else bootstrap it — via pip; else via the standalone installer (curl/wget),
# first installing curl + CA certs from the distro package manager when the image has no
# downloader at all (bare task images, e.g. Harbor's). It installs to ~/.local/bin, which
# we prepend to PATH so the next `uv run` finds it; uv then resolves each script's inline
# deps into its own cache, isolated from the eval process. (Needs network + one of
# uv / pip / curl / wget / apt-get / apk.)
_INSTALL_CURL = (  # only when the image has no downloader; needs a known package manager
    "{ command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; } "
    "|| { apt-get update -qq && apt-get install -y -qq curl ca-certificates; } "
    "|| apk add --no-cache curl ca-certificates"
)
_DOWNLOAD_UV = (
    "{ command -v curl >/dev/null 2>&1 && curl -LsSf https://astral.sh/uv/install.sh | sh; } "
    "|| { command -v wget >/dev/null 2>&1 && wget -qO- https://astral.sh/uv/install.sh | sh; }"
)
_ENSURE_UV = (
    'export PATH="$HOME/.local/bin:$PATH" UV_INSTALL_DIR="$HOME/.local/bin"; '
    "command -v uv >/dev/null 2>&1 "
    "|| pip install -q uv 2>/dev/null "
    f"|| {{ {_INSTALL_CURL}; {_DOWNLOAD_UV}; }}"
)

# The single port a self-publishing runtime (modal/prime) forwards to a public URL for a server
# hosted in its sandbox. A server placed in such a runtime binds this (on 0.0.0.0) and is reached
# at the runtime's public URL.
SERVICE_PORT = 8000


@dataclass(frozen=True)
class ProgramResult:
    exit_code: int
    stdout: str
    stderr: str


def parse_gpu(gpu: str | None) -> tuple[str | None, int]:
    """A Modal-style GPU spec -> (type, count) for providers that want them split:
    "A100" -> ("A100", 1), "A100:2" -> ("A100", 2), "2" -> (None, 2) (count only,
    provider-chosen type), None/"" -> (None, 0)."""
    if not gpu:
        return None, 0
    head, _, tail = gpu.partition(":")
    if tail:
        return head, int(tail)
    if head.isdigit():
        return None, int(head)
    return head, 1


# `stop()` frees a runtime's external resource on the normal path (the rollout's `finally`).
# A Ctrl-C / SIGTERM can cancel that `finally` mid-teardown, so runtimes are tracked in
# `_LIVE` and freed by a *synchronous* `atexit` hook (`cleanup`) — sync because the event
# loop is gone at interpreter shutdown. SIGKILL runs none of this.
_LIVE: "weakref.WeakSet[Runtime]" = weakref.WeakSet()
_atexit_armed = False


def register(runtime: "Runtime") -> None:
    """Track a runtime so the atexit hook can free it if a signal cuts its `finally` short.
    Weak, so a finished rollout's runtime drops out on its own; arms the hook once."""
    global _atexit_armed
    _LIVE.add(runtime)
    if not _atexit_armed:
        _atexit_armed = True
        atexit.register(cleanup_at_exit)


def cleanup_at_exit() -> None:
    """Synchronously free any runtime still live at interpreter shutdown — a Ctrl-C /
    SIGTERM cancelled its `finally` mid-teardown. Sync on purpose (the event loop is gone);
    best-effort and idempotent (a clean `stop` already ran it)."""
    for runtime in list(_LIVE):
        with contextlib.suppress(Exception):
            runtime.cleanup()


class Runtime(ABC):
    is_local: ClassVar[bool] = True
    """Whether this runtime shares the host network — a program inside it reaches a host service
    at localhost (no tunnel) and a service inside it is reachable at localhost. True for
    subprocess / docker(--network host); remote runtimes (modal/prime) override to False (they
    need a tunnel each way: `host_endpoint` inward, `expose` outward)."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or f"vf-{uuid.uuid4().hex[:12]}"
        """Resource name — the subprocess workdir, docker `--name`, prime sandbox name.
        The rollout passes its trace id, so the provisioned resource is greppable back to
        the rollout it serves; falls back to a unique `vf-` name (standalone / tool
        runtimes, where there's no single owning rollout)."""

    # --- identity / display ---

    @property
    def type(self) -> str:
        """The runtime's config discriminator ("subprocess" / "docker" / "prime" / "modal")."""
        return self.config.type

    @property
    def descriptor(self) -> str | None:
        """A short resolved id for display (None until provisioned). Overridden per
        runtime: subprocess workdir, docker image, prime sandbox id."""
        return None

    # --- lifecycle ---

    @abstractmethod
    async def start(self) -> None:
        """Provision execution (workspace / container / sandbox). Use `expose` to turn a
        host port into a URL the program can reach."""

    async def stop(self) -> None:
        """Free the provisioned resource on the normal path, off the event loop. Override
        only for teardown that must be async (e.g. a remote API call)."""
        await asyncio.to_thread(self.cleanup)

    def cleanup(self) -> None:
        """Synchronously free the provisioned resource — best-effort and idempotent. The
        source of truth for teardown: usable from the atexit backstop where async machinery
        is dead, and run off the event loop by `stop` on the normal path. Default no-op."""

    # --- execution ---

    @abstractmethod
    async def run(
        self, argv: list[str], env: dict[str, str], cpu_bound: bool = False
    ) -> ProgramResult:
        """Run `argv` (with the interception env vars `env`) to completion. `cpu_bound` flags a
        CPU-heavy command (e.g. a verifier that cold-imports sympy): the subprocess runtime caps
        how many such run at once (≈ the core count) so a burst of scores at an eval's start
        doesn't oversubscribe the cores — a no-op on isolated runtimes (their work runs in a
        sandbox, off the host)."""

    async def run_program(
        self, argv: list[str], env: dict[str, str], cpu_bound: bool = False
    ) -> ProgramResult:
        """Run the harness's MAIN program — the rollout itself (a possibly long-lived, stateful,
        agentic run) — as opposed to the short idempotent infra ops (write / mv / install /
        provisioning) that go through `run`. Identical to `run` here; `RetryingRuntime` overrides it
        to NOT retry, because re-running the program against the rollout's persistent trace would
        fork a duplicate branch (and re-execute against a runtime already mutated by the first
        attempt). A transient transport fault mid-program surfaces as a ProgramError for this
        rollout instead of a silent full restart."""
        return await self.run(argv, env, cpu_bound)

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        """Start `argv` as a background process in the runtime (combined output to
        `log`, a path in the workspace) and return immediately. It runs until `stop()`
        tears the runtime down. Used to host a tool server colocated with the harness."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support run_background"
        )

    async def run_uv_script(
        self,
        script: str | bytes,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cpu_bound: bool = False,
    ) -> ProgramResult:
        """Run a self-contained uv script (PEP 723 inline deps) in this runtime, with
        `args` as its positional arguments (the script's `sys.argv[1:]`). `cpu_bound` flags a
        CPU-heavy script (e.g. a math verifier) for the subprocess runtime's concurrency cap
        (see `run`); pass it for scoring scripts so a burst at an eval's start doesn't thrash.

        Writes `script`, ensures `uv` is present, and runs `uv run` — so the script's
        dependencies resolve into uv's cache inside the runtime, never the eval process.
        Built on `write`/`run`, so it works the same on every runtime. `args` are
        forwarded via the shell's `"$@"` (never interpolated), so spaces / quotes /
        newlines in them are safe; pass structured data as a JSON string if you need to.

        The script is written to a stable, content-addressed path (NOT the per-rollout
        workspace): uv keys its per-script environment by the script's full path, so a
        unique path per call would mint a fresh env every rollout. A path derived from the
        content means identical scripts share one path → uv reuses one env, bounded by the
        number of distinct scripts. Published via a unique temp + atomic `mv`, so
        concurrent rollouts writing the same content never race a half-written read."""
        data = script.encode() if isinstance(script, str) else script
        path = f"/tmp/vf-scripts/{hashlib.sha256(data).hexdigest()}.py"
        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
        await self.write(tmp, data)
        await self.run(
            ["sh", "-c", f"mv -f {shlex.quote(tmp)} {shlex.quote(path)}"], {}
        )
        command = f'{_ENSURE_UV}; exec uv run {shlex.quote(path)} "$@"'
        # The write/mv above are idempotent infra (retried); the script run is the rollout's
        # program — `run_program`, so it is not retried (see Runtime.run_program).
        return await self.run_program(
            ["sh", "-c", command, path, *(args or [])], env or {}, cpu_bound
        )

    # --- filesystem ---

    @abstractmethod
    async def read(self, path: str) -> bytes:
        """Read a file from the runtime's workspace. The caller need not know
        whether that's the host fs or across a container/sandbox boundary."""

    @abstractmethod
    async def write(self, path: str, data: bytes) -> None:
        """Write a file into the runtime's workspace, creating parent dirs."""

    # --- networking ---

    @property
    def published_port(self) -> int | None:
        """A fixed port this runtime exposes to the outside at startup, declared up front to the
        provider (Modal forwards only ports named at `Sandbox.create`). When set, a server placed
        here binds it instead of a host-chosen free port, and `expose` returns its public URL.
        `None` for host-networked runtimes (subprocess/docker), which pick a free port and are
        reached over the shared host network."""
        return None

    async def expose(self, port: int) -> str | None:
        """Publish a port running *inside this runtime* to a URL reachable from the host/outside,
        or None when local (it's on the host network — reach it at localhost). A remote runtime
        overrides this with the provider's native port exposure (modal `tunnels()`, prime
        `client.expose`), torn down with the sandbox in `stop()`. The reverse of `host_endpoint`
        (which reaches a host port from inside a runtime)."""
        return None


class RetryingRuntime(Runtime):
    """Wraps a runtime to retry each call on a transient error (tenacity, up to
    `max_retries` retries). A program's own failure surfaces as a `ProgramResult` (non-zero
    exit), not an exception, so retries fire only on infra/transport faults — provisioning,
    exec transport, file I/O across the runtime boundary. `CancelledError` (a
    `BaseException`) and `NotImplementedError` (an unsupported op) are never retried. Sync
    teardown (`cleanup`) and display (`descriptor`) delegate straight through. The rollout's
    program exec (`run_program`, including the script run inside `run_uv_script`) is deliberately
    NOT retried: re-running a stateful/agentic program against the rollout's persistent trace would
    fork a duplicate branch (and re-run against a runtime the first attempt already mutated), so a
    mid-program transport fault surfaces as a ProgramError for that rollout. `run_uv_script`'s
    write/mv staging still runs over the retrying `write`/`run`."""

    def __init__(self, inner: Runtime, max_retries: int) -> None:
        super().__init__(inner.name)
        self.inner = inner
        self.max_retries = max_retries
        # One Retrying, reused across (and concurrent within) calls: the control flow runs
        # off a per-call RetryCallState, so only its bookkeeping `.statistics` is shared.
        self._retrying = AsyncRetrying(
            stop=stop_after_attempt(max_retries + 1),
            retry=retry_if_exception_type(Exception)
            & retry_if_not_exception_type(NotImplementedError),
            before_sleep=self._log_retry,
            reraise=True,
        )

    def _log_retry(self, state: RetryCallState) -> None:
        # before_sleep fires after a failed attempt, before the imminent retry — so
        # attempt_number is the retry index (1 on the first retry); count out of max_retries.
        logger.warning(
            "retrying runtime.%s (retry %d/%d) after error: %s",
            getattr(state.fn, "__name__", "call"),
            state.attempt_number,
            self.max_retries,
            state.outcome.exception(),
        )

    async def _retry(self, fn, *args):
        return await self._retrying(fn, *args)

    async def start(self) -> None:
        await self._retry(self.inner.start)

    async def stop(self) -> None:
        await self._retry(self.inner.stop)

    def cleanup(self) -> None:
        self.inner.cleanup()

    async def run(
        self, argv: list[str], env: dict[str, str], cpu_bound: bool = False
    ) -> ProgramResult:
        return await self._retry(self.inner.run, argv, env, cpu_bound)

    async def run_program(
        self, argv: list[str], env: dict[str, str], cpu_bound: bool = False
    ) -> ProgramResult:
        """The rollout's program exec is NOT retried (see the class docstring) — straight to
        inner."""
        return await self.inner.run(argv, env, cpu_bound)

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        await self._retry(self.inner.run_background, argv, env, log)

    @property
    def type(self) -> str:
        return self.inner.type

    @property
    def published_port(self) -> int | None:
        return self.inner.published_port

    @property
    def is_local(self) -> bool:
        return self.inner.is_local

    @property
    def descriptor(self) -> str | None:
        return self.inner.descriptor

    async def expose(self, port: int) -> str | None:
        return await self._retry(self.inner.expose, port)

    async def read(self, path: str) -> bytes:
        return await self._retry(self.inner.read, path)

    async def write(self, path: str, data: bytes) -> None:
        await self._retry(self.inner.write, path, data)


@contextlib.asynccontextmanager
async def host_endpoint(port: int, is_local: bool, labels: list[str] | None = None):
    """Yield a URL a program *inside a runtime* uses to reach a HOST service on `port`. A local
    runtime shares the host network → localhost; a remote one needs a host-side reverse tunnel
    (`prime_tunnel`), torn down on exit. This is the host-side, provider-agnostic counterpart to
    `Runtime.expose` (which publishes a port running *inside* a runtime) — so the runtime only
    reports `is_local` and callers (interception pool, rollout, tool serving) bridge to the host
    here, rather than every runtime reimplementing the tunnel."""
    if is_local:
        yield f"http://127.0.0.1:{port}"
        return
    from prime_tunnel import Tunnel

    from verifiers.v1.errors import ProgramError
    from verifiers.v1.runtimes.limiters import TUNNEL_LIMITER

    tunnel = Tunnel(local_port=port, labels=labels or None)
    try:
        async with (
            TUNNEL_LIMITER
        ):  # shared prime_tunnel rate (512/min, runtime-independent)
            url = str(await tunnel.start()).rstrip("/")
    except Exception as e:
        raise ProgramError(f"host tunnel failed (port {port}): {e}") from e
    try:
        yield url
    finally:
        with contextlib.suppress(Exception):
            tunnel.sync_stop()


class _Host:
    """The host network as a `reachable_url` location: shares the host network (so it's `is_local`)
    and publishes nothing itself (it's reached *into* via `host_endpoint`, not via `expose`)."""

    is_local = True


HOST = _Host()
"""The host network, as a service location (e.g. the interception server) or a consumer (the
framework driving a user sim) — see `reachable_url`."""


@contextlib.asynccontextmanager
async def reachable_url(
    service, port: int, *, consumer=None, consumer_is_local: bool = True
):
    """Yield a URL for the service at (`service`, `port`) reachable from its consumer — the single
    place tool / user / interception reachability is decided, over the two primitives `expose`
    (publish *out* of a runtime) and `host_endpoint` (reach *into* the host from a runtime).

    `service` is the `Runtime` the service runs in, or `HOST` (a host-network service). `consumer` is
    the consuming `Runtime` (used for the colocated check and its locality); leave it `None` for a
    host consumer or an eval-level consumer with no single instance (a shared tool reused by every
    rollout's harness) and pass its locality as `consumer_is_local`:

    - same location (a colocated tool in the consumer's own runtime, or host -> host): localhost;
    - the service runs in a sandbox (a remote runtime): its own published URL (`expose`), reachable
      from anywhere;
    - the service is on the host network: localhost to a host-network consumer, else a host tunnel
      (`host_endpoint`)."""
    is_local = consumer.is_local if consumer is not None else consumer_is_local
    if service is consumer:  # colocated in the consumer's runtime (or host -> host)
        yield f"http://127.0.0.1:{port}"
    elif (
        service is not HOST and not service.is_local
    ):  # in a sandbox → it publishes its own port
        yield await service.expose(port)
    else:  # on the host network → reach it from wherever the consumer runs
        async with host_endpoint(port, is_local) as url:
            yield url
