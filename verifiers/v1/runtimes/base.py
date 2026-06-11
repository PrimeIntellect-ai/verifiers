"""The runtime contract: provision execution, run the program, tear down.

A runtime decides WHERE the program runs and HOW it reaches the host interception
server. Concrete runtimes live alongside this base; harnesses and the Environment
depend only on this contract, so they stay runtime-agnostic.
"""

import asyncio
import atexit
import contextlib
import fcntl
import hashlib
import logging
import os
import shlex
import tempfile
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass

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


# Host-global leaky buckets pacing a remote runtime's resource creation (Modal sandboxes,
# Prime tunnels) under the provider's per-account rate limit. Backed by a lock file so the
# rate holds across EVERY process on the host — the single-process eval and all the elastic
# env-server worker processes alike — not just within one process, so the configured rate is
# the actual account-wide rate (assuming one env-server host). Keyed by name: one bucket file
# per name, shared by every process (and every run) on the host that opens it.
_LIMITER_DIR = os.path.join(tempfile.gettempdir(), "vf-rate-limiters")


class CreationLimiter:
    """An async leaky bucket shared across processes via a lock file: each `async with`
    reserves the next `1/per_sec`-spaced slot (advancing the on-disk cursor under an exclusive
    flock) and sleeps until it, so the aggregate creation rate across all host processes stays
    at `per_sec`. The reservation runs off the event loop; the wait does not hold the lock."""

    def __init__(self, name: str, per_sec: float) -> None:
        self._interval = 1 / per_sec
        self._path = os.path.join(_LIMITER_DIR, f"{name}.bucket")

    def _reserve(self) -> float:
        os.makedirs(_LIMITER_DIR, exist_ok=True)
        # monotonic is host-wide on Linux, so the cursor is comparable across processes.
        with open(self._path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                data = f.read().strip()
                now = time.monotonic()
                slot = max(now, float(data) if data else 0.0)
                f.seek(0)
                f.truncate()
                f.write(repr(slot + self._interval))
                f.flush()
                return slot - now
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    async def __aenter__(self) -> "CreationLimiter":
        wait = await asyncio.to_thread(self._reserve)
        if wait > 0:
            await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


_creation_limiters: dict[str, CreationLimiter] = {}


def creation_limiter(per_sec: float | None, name: str) -> CreationLimiter | None:
    """A host-global limiter pacing `name`'s creation to `per_sec`/s (None/<= 0 disables).

    All callers (and processes) sharing a `name` share one bucket, so use one rate per name."""
    if not per_sec or per_sec <= 0:
        return None
    limiter = _creation_limiters.get(name)
    if limiter is None:
        limiter = _creation_limiters[name] = CreationLimiter(name, per_sec)
    return limiter


# The prime_tunnel service caps tunnel starts at 512/min per API token — a property of the
# tunnel service, shared by every runtime that opens a prime_tunnel (prime AND modal). One
# host-global limiter, not a per-runtime config knob.
_TUNNELS_PER_MIN = 512
_TUNNEL_LIMITER = creation_limiter(_TUNNELS_PER_MIN / 60, "prime-tunnel")


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
    def __init__(self, name: str | None = None) -> None:
        self.name = name or f"vf-{uuid.uuid4().hex[:12]}"
        """Resource name — the subprocess workdir, docker `--name`, prime sandbox name.
        The rollout passes its trace id, so the provisioned resource is greppable back to
        the rollout it serves; falls back to a unique `vf-` name (standalone / tool
        runtimes, where there's no single owning rollout)."""

    @abstractmethod
    async def start(self) -> None:
        """Provision execution (workspace / container / sandbox). Use `expose` to turn a
        host port into a URL the program can reach."""

    def cleanup(self) -> None:
        """Synchronously free the provisioned resource — best-effort and idempotent. The
        source of truth for teardown: usable from the atexit backstop where async machinery
        is dead, and run off the event loop by `stop` on the normal path. Default no-op."""

    async def stop(self) -> None:
        """Free the provisioned resource on the normal path, off the event loop. Override
        only for teardown that must be async (e.g. a remote API call)."""
        await asyncio.to_thread(self.cleanup)

    async def expose(self, port: int) -> str:
        """A base URL the program (inside this runtime) can use to reach a host service
        on localhost `port` — the interception endpoint and host-side tool servers both
        go through this. Default: localhost, which works when the runtime shares the host
        network (subprocess, docker --network host). Remote runtimes (prime) override to
        tunnel the port."""
        return f"http://127.0.0.1:{port}"

    @abstractmethod
    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """Run `argv` (with the interception env vars `env`) to completion."""

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        """Start `argv` as a background process in the runtime (combined output to
        `log`, a path in the workspace) and return immediately. It runs until `stop()`
        tears the runtime down. Used to host a tool server colocated with the harness."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support run_background"
        )

    @property
    def descriptor(self) -> str | None:
        """A short resolved id for display (None until provisioned). Overridden per
        runtime: subprocess workdir, docker image, prime sandbox id."""
        return None

    async def public_url(self, port: int) -> str | None:
        """A URL anyone can use to reach `port` running *inside this runtime*, or None if
        this runtime can't self-publish (it's on the host network, so the caller reaches
        it via the harness runtime's `expose`). A remote runtime overrides this to publish
        the port (e.g. a prime sandbox exposes it natively). Cleaned up by `stop()`."""
        return None

    @abstractmethod
    async def read(self, path: str) -> bytes:
        """Read a file from the runtime's workspace. The caller need not know
        whether that's the host fs or across a container/sandbox boundary."""

    @abstractmethod
    async def write(self, path: str, data: bytes) -> None:
        """Write a file into the runtime's workspace, creating parent dirs."""

    async def run_uv_script(
        self,
        script: str | bytes,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> ProgramResult:
        """Run a self-contained uv script (PEP 723 inline deps) in this runtime, with
        `args` as its positional arguments (the script's `sys.argv[1:]`).

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
        return await self.run(["sh", "-c", command, path, *(args or [])], env or {})


class RetryingRuntime(Runtime):
    """Wraps a runtime to retry each call on a transient error (tenacity, up to
    `max_retries` retries). A program's own failure surfaces as a `ProgramResult` (non-zero
    exit), not an exception, so retries fire only on infra/transport faults — provisioning,
    exec transport, file I/O across the runtime boundary. `CancelledError` (a
    `BaseException`) and `NotImplementedError` (an unsupported op) are never retried. Sync
    teardown (`cleanup`) and display (`descriptor`) delegate straight through;
    `run_uv_script` is inherited, so it runs over the retrying `write`/`run`."""

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
        logger.warning(
            "retrying runtime.%s (attempt %d/%d) after error: %s",
            getattr(state.fn, "__name__", "call"),
            state.attempt_number,
            self.max_retries + 1,
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

    async def expose(self, port: int) -> str:
        return await self._retry(self.inner.expose, port)

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return await self._retry(self.inner.run, argv, env)

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        await self._retry(self.inner.run_background, argv, env, log)

    @property
    def descriptor(self) -> str | None:
        return self.inner.descriptor

    async def public_url(self, port: int) -> str | None:
        return await self._retry(self.inner.public_url, port)

    async def read(self, path: str) -> bytes:
        return await self._retry(self.inner.read, path)

    async def write(self, path: str, data: bytes) -> None:
        await self._retry(self.inner.write, path, data)
