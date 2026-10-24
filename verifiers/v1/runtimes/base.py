"""RuntimeBackend ownership and provider operations."""

from __future__ import annotations

import asyncio
import atexit
import base64
import contextlib
import copy
import logging
import shlex
import uuid
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError, missing_file_kind
from verifiers.v1.utils.aio import run_shielded

if TYPE_CHECKING:
    from verifiers.v1.runtimes.runtime import Runtime

logger = logging.getLogger(__name__)

# Ensure `uv` supports both script commands used below. Reuse a compatible installation;
# upgrade older binaries bundled in task images. Prefer pip on
# Python images, then fall back to the standalone installer (curl/wget), installing curl + CA
# certs when a bare image has no downloader. Both install paths land in ~/.local/bin, which we
# prepend to PATH first. (Installing needs network + one of pip / curl / wget / apt-get / apk;
# a box with none of them gets the host's static uv instead, see `host_uv`.)
_INSTALL_CURL = (  # only when the image has no downloader; needs a known package manager
    "{ command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; } "
    "|| { apt-get update -qq && apt-get install -y -qq curl ca-certificates; } "
    "|| apk add --no-cache curl ca-certificates"
)
_DOWNLOAD_UV = (
    "{ command -v curl >/dev/null 2>&1 && curl -LsSf https://astral.sh/uv/install.sh | sh; } "
    "|| { command -v wget >/dev/null 2>&1 && wget -qO- https://astral.sh/uv/install.sh | sh; }"
)
FALLBACK_BIN = "/tmp/.verifiers/bin"
"""Where the host's uv goes when the user cannot write `$HOME/.local/bin` (a home whose
`.local` a root step created first); on PATH right after it."""
_ENSURE_UV = (
    f'export PATH="$HOME/.local/bin:{FALLBACK_BIN}:$PATH" UV_INSTALL_DIR="$HOME/.local/bin"; '
    "uv_supports_scripts() { "
    "uv sync --help 2>/dev/null | grep -q -- '--script' "
    "&& uv python find --help 2>/dev/null | grep -q -- '--script'; }; "
    "uv_supports_scripts "
    "|| { pip install -q -U --user uv 2>/dev/null && hash -r && uv_supports_scripts; } "
    f"|| {{ {{ {_INSTALL_CURL}; {_DOWNLOAD_UV}; }} && hash -r && uv_supports_scripts; }}"
)

# The single port a sandbox runtime forwards out for a server hosted in it, e.g. a host
# loopback port on Docker. A server placed in such
# a runtime binds this (on 0.0.0.0) and is reached at the URL `expose` returns.
SERVICE_PORT = 8000


@dataclass(frozen=True)
class ProgramResult:
    exit_code: int
    stdout: str
    stderr: str


class RuntimeProcess(ABC):
    """A process whose stdio streams across a runtime boundary."""

    stdout: AsyncIterator[bytes]
    stderr: AsyncIterator[bytes]

    @abstractmethod
    async def write(self, data: bytes) -> None:
        pass

    @abstractmethod
    async def wait(self) -> int:
        pass

    @abstractmethod
    async def poll(self) -> int | None:
        """Return the underlying process's exit code without waiting for it to exit."""

    @abstractmethod
    async def terminate(self) -> None:
        pass

    @abstractmethod
    async def kill(self) -> None:
        pass


def parse_gpu(gpu: str | None) -> tuple[str | None, int]:
    """A GPU spec -> (type, count) for providers that want them split:
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


# `stop()` frees a runtime's external resource on the normal path (the rollout's `finally`),
# shielded so a Ctrl-C / SIGTERM task cancellation can't cut it short. A second Ctrl-C
# raises KeyboardInterrupt out of the event loop itself — no task-level shield survives
# that — so runtimes are also tracked in `_LIVE` and freed by a *synchronous* `atexit`
# hook (`cleanup`), sync because the loop is gone at interpreter shutdown. SIGKILL runs
# none of this.
_LIVE: weakref.WeakSet[RuntimeBackend] = weakref.WeakSet()
_atexit_armed = False


def register(backend: RuntimeBackend) -> None:
    """Track a runtime so the atexit hook can free it if a signal cuts its `finally` short.
    Weak, so a finished rollout's runtime drops out on its own; arms the hook once."""
    global _atexit_armed
    _LIVE.add(backend)
    if not _atexit_armed:
        _atexit_armed = True
        atexit.register(cleanup_at_exit)


def cleanup_at_exit() -> None:
    """Synchronously free any runtime still live at interpreter shutdown — a Ctrl-C /
    SIGTERM cancelled its `finally` mid-teardown. Sync on purpose (the event loop is gone);
    best-effort and idempotent (a clean `stop` already ran it)."""
    for backend in list(_LIVE):
        with contextlib.suppress(Exception):
            backend.cleanup()


class BaseRuntimeInfo(BaseConfig):
    id: str | None = None
    borrowed: bool = False
    """Whether the run was placed into a live box owned by someone else
    (`Agent.run(runtime=...)`) rather than provisioning its own."""
    image_digest: str | None = Field(default=None, exclude=True)
    """Read from records written while verifiers recorded each box's image digest, so
    they still load; never set or written."""


@dataclass
class RuntimeCache:
    """Framework state for one provisioned resource, kept off the backend contract:
    prepared uv scripts and the tool-server sources installed into it."""

    uv_interpreters: dict[str, str] = field(default_factory=dict)
    uv_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    curl_ready: bool = False
    mcp_sources: set[str] = field(default_factory=set)
    mcp_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RuntimeBackend(ABC):
    """Provider implementation for one container, VM, or local execution environment.

    A backend implements `start`, `exec`, `write`, and `cleanup` or `teardown`, and
    may override `run`, `read`, `expose`, `prepare_execution`, and `host_url`. Its
    `SUPPORTS_*` flags declare optional features. Tasks and harnesses use it through
    `Runtime` views (`bind`), which add user and environment bindings.
    """

    is_local: bool = False
    """Whether this runtime exchanges host-local URLs without a public tunnel, as
    subprocess and docker do. Remote runtimes use a host `Tunnel` inward plus `expose`
    outward."""

    SUPPORTS_TASK_IMAGES: ClassVar[bool] = False
    """Whether the runtime boots a task image in its own filesystem. Tasks with an image,
    container-only harnesses, and isolated grading require it."""

    SUPPORTS_USER_SWITCHING: ClassVar[bool] = False
    """Whether handles can bind another existing user."""

    SUPPORTS_PORT_EXPOSURE: ClassVar[bool] = False
    """Whether `expose` can publish a port inside this runtime to the outside, so a
    server here is reachable from another runtime."""

    SUPPORTS_COMPOSE: ClassVar[bool] = False
    """Whether the runtime can run a task's Docker Compose services."""

    scripts_dir: ClassVar[str] = "/tmp/vf-scripts"
    """Digest-keyed PEP 723 scripts. The default follows the runtime's TMPDIR;
    explicit backend paths, including the subprocess per-user cache, stay fixed."""

    info: BaseRuntimeInfo

    def __init__(self, name: str | None = None) -> None:
        self.name = name or f"vf-{uuid.uuid4().hex[:12]}"
        # Per-run task values live on the runtime rather than its serializable config/info.
        # Explicit process values (model credentials, proxy settings, etc.) override these.
        self.env: dict[str, str] = {}
        self.cache = RuntimeCache()
        self._setup_claimed = False
        self._background: list[RuntimeProcess] = []
        self.stopped = False
        """Whether teardown has begun (set by `stop`). A stopped runtime is dead: a rollout
        refuses to borrow one — the owner tore it down, so any use is a lifetime bug in the
        borrowing program, caught up front instead of failing opaquely mid-harness."""

    def bind(self, *, env: Mapping[str, str] | None = None) -> Runtime:
        """Access this resource with immutable bindings; does not provision anything."""
        from verifiers.v1.runtimes.runtime import Runtime

        return Runtime(self, env=self.env if env is None else env)

    def deploy(self, **kwargs):
        from verifiers.v1.runtimes.deployment import Deployment

        return Deployment(self, **kwargs)

    @property
    def type(self) -> str:
        return self.config.type

    @abstractmethod
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        """Free the provisioned resource on the normal path (the owner's `finally`),
        shielded from cancellation: a Ctrl-C / SIGTERM cancels that `finally` mid-await,
        and an interrupted teardown leaks the container / paid sandbox. Runs `teardown`
        to completion, then re-raises the cancellation. Framework method — override
        `teardown`, not this."""
        self.stopped = True  # before the await: no new borrows once teardown begins
        await run_shielded(self._freed())

    async def _freed(self) -> None:
        """`teardown`, then the atexit backstop forgets the runtime: its resource is
        freed, and a second, synchronous free at exit would only repeat it (one blocking
        provider call per runtime, holding the process's exit)."""
        await self.teardown()
        _LIVE.discard(self)

    async def teardown(self) -> None:
        """Free the provisioned resource, off the event loop. Override only for teardown
        that must be async (e.g. a remote API call); `stop` shields it from cancellation.
        Best-effort and idempotent, like `cleanup`. An override must not consume state
        `cleanup` keys off before its first await: if the event loop dies mid-teardown
        (second Ctrl-C), the atexit backstop must still find the resource."""
        await asyncio.to_thread(self.cleanup)

    def cleanup(self) -> None:
        """Synchronously free the provisioned resource — best-effort and idempotent. The
        source of truth for teardown: usable from the atexit backstop where async machinery
        is dead, and run off the event loop by `stop` on the normal path. Default no-op."""

    def adopt(self, other: RuntimeBackend) -> RuntimeBackend:
        """Become the box `other` is, in place: this backend and every `Runtime` view
        bound to it now act on `other`'s box, and `other` shares that state too. Returns
        a backend holding the box this one was, for its owner to tear down. Both must be
        the same backend type."""
        if type(other) is not type(self):
            raise TypeError(
                f"a {type(self).__name__} cannot adopt a {type(other).__name__}"
            )
        state = self.__dict__
        old = copy.copy(self)
        old.__dict__ = dict(state)
        state.clear()
        state.update(other.__dict__)
        other.__dict__ = state
        return old

    @property
    def restored(self) -> bool:
        """Whether this box started from a checkpoint (`restore_config`) rather than its
        image: it already holds a past run's files, but none of its processes."""
        return False

    @property
    def supports_checkpoint(self) -> bool:
        """Whether `checkpoint()` and `delete_checkpoint()` are implemented for this
        backend instance."""
        return (
            type(self).checkpoint is not RuntimeBackend.checkpoint
            and type(self).delete_checkpoint is not RuntimeBackend.delete_checkpoint
        )

    async def checkpoint(self) -> str:
        """Save the box's filesystem and return the checkpoint's id once it can be
        restored (`restore_config`). Files survive; running processes do not. A
        checkpoint that fails or is cancelled before it is restorable is not left
        behind for the caller to clean up. Recording it on a trace is the caller's
        (`Rollout.checkpoint`)."""
        raise NotImplementedError(
            f"{type(self).__name__} has no filesystem checkpoints"
        )

    async def checkpoint_sizes(self, checkpoint_ids: list[str]) -> dict[str, int]:
        """The stored bytes of each of `checkpoint_ids` the provider still holds and
        reports a size for; ids it does not hold are left out. Default: none known."""
        return {}

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint this backend's provider holds (a `checkpoint()` id).
        Works on a backend that was never started or is already torn down: the
        checkpoint outlives its box. A checkpoint that no longer exists counts as
        deleted. Raises `SandboxError` when the provider refuses the delete."""
        raise NotImplementedError(
            f"{type(self).__name__} has no filesystem checkpoints"
        )

    @abstractmethod
    async def exec(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        """Start `argv` as `runtime.user` in the working directory, with
        `runtime.process_env(env)`, and return a handle that streams its stdio."""

    async def run(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        """Run `argv` to completion. Override when the provider has a cheaper or sturdier
        blocking call; the default drains an `exec`, with stdin closed."""
        process = await self.exec(
            runtime, ["sh", "-c", 'exec "$@" < /dev/null', "sh", *argv], env
        )
        try:
            stdout, stderr, exit_code = await asyncio.gather(
                _drain(process.stdout), _drain(process.stderr), process.wait()
            )
        except BaseException:
            with contextlib.suppress(Exception):
                await run_shielded(process.kill())
            raise
        return ProgramResult(
            exit_code, stdout.decode(errors="replace"), stderr.decode(errors="replace")
        )

    async def run_background(
        self, runtime: Runtime, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        """Start `argv` with its combined output in `log`, a path in the working
        directory, and return; it runs until teardown ends the runtime. The default
        keeps an `exec` handle open for the process's life. Override when the provider
        runs detached jobs, so a long-lived server holds no client connection."""
        script = f'exec "$@" > {shlex.quote(log)} 2>&1 < /dev/null'
        process = await self.exec(runtime, ["sh", "-c", script, "sh", *argv], env)
        # Prune in place: another start can append while `poll` awaits.
        self._background.append(process)
        for handle in list(self._background):
            if await handle.poll() is not None and handle in self._background:
                self._background.remove(handle)

    async def read(
        self, runtime: Runtime, path: str, max_bytes: int | None = None
    ) -> bytes:
        """Read at most `max_bytes` bytes at the source, or the whole file if None.

        Overrides must accept `max_bytes` and enforce it before transferring data.
        `Runtime.read` checks for overflow using one extra byte. The default reads
        through `run`, base64-encoded because `run` returns decoded text.
        """
        # Through a temp file, not a pipe: `head | base64` exits with base64's 0
        # even when the path is missing, and a missing file must raise here just
        # as it does from a native binary read.
        result = await runtime.run(
            [
                "sh",
                "-c",
                (
                    "t=$(mktemp) || exit 1; "
                    'if [ -n "$1" ]; then head -c "$1" -- "$2"; else cat -- "$2"; fi '
                    '> "$t" || { rm -f "$t"; exit 1; }; '
                    'base64 < "$t"; rc=$?; rm -f "$t"; exit $rc'
                ),
                "sh",
                "" if max_bytes is None else str(max_bytes),
                path,
            ],
            {},
        )
        if result.exit_code:
            detail = result.stderr.strip()[-500:]
            raise SandboxError(
                f"read {path!r}: {detail}", kind=missing_file_kind(detail, "io")
            )
        return base64.b64decode(result.stdout)

    @abstractmethod
    async def write(self, runtime: Runtime, path: str, data: bytes) -> None:
        """Write `data` to `path`, creating parent directories."""

    def host_url(self, url: str) -> str:
        """The URL a program inside this runtime uses to reach a host-bound `url`."""
        return url

    async def isolate_processes(self) -> None:
        """Give this box's later commands a PID namespace of their own, as a container's
        processes have, where the box itself shows them more (its own init and daemons).
        A container runtime's processes already have one: nothing to do."""

    async def prepare_setup(self) -> None:
        """Open egress for trusted setup when reusing a restricted runtime."""
        if not self.network_restricted:
            return
        if self._setup_claimed:
            await self.prepare_execution(None)
        self._setup_claimed = True

    async def prepare_execution(
        self, routes: list[str] | None, network: NetworkPolicyConfig | None = None
    ) -> None:
        """Last setup step, right before the agent starts. Restricted runtimes enforce
        their policy here, or `network` instead of it; `routes` identifies the
        interception and MCP endpoints. None restores unrestricted egress for another
        trusted setup phase."""

    @property
    def network_restricted(self) -> bool:
        """Whether the agent phase uses filtered networking (see `prepare_execution`)."""
        return (
            isinstance(self.config, NetworkPolicyConfig)
            and self.config.network_restricted
        )

    @property
    def published_port(self) -> int | None:
        """A fixed port this runtime exposes to the outside at startup, declared up front to the
        provider (some forward only ports named at creation). When set, a server placed
        here binds it instead of a host-chosen free port, and `expose` returns its public URL.
        `None` for runtimes whose services already sit on host loopback (subprocess),
        which pick a free port."""
        return None

    async def expose(self, port: int) -> str:
        """Publish a port running *inside this runtime* to a URL reachable from the host/outside.
        A runtime on the host network is reached at host loopback as is; a sandbox overrides
        this with its port exposure (e.g. a container engine's published port), torn
        down with the sandbox in `stop()`. The reverse of a
        host `Tunnel` (interception.tunnel, which reaches a host port from inside a runtime)."""
        return f"http://127.0.0.1:{port}"


async def _drain(stream: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in stream])
