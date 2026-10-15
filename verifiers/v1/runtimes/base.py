"""RuntimeBackend ownership and provider operations."""

from __future__ import annotations

import asyncio
import atexit
import base64
import contextlib
import logging
import uuid
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic_config import BaseConfig

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.utils.aio import run_shielded

if TYPE_CHECKING:
    from verifiers.v1.runtimes.runtime import Runtime

logger = logging.getLogger(__name__)

# Ensure `uv` supports both script commands used below. Reuse a compatible installation;
# upgrade older binaries bundled in task images. Prefer pip on
# Python images, then fall back to the standalone installer (curl/wget), installing curl + CA
# certs when a bare image has no downloader. Both install paths land in ~/.local/bin, which we
# prepend to PATH first. (Installing needs network + one of pip / curl / wget / apt-get / apk.)
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
    "uv_supports_scripts() { "
    "uv sync --help 2>/dev/null | grep -q -- '--script' "
    "&& uv python find --help 2>/dev/null | grep -q -- '--script'; }; "
    "uv_supports_scripts "
    "|| { pip install -q -U --user uv 2>/dev/null && hash -r && uv_supports_scripts; } "
    f"|| {{ {{ {_INSTALL_CURL}; {_DOWNLOAD_UV}; }} && hash -r && uv_supports_scripts; }}"
)

# The single port a sandbox runtime forwards out for a server hosted in it: a public URL on
# modal/prime, a host loopback port on the local container engines. A server placed in such
# a runtime binds this (on 0.0.0.0) and is reached at the URL `expose` returns.
SERVICE_PORT = 8000


@dataclass(frozen=True)
class ProgramResult:
    exit_code: int
    stdout: str
    stderr: str


class RuntimeProcess(ABC):
    """A live process whose stdio crosses a runtime boundary."""

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


class RuntimeBackend(ABC):
    """Provider implementation for one container, VM, or local execution environment.

    Holds shared resource state and implements lifecycle, networking, commands, and
    file access. Handles bind user/env defaults without duplicating that resource.
    """

    is_local: bool = True
    """Whether this runtime exchanges host-local URLs without a public tunnel. True for
    subprocess and docker; remote runtimes override to False and use a host `Tunnel`
    inward plus `expose` outward."""

    supports_compose: ClassVar[bool] = False
    """Whether the runtime can host a task's Docker Compose services."""

    supports_user: ClassVar[bool] = False
    """Whether handles can bind another existing user."""

    supports_expose: ClassVar[bool] = True
    """Whether `expose` can publish a port inside this runtime to the outside, so a
    server here is reachable from another runtime."""

    scripts_dir: ClassVar[str] = "/tmp/vf-scripts"
    """Digest-keyed PEP 723 scripts inside the runtime. Sandboxes own their `/tmp`;
    the host subprocess runtime overrides this with its per-user cache."""

    info: BaseRuntimeInfo

    @property
    def supports_live_processes(self) -> bool:
        """Whether `open_process()` is implemented for this runtime instance."""
        return type(self)._open_process is not RuntimeBackend._open_process

    def __init__(self, name: str | None = None) -> None:
        self.name = name or f"vf-{uuid.uuid4().hex[:12]}"
        # Per-run task values live on the runtime rather than its serializable config/info.
        # Explicit process values (model credentials, proxy settings, etc.) override these.
        self.env: dict[str, str] = {}
        self._uv_interpreters: dict[str, str] = {}
        self._uv_script_locks: dict[str, asyncio.Lock] = {}
        self._mcp_sources: set[str] = set()
        self._mcp_install_lock = asyncio.Lock()
        self._setup_claimed = False
        self._curl_ready = False
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
        await run_shielded(self.teardown())

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

    @abstractmethod
    async def _run(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        pass

    async def _run_program(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        """Run the harness's MAIN program — the rollout itself (a possibly long-lived, stateful,
        agentic run) — as opposed to the short idempotent infra ops (write / mv / install /
        provisioning) that go through `run`. No framework layer may replay this argv: doing so
        against the rollout's persistent trace would fork a duplicate branch. Provider SDKs may
        still retry individual safe transport operations underneath `run`."""
        return await runtime.run(argv, env)

    async def _open_process(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        """Start a live process for a rollout-scoped harness session."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support live processes"
        )

    async def _run_background(
        self, runtime: Runtime, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        """Start `argv` as a background process in the runtime (combined output to
        `log`, a path in the workspace) and return immediately. It runs until `stop()`
        tears the runtime down. Used to host a tool server colocated with the harness."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support run_background"
        )

    @abstractmethod
    async def _read(
        self, runtime: Runtime, path: str, max_bytes: int | None = None
    ) -> bytes:
        """Read at most `max_bytes` bytes at the source, or the whole file if None.

        Overrides must accept `max_bytes` and enforce it before transferring data.
        The public `read` method checks for overflow using one extra byte.
        Runtimes without bounded binary reads delegate capped reads here, using
        base64 because `run` returns decoded text.
        """
        assert max_bytes is not None
        # Through a temp file, not a pipe: `head | base64` exits with base64's 0
        # even when the path is missing, and a missing file must raise here just
        # as it does from a native binary read.
        result = await runtime.run(
            [
                "sh",
                "-c",
                (
                    "t=$(mktemp) || exit 1; "
                    'head -c "$1" -- "$2" > "$t" || { rm -f "$t"; exit 1; }; '
                    'base64 < "$t"; rc=$?; rm -f "$t"; exit $rc'
                ),
                "sh",
                str(max_bytes),
                path,
            ],
            {},
        )
        if result.exit_code:
            raise SandboxError(f"read {path!r}: {result.stderr.strip()[-500:]}")
        return base64.b64decode(result.stdout)

    @abstractmethod
    async def _write(self, runtime: Runtime, path: str, data: bytes) -> None:
        pass

    def host_url(self, url: str) -> str:
        """The URL a program inside this runtime uses to reach a host-bound `url`."""
        return url

    async def prepare_setup(self) -> None:
        """Open egress for trusted setup when reusing a restricted runtime."""
        if not self.network_restricted:
            return
        if self._setup_claimed:
            await self.prepare_execution(None)
        self._setup_claimed = True

    async def prepare_execution(self, routes: list[str] | None) -> None:
        """Last setup step, right before the agent starts. Restricted runtimes enforce
        their policy here; `routes` identifies the interception and MCP endpoints.
        None restores unrestricted egress for another trusted setup phase."""

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
        provider (Modal forwards only ports named at `Sandbox.create`). When set, a server placed
        here binds it instead of a host-chosen free port, and `expose` returns its public URL.
        `None` for runtimes whose services already sit on host loopback (subprocess),
        which pick a free port."""
        return None

    async def expose(self, port: int) -> str:
        """Publish a port running *inside this runtime* to a URL reachable from the host/outside.
        A runtime on the host network is reached at host loopback as is; a sandbox overrides
        this with its port exposure (modal `tunnels()`, prime `client.expose`, a container
        engine's published port), torn down with the sandbox in `stop()`. The reverse of a
        host `Tunnel` (interception.tunnel, which reaches a host port from inside a runtime)."""
        return f"http://127.0.0.1:{port}"
