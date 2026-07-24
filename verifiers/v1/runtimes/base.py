"""Execution runtime contract."""

import asyncio
import atexit
import contextlib
import hashlib
import logging
import shlex
import uuid
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import ClassVar, Self

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

# Ensure the latest `uv` is available for our PEP 723 scripts: prefer pip on Python images,
# then fall back to the standalone installer (curl/wget), installing curl + CA certs when a
# bare image has no downloader. Both paths install to ~/.local/bin, which we prepend to PATH.
# (Needs network + one of pip / curl / wget / apt-get / apk.)
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
    "pip install -q -U --user uv 2>/dev/null "
    f"|| {{ {_INSTALL_CURL}; {_DOWNLOAD_UV}; }}"
)

# A reused restricted box must stop processes left by the prior untrusted agent
# before trusted setup reopens egress. PID + kernel start time survives PID reuse.
_PROCESS_SNAPSHOT = r"""
for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    stat=$(cat "$proc/stat" 2>/dev/null) || continue
    rest=${stat##*) }
    set -- $rest
    printf '%s:%s\n' "$pid" "${20}"
done
"""
_KILL_AFTER_SNAPSHOT = r"""
baseline="
$VF_PROCESS_BASELINE
"
protected=" "
pid=$$
while [ "$pid" -gt 1 ] 2>/dev/null; do
    protected="$protected$pid "
    stat=$(cat "/proc/$pid/stat" 2>/dev/null) || break
    rest=${stat##*) }
    set -- $rest
    pid=$2
done
while :; do
    targeted=
    for proc in /proc/[0-9]*; do
        pid=${proc##*/}
        case "$protected" in *" $pid "*) continue ;; esac
        stat=$(cat "$proc/stat" 2>/dev/null) || continue
        rest=${stat##*) }
        set -- $rest
        case "$1" in Z|X) continue ;; esac
        identity="$pid:${20}"
        case "$baseline" in *"
$identity
"*) continue ;; esac
        if kill -STOP "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
            targeted=1
        fi
    done
    [ -z "$targeted" ] && break
done
"""

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


# `stop()` frees a runtime's external resource on the normal path (the rollout's `finally`),
# shielded so a Ctrl-C / SIGTERM task cancellation can't cut it short. A second Ctrl-C
# raises KeyboardInterrupt out of the event loop itself — no task-level shield survives
# that — so runtimes are also tracked in `_LIVE` and freed by a *synchronous* `atexit`
# hook (`cleanup`), sync because the loop is gone at interpreter shutdown. SIGKILL runs
# none of this.
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


class NetworkPolicyConfig(BaseConfig):
    """Shared execution-time policy surface for runtimes that support it."""

    allow: list[str] = ["*"]
    """Destinations allowed during execution; `*` leaves egress unrestricted."""
    block: list[str] = []
    """Destinations denied during execution."""

    @property
    def network_restricted(self) -> bool:
        return "*" not in self.allow or bool(self.block)

    def with_task_network_policy(self, allow: list[str], block: list[str]) -> Self:
        if "*" not in allow:
            allow = (
                allow
                if "*" in self.allow
                else list(dict.fromkeys([*allow, *self.allow]))
            )
        else:
            allow = self.allow
        block = list(dict.fromkeys([*block, *self.block]))
        return type(self).model_validate(
            {**self.model_dump(), "allow": allow, "block": block}
        )


class BaseRuntimeInfo(BaseConfig):
    id: str | None = None
    borrowed: bool = False
    """Whether the run was placed into a live box owned by someone else
    (`Agent.run(runtime=...)`) rather than provisioning its own."""


class Runtime(ABC):
    is_local: ClassVar[bool] = True
    """Whether this runtime exchanges host-local URLs without a public tunnel. True for
    subprocess and Docker (directly or through Docker's policy proxy); remote runtimes
    override to False and use a host `Tunnel` inward plus `expose` outward."""

    info: BaseRuntimeInfo

    def __init__(self, name: str | None = None) -> None:
        self.name = name or f"vf-{uuid.uuid4().hex[:12]}"
        self._uv_interpreters: dict[str, str] = {}
        self._uv_script_locks: dict[str, asyncio.Lock] = {}
        self._setup_claimed = False
        self._process_baseline: str | None = None
        self.execution_prepared = False
        """Whether a rollout successfully activated this runtime's execution policy."""
        self.stopped = False
        """Whether teardown has begun (set by `stop`). A stopped runtime is dead: a rollout
        refuses to borrow one — the owner tore it down, so any use is a lifetime bug in the
        borrowing program, caught up front instead of failing opaquely mid-harness."""

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
    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        pass

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """Run the harness's MAIN program — the rollout itself (a possibly long-lived, stateful,
        agentic run) — as opposed to the short idempotent infra ops (write / mv / install /
        provisioning) that go through `run`. No framework layer may replay this argv: doing so
        against the rollout's persistent trace would fork a duplicate branch. Provider SDKs may
        still retry individual safe transport operations underneath `run`."""
        return await self.run(argv, env)

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        """Start `argv` as a background process in the runtime (combined output to
        `log`, a path in the workspace) and return immediately. It runs until `stop()`
        tears the runtime down. Used to host a tool server colocated with the harness."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support run_background"
        )

    async def prepare_uv_script(
        self,
        script: str | bytes,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        data = script.encode() if isinstance(script, str) else script
        digest = hashlib.sha256(data).hexdigest()
        path = f"/tmp/vf-scripts/{digest}.py"
        if digest not in self._uv_interpreters:
            async with self._uv_script_locks.setdefault(digest, asyncio.Lock()):
                if digest not in self._uv_interpreters:
                    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
                    await self.write(tmp, data)
                    command = (
                        f"mv -f {shlex.quote(tmp)} {shlex.quote(path)} "
                        f"&& {{ {_ENSURE_UV}; }} "
                        f"&& uv sync --script {shlex.quote(path)} -q --no-config "
                        f"&& uv python find --script {shlex.quote(path)} --no-config"
                    )
                    result = await self.run(["sh", "-c", command], env or {})
                    if result.exit_code != 0:
                        raise RuntimeError(
                            "failed to prepare uv script: "
                            f"{result.stderr.strip()[-2000:]}"
                        )
                    self._uv_interpreters[digest] = result.stdout.strip().splitlines()[
                        -1
                    ]
        interpreter = self._uv_interpreters[digest]
        venv = str(PurePosixPath(interpreter).parent.parent)
        command = (
            'export VIRTUAL_ENV="$1" PATH="${1}/bin:$HOME/.local/bin:$PATH" '
            'UV_INSTALL_DIR="$HOME/.local/bin" UV_RUN_RECURSION_DEPTH=1; '
            'shift; exec "$@"'
        )
        return [
            "sh",
            "-c",
            command,
            "uv-script",
            venv,
            interpreter,
            path,
        ]

    async def run_uv_script(
        self,
        script: str | bytes,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> ProgramResult:
        """The script is written to a stable, content-addressed path rather than the per-rollout
        workspace: uv keys its per-script environment by the script's full path, so a
        unique path per call would mint a fresh env every rollout. A path derived from the
        content means identical scripts share one path → uv reuses one env, bounded by the
        number of distinct scripts. Published via a unique temp + atomic `mv`, so
        concurrent rollouts writing the same content never race a half-written read."""
        argv = await self.prepare_uv_script(script, env)
        return await self.run([*argv, *(args or [])], env or {})

    @abstractmethod
    async def read(self, path: str) -> bytes:
        pass

    @abstractmethod
    async def write(self, path: str, data: bytes) -> None:
        pass

    def host_url(self, url: str) -> str:
        """The URL a program inside this runtime uses to reach a host-bound `url`."""
        return url

    @contextlib.asynccontextmanager
    async def rollout(self) -> AsyncIterator[None]:
        """Claim one restricted rollout, reopening trusted setup only between runs."""
        if not self.network_restricted:
            yield
            return
        if self._setup_claimed:
            raise SandboxError(
                f"network-filtered {self.type} runtime {self.name!r} is already in "
                "use; wait for its rollout to finish before reusing it"
            )
        self._setup_claimed = True
        try:
            if self._process_baseline is not None:
                result = await self.run(
                    ["sh", "-c", _KILL_AFTER_SNAPSHOT],
                    {"VF_PROCESS_BASELINE": self._process_baseline},
                )
                if result.exit_code != 0:
                    raise SandboxError(
                        f"failed to isolate reused {self.type} runtime processes: "
                        f"{result.stderr.strip()[-500:]}"
                    )
            snapshot = await self.run(["sh", "-c", _PROCESS_SNAPSHOT], {})
            if snapshot.exit_code != 0:
                raise SandboxError(
                    f"failed to snapshot {self.type} runtime processes: "
                    f"{snapshot.stderr.strip()[-500:]}"
                )
            self._process_baseline = snapshot.stdout
            if self.execution_prepared:
                await self._apply_network_policy(None)
                self.execution_prepared = False
            yield
        finally:
            self._setup_claimed = False

    async def prepare_execution(self, routes: list[str]) -> None:
        """Last setup step, right before the agent starts. Restricted runtimes enforce
        their policy here while keeping the interception and MCP `routes` reachable."""
        if not self.network_restricted:
            return
        await self._apply_network_policy(routes)
        self.execution_prepared = True

    async def _apply_network_policy(self, routes: list[str] | None) -> None:
        """Apply execution routes, or restore unrestricted trusted setup for None."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support restricted networking"
        )

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
        `None` for local runtimes (subprocess/docker), which pick a free port."""
        return None

    async def expose(self, port: int) -> str | None:
        """Publish a port running *inside this runtime* to a URL reachable from the host/outside,
        or None when local. A remote runtime overrides this with the provider's native port
        exposure (modal `tunnels()`, prime `client.expose`), torn down with the sandbox in
        `stop()`. The reverse of a host `Tunnel` (interception.tunnel, which reaches a host
        port from inside a runtime)."""
        return None
