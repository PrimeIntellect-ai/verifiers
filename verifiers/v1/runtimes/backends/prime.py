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
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, TypeVar
from urllib.parse import urlsplit

from prime_sandboxes import APIError
from prime_sandboxes.models import validate_egress_lists

from verifiers.v1.configs.runtime import NetworkPolicyConfig, PrimeConfig
from verifiers.v1.errors import (
    SandboxError,
    SandboxErrorKind,
    http_status,
    missing_file_kind,
    sandbox_error_kind,
)
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
    parse_gpu,
)
from verifiers.v1.runtimes.limiters import CreationBacklogFull, creation_limiter
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.aio import cancel_requested, run_shielded
from verifiers.v1.utils.artifacts import validate_runtime_mounts
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
_PUBLIC_DOMAIN = "sandbox-v2-prod.pinfra.io"
"""Domain of the public URLs `https://<port>-<sandbox-id>.<domain>` that a LAN network's
`internet` rule publishes ports at."""
_EXPOSE_TAG = "vf-exposed"

_LIVENESS_INTERVAL_SECONDS = 60
"""How often a running job is checked for a dead wrapper. The SDK's wrapper writes the
exit file; if a command inside the box kills the wrapper (`pkill -f`, `killall sh`),
the file never appears and polling alone would wait forever."""
_LIVENESS_MISSES = 2
"""Consecutive checks that must find the job dead before it is reported dead, so the
moment between the job's output closing and its exit file appearing is never one."""

_BOOT_WAIT_ATTEMPTS = 180
"""Creation-wait budget for a box booting from its image, in the SDK's polls (~355 s)."""
_SECONDS_PER_WAIT_ATTEMPT = 355 / 180
"""About how long one of the SDK's creation polls takes (`_BOOT_WAIT_ATTEMPTS` is ~355 s)."""
_RESTORE_WAIT_ATTEMPTS = 420
"""Creation-wait budget for a box restored from a checkpoint (~14 min): a restore
copies the checkpoint's data first, and a large box's takes longer than a boot."""
_MISSING_CWD = re.compile(r"cwd '([^']+)' does not exist")
"""The platform's refusal of a command whose working directory is not on the box."""

_CHECKPOINT_DEADLINE_SECONDS = 900
_EGRESS_DEADLINE_SECONDS = 300
"""How long a sandbox's egress policy may take to apply before the box is refused. Under
load the SDK's calls queue in the process for tens of seconds before they are sent."""
"""Wait for a checkpoint to become restorable. An incremental checkpoint takes seconds;
the first of a box holding several GB can take minutes."""

_NO_ROUTE = r"""
set -u
saved=/run/vf-egress-gateway
# The platform's own addresses stay routed: its agent endpoint, and the private ranges
# its control plane reaches the box from (an exec's replies leave by the default route).
KEEP="192.0.2.0/24 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 100.64.0.0/10"
gateway() {  # the default route's gateway, from /proc/net/route's little-endian hex
  g=$(awk 'NR > 1 && $2 == "00000000" && $3 != "00000000" { print $3; exit }' /proc/net/route)
  [ -n "$g" ] || return 0
  printf '%d.%d.%d.%d\n' "0x${g#??????}" "0x$(echo "$g" | cut -c5-6)" \
    "0x$(echo "$g" | cut -c3-4)" "0x$(echo "$g" | cut -c1-2)"
}
route() {  # route <block|open> <gateway>
  if command -v ip >/dev/null 2>&1; then
    if [ "$1" = block ]; then
      for net in $KEEP; do ip route replace "$net" via "$2" || return 1; done
      ip route replace unreachable default
    else
      ip route replace default via "$2"
    fi
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$1" "$2" $KEEP <<'PY'
import fcntl, socket, struct, sys
def addr(ip):
    return struct.pack("HH4s8x", socket.AF_INET, 0, socket.inet_aton(ip))
def change(request, dst, mask, flags, gw="0.0.0.0"):
    # x86-64 `struct rtentry`: the three sockaddrs at 8, flags at 56; 120 bytes.
    route = bytearray(120)
    route[8:56] = addr(dst) + addr(gw) + addr(mask)
    route[56:58] = struct.pack("H", flags)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            fcntl.ioctl(s, request, route)
        except (FileExistsError, ProcessLookupError):
            pass
UP, GATEWAY, HOST, REJECT = 0x1, 0x2, 0x4, 0x200
ADD, DELETE = 0x890B, 0x890C
mode, gw = sys.argv[1:3]
if mode == "block":
    for net in sys.argv[3:]:
        dst, bits = net.split("/")
        mask = (0xFFFFFFFF << (32 - int(bits))) & 0xFFFFFFFF
        flags = UP | GATEWAY | (HOST if bits == "32" else 0)
        change(ADD, dst, socket.inet_ntoa(struct.pack("!I", mask)), flags, gw)
    change(DELETE, "0.0.0.0", "0.0.0.0", UP | GATEWAY, gw)
    try:
        change(ADD, "0.0.0.0", "0.0.0.0", UP | REJECT)
    except OSError:
        change(ADD, "0.0.0.0", "0.0.0.0", UP | GATEWAY, gw)
        raise
else:
    change(DELETE, "0.0.0.0", "0.0.0.0", UP | REJECT)
    change(ADD, "0.0.0.0", "0.0.0.0", UP | GATEWAY, gw)
PY
  elif command -v perl >/dev/null 2>&1; then
    perl - "$1" "$2" $KEEP <<'PL'
use strict; use Socket;
my ($mode, $gw, @keep) = @ARGV;
my ($UP, $GATEWAY, $HOST, $REJECT, $ADD, $DELETE) = (1, 2, 4, 0x200, 0x890B, 0x890C);
socket(my $s, AF_INET, SOCK_DGRAM, 0) or die "socket: $!";
sub addr { pack("S n a4 x8", AF_INET, 0, inet_aton($_[0])) }
sub change {  # x86-64 `struct rtentry`, as the Python path packs it
  my ($request, $dst, $mask, $flags, $via) = @_;
  my $route = pack("x8 a16 a16 a16 S x62", addr($dst), addr($via // "0.0.0.0"),
                   addr($mask), $flags);
  ioctl($s, $request, $route) or $!{EEXIST} or $!{ESRCH} or die "route: $!";
}
if ($mode eq "block") {
  for (@keep) {
    my ($dst, $bits) = split m{/};
    my $mask = inet_ntoa(pack("N", (0xFFFFFFFF << (32 - $bits)) & 0xFFFFFFFF));
    change($ADD, $dst, $mask, $UP | $GATEWAY | ($bits == 32 ? $HOST : 0), $gw);
  }
  change($DELETE, "0.0.0.0", "0.0.0.0", $UP | $GATEWAY, $gw);
  eval { change($ADD, "0.0.0.0", "0.0.0.0", $UP | $REJECT); 1 }
    or do { change($ADD, "0.0.0.0", "0.0.0.0", $UP | $GATEWAY, $gw); die $@ };
} else {
  change($DELETE, "0.0.0.0", "0.0.0.0", $UP | $REJECT);
  change($ADD, "0.0.0.0", "0.0.0.0", $UP | $GATEWAY, $gw);
}
PL
  else
    echo "no ip, python3 or perl in this image" >&2; return 1
  fi
}
if [ "$1" = block ]; then
  gw=$(cat "$saved" 2>/dev/null || gateway)
  [ -n "$gw" ] || { echo "no default route" >&2; exit 1; }
  printf '%s\n' "$gw" > "$saved" && route block "$gw"
elif [ -f "$saved" ]; then
  route open "$(cat "$saved")" && rm -f "$saved"
fi
"""
"""The in-box half of a deny-all policy. The platform's filter is a proxy that accepts
each TCP handshake and resets at the first byte, so `connect()` to any address succeeds
where Docker's `network_mode: none` (Harbor's no-network) refuses it. Replacing the box's
default route with an unreachable one refuses it here too (ENETUNREACH), keeping routes
to the platform's agent endpoint and private ranges; a later open policy restores the
saved gateway. Uses `ip`, else Python's or Perl's route ioctls; with none of them
the platform's filter is left alone."""

IMAGE_RUN = (
    '{ [ "$(stat -f -c %T /run)" = tmpfs ] && [ -d /run ] && m=$(mktemp -d) && '
    'mount --bind / "$m" && { [ -d "$m/run" ] && [ ! -L "$m/run" ] && '
    'rm -f "$m/run/vf-egress-gateway" "$m/run/vf-pid-namespace" '
    '"$m/run/vf-pid-namespace.parent" && cp -a /run/. "$m/run/" && '
    'mount --bind "$m/run" /run; umount "$m"; }; rmdir "$m"; } >/dev/null 2>&1;'
)
"""Give back what the image keeps under /run (a service's socket directory, a checker's
private work directory, made at build time), which the tmpfs Prime mounts there hides,
as a container started from the image would have it: the image's /run is still on the
root filesystem, under the mount, and a non-recursive bind of / reaches it. That
directory itself is bound over the tmpfs, as Docker leaves it: what is under it is
plain directories on the disk, which a grader can `rm -rf` and recreate (binding each
one made it a mount point, EBUSY), and a checker that copies a workspace into its /run
directory has the disk's room, not the tmpfs's few hundred MB. What Prime itself put
in the tmpfs is copied over. /run then lives on the box's disk, so a checkpoint keeps
it, as Docker keeps /run across a container's restart; the box's own state there that
describes the box it was written on (the saved egress gateway, the PID namespace's
init) is dropped first. A /run that is no longer a tmpfs (already bound) is left as it
is; a box without `mount` keeps the empty tmpfs."""

_PID_NAMESPACE_FILE = "/run/vf-pid-namespace"
_REAPER = """
import os, signal
for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
    signal.signal(sig, signal.SIG_IGN)
def reap(*_):
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return
signal.signal(signal.SIGCHLD, reap)
reap()
while True:
    signal.pause()
"""
"""The namespace's PID 1 where the box has Python: it reaps orphans, as the box's own
init does, so a grader that kills a process and checks it is gone does not find its
zombie; it has no children of its own a grader could kill. Elsewhere it is Harbor's
`sleep infinity`."""
_PID_NAMESPACE = rf"""
set -u
f={_PID_NAMESPACE_FILE}
REAPER={shlex.quote(_REAPER)}
p=$(cat "$f" 2>/dev/null) && [ -n "$p" ] && [ -e "/proc/$p/ns/pid" ] && \
  [ "$(readlink "/proc/$p/ns/pid")" != "$(readlink /proc/self/ns/pid)" ] && echo "$p" && exit 0
command -v unshare >/dev/null 2>&1 && command -v nsenter >/dev/null 2>&1 || {{
  echo "no unshare/nsenter in this image" >&2; exit 3; }}
rm -f "$f" "$f.parent"
setsid sh -c 'echo $$ > "$0.parent"; exec unshare --pid --fork --mount-proc \
  sh -c "command -v python3 >/dev/null 2>&1 && exec python3 -I -S -c \"\$0\"; exec sleep infinity" \
  "$1"' "$f" "$REAPER" </dev/null >/dev/null 2>&1 &
for i in $(seq 1 100); do
  u=$(cat "$f.parent" 2>/dev/null)
  c=$( [ -n "$u" ] && cat "/proc/$u/task/$u/children" 2>/dev/null | awk '{{print $1}}')
  [ -n "$c" ] || c=$( [ -n "$u" ] && awk -v u="$u" '$4 == u {{print $1; exit}}' \
    /proc/[0-9]*/stat 2>/dev/null)
  if [ -n "$c" ] && [ -e "/proc/$c/ns/pid" ] && \
     [ "$(readlink "/proc/$c/ns/pid")" != "$(readlink /proc/self/ns/pid)" ]; then
    printf '%s\n' "$c" > "$f" && chmod 600 "$f" && echo "$c" && exit 0
  fi
  sleep 0.1
done
echo "the PID namespace's init did not start" >&2; exit 1
"""
"""Start the box's PID namespace (`isolate_processes`): `unshare` in a session of its
own, its forked child the namespace's init, recorded for each command's `nsenter` and
printed."""

_IMAGE_WORKDIR = (
    "pwd; h=$(awk -F: '$1 == \"root\" {print $6; exit}' /etc/passwd 2>/dev/null); "
    "printf '%s\\n' \"${h:-$HOME}\""
)
"""The gateway's starting directory, and root's login directory, which the gateway
starts in when the image declares no WORKDIR: root's passwd entry, whatever the image's
HOME says."""

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


def _redacted(text: str, env: dict[str, str]) -> str:
    """`text` with every env value of 8 or more characters replaced by `***`."""
    for value in sorted(
        {v for v in env.values() if len(v) >= 8}, key=len, reverse=True
    ):
        text = text.replace(value, "***")
    return text


class _JobDied(Exception):
    """A background job's wrapper died before it wrote the job's exit status."""


_WRAPPER_LOST = "exited without recording an exit code"
"""The SDK's report (an untyped `APIError`) of a background job whose status reads
`lost`: its wrapper is gone and no exit file was written."""


def _wrapper_lost(error: BaseException) -> bool:
    """Whether the SDK reported the job's wrapper lost. Like `_JobDied`, the platform's
    failure, not the command's: an `io` outage, so the caller retries the stage and its
    calls continue from their latest saves on the outage budget instead of failing it."""
    return isinstance(error, APIError) and str(error).endswith(_WRAPPER_LOST)


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


_TERMINAL_CHECKPOINT = re.compile(
    r"\bcheckpoint(?:\s+[\w-]+)?\s+(?:(?:is|was)\s+|has\s+been\s+)?(?:deleting|deleted|expired)\b",
    re.IGNORECASE,
)


def checkpoint_missing(error: BaseException) -> bool:
    """A restore create refused a gone checkpoint, including its terminal 409 state.
    Other conflicts (in use, not ready yet), quota and account errors stand as-is.
    The caller must distinguish a refused create from an already-created box."""
    status = http_status(error)
    if status in (404, 410):
        return True
    if status not in (None, 409) or sandbox_error_kind(error) != "unknown":
        return False
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if http_status(current) != 409:
            current = current.__cause__ or current.__context__
            continue
        messages = [str(current)]
        response = getattr(current, "response", None)
        try:
            body = response.json() if response is not None else None
        except ValueError:
            body = None
        if isinstance(body, dict):
            detail = body.get("detail")
            for value in (body, detail):
                if isinstance(value, dict) and str(
                    value.get("checkpoint_status", "")
                ).lower() in {"deleting", "deleted", "expired"}:
                    return True
            messages.extend(
                body[key]
                for key in ("detail", "message", "error")
                if isinstance(body.get(key), str)
            )
        if any(_TERMINAL_CHECKPOINT.search(message) for message in messages):
            return True
        current = current.__cause__ or current.__context__
    return False


@dataclass
class _SharedClient:
    client: Any
    leases: int = 0


_STATS_INTERVAL_SECONDS = 60.0
"""How often the host-side Prime call timings are summarized (`_CallStats`)."""
stats_logger = logging.getLogger(__name__ + ".stats")
stats_logger.setLevel(logging.INFO)


class _CallStats:
    """How long this process's Prime calls take, by phase, summarized in one log line a
    minute: an exec's start request, its status polls, the poll that brings its output
    (which waits in the SDK's output pool), the exec as a whole, and a read's size check
    and download. The line also counts the live boxes and the output pool's queue."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.since = time.monotonic()

    def record(self, phase: str, seconds: float) -> None:
        self.samples[phase].append(seconds)
        now = time.monotonic()
        if now - self.since >= _STATS_INTERVAL_SECONDS:
            self.flush(now)

    def flush(self, now: float) -> None:
        parts = []
        for phase, values in sorted(self.samples.items()):
            values.sort()
            p50 = values[len(values) // 2]
            p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
            parts.append(
                f"{phase} n={len(values)} p50={p50:.2f}s p95={p95:.2f}s max={values[-1]:.2f}s"
            )
        boxes = sum(shared.leases for shared in _shared_clients.values())
        pending = inflight = 0
        for shared in _shared_clients.values():
            pool = getattr(shared.client, "_background_job_output_coordinator", None)
            pending += getattr(pool, "_pending_count", 0)
            inflight += len(getattr(pool, "_inflight", ()))
        stats_logger.info(
            "prime calls over %.0fs: %s; live boxes %d, output pool %d queued %d reading "
            "(%d workers)",
            now - self.since,
            "; ".join(parts) or "none",
            boxes,
            pending,
            inflight,
            _OUTPUT_READS,
        )
        self.samples.clear()
        self.since = now


_call_stats = _CallStats()


# One `AsyncSandboxClient` per event loop, leased by every live runtime on it. The SDK
# coalesces concurrent status polls per client into batched requests (up to 100 ids
# each), so sharing a client turns N runtimes' creation/job polls into a few batch
# calls. Keyed by loop (not plain process-global) so a client's tasks stay on the loop
# that created it.
_shared_clients: dict[asyncio.AbstractEventLoop, _SharedClient] = {}

_discards: set[asyncio.Task] = set()
"""Background deletes of abandoned checkpoints, held so they are not collected."""


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


def run_labels() -> list[str]:
    """The labels every Prime sandbox and tunnel this process opens carries, whatever its
    config adds: the base labels and the run scope. ``runtimes.sweep`` deletes by them."""
    return list(dict.fromkeys([*BASE_LABELS, run_scope()]))


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


HOSTS_ENTRY = (
    'h=$(hostname); grep -qE "[[:space:]]$h([[:space:]]|$)" /etc/hosts 2>/dev/null'
    ' || { printf "127.0.1.1\\t%s\\n" "$h" >> /etc/hosts; } 2>/dev/null || true'
)
"""Add the box's hostname to /etc/hosts when it is missing (never fails the start)."""

NVLINK_FABRIC = r"""
import ctypes, os, platform, re, subprocess, sys, time

def fabric():
    try:
        out = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    states = re.findall(r"^\s*Fabric\s*\n\s*State\s*:\s*(.+?)\s*$", out, re.M)
    return [s for s in states if s not in ("N/A", "Completed")]

stuck = fabric()
if not stuck:
    print("nvlink: fabric", "unknown" if stuck is None else "ready")
    sys.exit(0)
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
INIT_MODULE = {"x86_64": 175, "aarch64": 105}[platform.machine()]
directory = "/usr/local/nvidia/modules"
loaded = [l.split()[0] for l in open("/proc/modules") if l.startswith("nvidia")]
# Dependents first: every nvidia module but the core one depends on it.
for name in sorted(loaded, key=lambda n: n == "nvidia"):
    for _ in range(30):
        if libc.delete_module(name.encode(), 0) == 0:
            break
        time.sleep(1)
    else:
        sys.exit(f"nvlink: cannot unload {name}: {os.strerror(ctypes.get_errno())}")
for name in sorted(loaded, key=lambda n: n != "nvidia"):
    image = open(f"{directory}/{name.replace('_', '-')}.ko", "rb").read()
    params = b"NVreg_NvLinkDisable=1" if name == "nvidia" else b""
    if libc.syscall(ctypes.c_long(INIT_MODULE), ctypes.create_string_buffer(image), ctypes.c_ulong(len(image)), ctypes.c_char_p(params)):
        sys.exit(f"nvlink: cannot load {name}: {os.strerror(ctypes.get_errno())}")
left = fabric()
if left:
    sys.exit(f"nvlink: fabric still {left} after reloading the driver without NVLink")
print(f"nvlink: fabric was {stuck}; reloaded {loaded} with NVreg_NvLinkDisable=1")
"""
"""Make CUDA usable on a GPU box whose NVLink fabric never finishes initializing.

Prime's single-GPU H200 VMs pass through one SXM GPU of an NVSwitch host. The guest
driver loads with NVLink on, probes for a fabric the VM cannot reach, and stays at
fabric state "In Progress": `nvidia-smi` works, but every `cuInit` returns 802 (system
not yet initialized), so nothing can train or serve a model. Reloading the driver with
`NVreg_NvLinkDisable=1` leaves the lone GPU without a fabric to wait for (state N/A),
and CUDA starts. A box whose fabric is ready, or that has no NVIDIA GPU, is left as it
is."""


class PrimeBackend(RuntimeBackend):
    SUPPORTS_TASK_IMAGES: ClassVar[bool] = True
    SUPPORTS_COMPOSE: ClassVar[bool] = True
    SUPPORTS_USER_SWITCHING: ClassVar[bool] = True
    SUPPORTS_PORT_EXPOSURE: ClassVar[bool] = True

    def __init__(self, config: PrimeConfig, name: str | None = None) -> None:
        ensure_prime_auth()
        super().__init__(name)
        self.config = config
        self.info = PrimeRuntimeInfo(**self.config.model_dump())
        self._client = None
        self._routes_blocked = False
        """`_route_egress` refused outbound connections in the box."""
        self._drops_privileges: bool | None = None
        """Whether this box can run a user's command through `_as_user` (None: not yet
        asked)."""
        self._pid_namespace = False
        """Whether `isolate_processes` gave this box's commands a PID namespace."""
        self._namespace_root: str | None = None
        """The namespace's root as the file gateway reaches it (`/proc/<init>/root`):
        the namespace has a mount table of its own, and what is mounted there (an agent's
        `mount --bind` over a directory) is invisible from outside it."""
        self.placed = False
        """The box has answered an exec: a placement lost from now on is the box lost,
        not an outage."""
        # A LAN network (id, tags) to join as the sandbox boots: prime-sandboxes joins it
        # once it is running, within `wait_for_creation` on the creating client.
        self.lan: tuple[str, list[str]] | None = None
        # Exposure state: a LAN network created on the first `expose`, separate from
        # `lan`, and the ports its `internet` rule publishes.
        self._expose_network_id: str | None = None
        self._exposed_ports: list[int] = []
        self._expose_lock = asyncio.Lock()

    @property
    def published_port(self) -> int:
        # Any port can be exposed, but the public URL reaches only servers bound to
        # 0.0.0.0, which `launch` arranges for a fixed published port.
        return SERVICE_PORT

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
        if self.config.mounts:
            options["volumes"] = [
                {"mount_path": target, **mount.model_dump()}
                for target, mount in self.config.mounts.items()
            ]
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
                    for volume in dict.fromkeys(
                        m.name for m in self.config.mounts.values()
                    ):
                        try:
                            await self._client.volumes.create(volume)
                        except APIError as error:
                            # Create returns 409 when this scoped name already
                            # exists, including when another runtime creates it
                            # concurrently.
                            if not str(error).startswith("HTTP 409:"):
                                raise
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
                if self.config.workdir is None:
                    cwd = await self._held(
                        "image workdir",
                        lambda: self._client.execute_command(
                            self.info.id, _IMAGE_WORKDIR
                        ),
                    )
                    lines = cwd.stdout.strip().splitlines()
                    if (
                        cwd.exit_code != 0
                        or len(lines) != 2
                        or not all(path.startswith("/") for path in lines)
                    ):
                        raise SandboxError(
                            f"Cannot resolve image working directory: {cwd.stderr}",
                            kind="exec",
                        )
                    workdir, root_home = lines
                    # The gateway uses root's home (its passwd entry, whatever the
                    # image's HOME says) when the image has no WORKDIR, where Docker
                    # uses /. Do not bind that private directory to every user's
                    # commands.
                    if workdir == root_home:
                        workdir = "/"
                    self.config = self.config.model_copy(update={"workdir": workdir})
                self.info.workdir = self.config.workdir
                prepared = await self._held(
                    "workdir",
                    lambda: self._client.execute_command(
                        self.info.id,
                        f"{IMAGE_RUN} mkdir -p {shlex.quote(self.config.workdir)}"
                        f" && mkdir -p -m 1777 /run/lock && {HOSTS_ENTRY}",
                        working_dir="/",
                    ),
                )
                # Prime boots /run as an empty tmpfs, including on checkpoint restore,
                # over what the image put there (`IMAGE_RUN` copies it back). Restore the
                # standard lock directory before image service launchers run.
                # Existing directories retain their permissions (mkdir -p -m only applies
                # the mode when creating a directory). The box's own hostname resolves
                # locally, as in Docker: Prime's /etc/hosts does not list it, so without a
                # network every lookup of it (each `sudo`) waits 20 s on DNS.
                if prepared.exit_code:
                    raise SandboxError(
                        f"Cannot prepare runtime directories: {prepared.stderr}",
                        kind="exec",
                    )
                if gpu_count:
                    await self._ready_gpu()
                await validate_runtime_mounts(self.bind(), [])
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
            kind = sandbox_error_kind(e)
            if (
                self.config.checkpoint
                and self.info.id is None
                and not isinstance(e, SandboxError)
                and checkpoint_missing(e)
            ):
                # The create itself, which named the checkpoint, was refused as not
                # found, deleted or deleting: it cannot restore this checkpoint. A 404 once the
                # box exists (a box reaped while booting) is about the box.
                kind = "checkpoint_missing"
                raise SandboxError(
                    f"prime sandbox provisioning failed: checkpoint "
                    f"{self.config.checkpoint} is gone (deleted or expired): {e}",
                    kind=kind,
                ) from e
            raise SandboxError(
                f"prime sandbox provisioning failed: {e}", kind=kind
            ) from e

    async def _boot(self) -> None:
        """Wait for the created box to boot. The SDK's readiness probe runs in the
        image's WORKDIR, so a box restored from a checkpoint whose filesystem lacks it
        (a run on the box deleted it) never passes the probe: the directory is made
        again from / and the wait runs once more."""

        attempts = (
            _RESTORE_WAIT_ATTEMPTS if self.config.checkpoint else _BOOT_WAIT_ATTEMPTS
        )
        if self.config.create_timeout_s:
            # A box that may queue for capacity (a GPU VM) waits as long as the creation
            # may take, not only the default boot budget.
            attempts = max(
                attempts, int(self.config.create_timeout_s / _SECONDS_PER_WAIT_ATTEMPT)
            )

        def wait() -> Any:
            return self._client.wait_for_creation(self.info.id, max_attempts=attempts)

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

    async def _ready_gpu(self) -> None:
        """Bring up CUDA on a GPU box whose NVLink fabric is stuck (`NVLINK_FABRIC`),
        before anything runs on it. A box it cannot fix is refused: every CUDA call
        there fails, so a rollout or a grader on it could only waste its time."""
        ready = await self._held(
            "gpu",
            lambda: self._client.execute_command(
                self.info.id,
                "command -v python3 >/dev/null || exit 0; "
                f"python3 -I -c {shlex.quote(NVLINK_FABRIC)}",
                working_dir="/",
                timeout=600,
            ),
        )
        output = f"{ready.stdout}{ready.stderr}".strip()
        if ready.exit_code:
            raise SandboxError(
                f"prime sandbox {self.info.id}: GPU unusable: {output}", kind="exec"
            )
        if "reloaded" in output:
            logger.warning("prime: sandbox %s: %s", self.info.id, output)

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
                async with asyncio.timeout(_EGRESS_DEADLINE_SECONDS):
                    delay = 0.1
                    while not status.applied:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 3)
                        status = await self._client.get_network(self.info.id)
            except TimeoutError as e:
                raise SandboxError(
                    f"prime egress policy was not applied within {_EGRESS_DEADLINE_SECONDS}s "
                    "on sandbox "
                    f"{self.info.id}; refusing to start the agent unrestricted",
                    kind=sandbox_error_kind(e),
                ) from e
        except SandboxError:
            raise
        except Exception as e:
            raise SandboxError(
                f"prime egress policy failed: {e}", kind=sandbox_error_kind(e)
            ) from e
        await self._route_egress(policy.get("deny") == ["*"])
        logger.info(
            "prime: egress policy applied on sandbox %s (allow=%s block=%s)",
            self.info.id,
            policy.get("allow"),
            policy.get("deny"),
        )

    async def _in_workdir(self, start: Callable[[], Awaitable[T]]) -> T:
        """`start()`, a command's launch, made again once the box's workdir exists:
        the platform refuses every command whose working directory is gone, so an
        agent or a script that removed the workdir (`rm -rf /app`, a replaced tree)
        otherwise ended the rollout at its next command, where Docker's container
        keeps its WORKDIR's mount point."""
        try:
            return await start()
        except Exception as e:
            missing = _MISSING_CWD.search(str(e))
            if missing is None or missing.group(1) != self.config.workdir:
                raise
            logger.warning(
                "prime: sandbox %s lost its working directory %s; making it again",
                self.info.id,
                missing.group(1),
            )
            made = await self._client.execute_command(
                self.info.id,
                f"mkdir -p {shlex.quote(missing.group(1))}",
                working_dir="/",
            )
            if made.exit_code:
                raise
            return await start()

    async def _route_egress(self, block: bool) -> None:
        """Refuse outbound connections in the box itself under a deny-all policy, and
        lift that once a policy opens it again (`_NO_ROUTE`). Best effort: the
        platform's filter still holds every destination either way."""
        if not block and not self._routes_blocked:
            return
        result = await self.run(
            Runtime(self),
            ["sh", "-c", _NO_ROUTE, "vf-egress", "block" if block else "open"],
            {},
        )
        if result.exit_code:
            logger.warning(
                "prime: in-box egress route %s failed on sandbox %s: %s",
                "block" if block else "restore",
                self.info.id,
                (result.stderr or result.stdout).strip()[-500:],
            )
        self._routes_blocked = block and not result.exit_code

    async def isolate_processes(self) -> None:
        """Give this box's later commands a PID namespace of their own, as a container's
        processes have: its first process is Harbor's `sleep infinity`, which no signal
        from inside ends, and every command runs inside it. A grader that kills every process
        but PID 1 and its own ancestors (Docker shows it nothing else) otherwise kills
        the platform's command daemon, and the box with it; one that refuses to grade
        while a kernel thread it cannot kill is listed sees none here either. Needs
        `unshare` and `nsenter` in the image; without them commands run as before."""
        if self._pid_namespace:
            return
        result = await self.run(Runtime(self), ["sh", "-c", _PID_NAMESPACE], {})
        self._pid_namespace = result.exit_code == 0
        if self._pid_namespace and (init := result.stdout.split()[-1:]):
            self._namespace_root = f"/proc/{init[0]}/root"
        if not self._pid_namespace:
            logger.warning(
                "prime: no PID namespace on sandbox %s: %s",
                self.info.id,
                (result.stderr or result.stdout).strip()[-300:],
            )

    async def _job(self, runtime: Runtime, command: str) -> tuple[str, str | None]:
        """`command` as the platform should start it, and the user to start it as: as
        the runtime's user (`_as_user`), inside the box's PID namespace once it has one
        (`isolate_processes`), and outside it should that namespace's init be gone."""
        command, user = await self._as_user(runtime, command)
        if not self._pid_namespace or user is not None:
            return command, user
        inner = shlex.quote(command)
        # Entering a mount namespace moves the working directory to its root: keep the
        # command's own, entered again by path inside the namespace. `--wd` alone keeps
        # the directory as found outside, where the namespace's mounts are not: relative
        # paths then missed a directory an agent had mounted over that absolute paths
        # (`Path.cwd()`) saw. The command gets a session of its own inside the namespace,
        # as a user's does (`_as_user`): one led from outside reads as session 0 there,
        # and a grader that stops a session it resolved to 0 stops the daemon's group.
        entered = (
            f'p=$(cat {_PID_NAMESPACE_FILE} 2>/dev/null); if [ -n "$p" ] && '
            f'[ -e "/proc/$p/ns/pid" ]; then exec nsenter -t "$p" -p -m --wd="$PWD" '
            f'-- sh -c \'cd -- "$0" 2>/dev/null; exec setsid -w sh -c "$1"\' '
            f'"$PWD" {inner}; fi; exec sh -c {inner}'
        )
        return entered, None

    async def _as_user(self, runtime: Runtime, command: str) -> tuple[str, str | None]:
        """`command` and the user the platform should start it as, for `runtime`'s user.

        The platform starts every command in its daemon's process group, and drops the
        box when that group holds a stopped process. A user's command started as that
        user puts its whole job there, so a grader that `pkill -STOP`s the agent's user
        (quiescing the agent before it grades) lost the box. Where the box has `setsid`
        and `setpriv`, the job starts as root and the command drops to the user in a
        session of its own, the way each `docker exec` gets one; otherwise the platform
        starts it as the user, as before."""
        user = runtime.user
        if user is None:
            return command, None
        if self._drops_privileges is None:
            try:
                probe = await self._client.execute_command(
                    self.info.id,
                    "command -v setsid >/dev/null && command -v setpriv >/dev/null",
                    timeout=60,
                )
                self._drops_privileges = probe.exit_code == 0
            except Exception as e:  # noqa: BLE001 - the platform's own user switch works too
                logger.debug("prime: privilege-drop probe failed: %s", e)
                return command, user
        if not self._drops_privileges:
            return command, user
        name = shlex.quote(user)
        home = (
            ""
            if runtime.home is not None
            else f"HOME=\"$(awk -F: -v u={name} '$1 == u {{print $6}}' /etc/passwd)\" "
        )
        dropped = (
            f'exec setsid -w setpriv --reuid={name} --regid="$(id -g {name})" '
            f"--init-groups -- env {home}USER={name} LOGNAME={name} "
            f"sh -c {shlex.quote(command)}"
        )
        return dropped, None

    async def run(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        try:
            # Poll directly so rollout cancellation owns the execution timeout.
            began = time.monotonic()
            command, user = await self._job(runtime, shlex.join(argv))
            job = await self._in_workdir(
                lambda: self._client.start_background_job(
                    self.info.id,
                    command,
                    working_dir=self.config.workdir,
                    env=runtime.process_env(env),
                    user=user,
                )
            )
            _call_stats.record("exec.start", time.monotonic() - began)
            delay = 0.1
            output_retries = 0
            missing = None
            loop = asyncio.get_running_loop()
            next_check = loop.time() + _LIVENESS_INTERVAL_SECONDS
            misses = 0
            while True:
                polled = time.monotonic()
                result = await self._held(
                    "job poll",
                    lambda: self._client.get_background_job(
                        self.info.id, job, timeout=_OUTPUT_DEADLINE_SECONDS
                    ),
                )
                _call_stats.record(
                    "exec.output" if result.completed else "exec.poll",
                    time.monotonic() - polled,
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
                    alive = await self._job_alive(job, user)
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
            raise SandboxError(f"prime exec failed: {e}", kind="io") from None
        except Exception as e:  # noqa: BLE001 - a sandbox/API failure is one rollout's problem
            # the SDK's error quotes the command with its env exported inline (HF_TOKEN
            # and other secrets): neither the message nor its cause may carry them
            raise SandboxError(
                f"prime exec failed: {_redacted(str(e), runtime.process_env(env))}",
                kind="io" if _wrapper_lost(e) else self._kind(e, "exec"),
            ) from None
        _call_stats.record("exec", time.monotonic() - began)
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
        """Whether a job without an exit status still runs: some process holds its output
        open, or its exit status has been written since. None when the check itself
        failed. Runs as the job's user, who can read the job's own file descriptors.

        An exit file that exists but is empty is not an exit status: a wrapper killed
        between creating the file and writing the code leaves one, and the job is then
        gone with nothing to report (TB-engineering 2026-10-25: two draws polled such a
        file for 14 hours). Only a non-empty file, or a live holder of the output, counts."""
        log = shlex.quote(job.stdout_log_file)
        command = (
            f"if [ -s {shlex.quote(job.exit_file)} ] || "
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

    async def exec(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        try:
            command, user = await self._job(runtime, shlex.join(argv))
            process = await self._in_workdir(
                lambda: self._client.open_process(
                    self.info.id,
                    command,
                    working_dir=self.config.workdir,
                    env=runtime.process_env(env),
                    user=user,
                )
            )
        except Exception as e:
            raise SandboxError(
                f"prime streaming process failed to start: {e}",
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

    async def checkpoint_sizes(self, checkpoint_ids: list[str]) -> dict[str, int]:
        """Stored bytes per checkpoint, from the platform's batch lookup (100 ids a
        request) on a short-lived client of its own; missing ids are left out."""
        from prime_sandboxes import AsyncSandboxClient

        sizes: dict[str, int] = {}
        async with AsyncSandboxClient() as client:
            for start in range(0, len(checkpoint_ids), 100):
                batch = list(dict.fromkeys(checkpoint_ids[start : start + 100]))
                found = await _repeat(lambda batch=batch: client.get_checkpoints(batch))
                for checkpoint in found.checkpoints:
                    if checkpoint.stored_bytes is not None:
                        sizes[checkpoint.id] = checkpoint.stored_bytes
        return sizes

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
        # A server-owned job runs without a client stream for its whole life.
        try:
            command, user = await self._job(
                runtime,
                f"exec {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null",
            )
            await self._in_workdir(
                lambda: self._client.start_background_job(
                    self.info.id,
                    command,
                    working_dir=self.config.workdir,
                    env=runtime.process_env(env),
                    user=user,
                )
            )
        except Exception as e:
            raise SandboxError(
                f"prime background launch failed: {e}",
                kind=self._kind(e, "exec"),
            ) from e

    async def read(
        self, runtime: Runtime, path: str, max_bytes: int | None = None
    ) -> bytes:
        # The file API acts as the default user; a user view reads through exec.
        if max_bytes is not None or runtime.user is not None:
            reader = "cat" if max_bytes is None else f"head -c {max_bytes}"
            try:
                # Stream binary output: execute_command buffers base64 text for the
                # entire file. Bound the source read and the host buffer independently.
                command, user = await self._job(
                    runtime, f"{reader} -- {shlex.quote(path)}"
                )
                process = await self._in_workdir(
                    lambda: self._client.open_process(
                        self.info.id,
                        command,
                        working_dir=self.config.workdir,
                        env=runtime.process_env({}),
                        user=user,
                    )
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
        began = time.monotonic()
        result = await self._held(
            "size",
            lambda: self._client.execute_command(
                self.info.id,
                f"wc -c < {shlex.quote(self._gateway(target))}",
                timeout=_OUTPUT_DEADLINE_SECONDS,
            ),
            **_TRANSFER_RETRIES,
        )
        _call_stats.record("read.size", time.monotonic() - began)
        if result.exit_code:
            detail = (result.stderr or "").strip()[-500:]
            raise SandboxError(detail, kind=missing_file_kind(detail, "io"))
        return int(result.stdout.strip())

    def _gateway(self, target: str) -> str:
        """`target` as the file gateway must name it to reach the file commands see:
        through the namespace's root once commands run in one (`isolate_processes`)."""
        return f"{self._namespace_root or ''}{target}"

    async def _upload(self, target: str, data: bytes) -> None:
        target = self._gateway(target)
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
        target = self._gateway(target)
        # The temporary file is removed on every exit, and its byte read stays off the event loop.
        with tempfile.TemporaryDirectory() as directory:
            download = Path(directory) / "download"
            began = time.monotonic()
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
            _call_stats.record("read.download", time.monotonic() - began)
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
                    box = self.info.id
                    await self._held("delete", lambda: client.delete(box))
            except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
                logger.warning(
                    "prime: failed to delete sandbox %s: %s", self.info.id, e
                )
            if self._expose_network_id is not None:
                try:
                    await client.delete_lan_network(self._expose_network_id)
                    self._expose_network_id = None
                except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
                    logger.warning(
                        "prime: failed to delete LAN network %s: %s",
                        self._expose_network_id,
                        e,
                    )
        finally:
            await _release_client(client)
