"""The error model — every rollout failure is attributed to one boundary, then recorded once.

Four mechanisms, each in one place:

1. Vocabulary (this module): `RolloutError` and the flat boundary types below. Each names the
   boundary a failure crossed — provider, harness, toolset, sandbox, task, or
   interception — so a recorded `trace.last_error.type` says where the rollout broke.
2. Classification (`boundary`): the one helper that runs a framework→code boundary and attributes
   any escaping error to that boundary's type. Extension code (task hooks, harness subclasses)
   raises plain Python errors — it never constructs a `vf` error type; `boundary` classifies them.
   Infra that fails raises its type at the source (`runtimes` → `SandboxError`, `clients` →
   `ProviderError`, tunnels → `TunnelError`); an already-typed `RolloutError` passes through unchanged.
3. Surfacing (`session.RolloutSession.error`): a model or tool call fails behind the harness
   subprocess and comes back as HTTP, so the interception server stashes the real error there and
   the rollout re-raises it once the harness returns — not a secondary `HarnessError`.
4. Capture (`Rollout`, mirrored by the env-server): the one place that records a failure (typed
   or not) onto the trace and never lets it cancel sibling rollouts. A bad rollout is data, not a
   crash.

The detail (status code, stderr, ...) comes from the wrapped inner error; we add a type only when
the boundary isn't already clear from it.
"""

import contextlib
import errno
import re
from collections.abc import AsyncIterator, Iterator
from typing import Literal

from openai import OpenAIError


class RolloutError(Exception):
    """Base for a failure recorded onto the trace rather than crashing the rollout."""


class ProviderError(RolloutError):
    """A model-provider call failed (transport, HTTP status, timeout, or malformed response).
    `status_code` is the HTTP status surfaced to the harness so its SDK retries transient faults
    (5xx/429/timeout) and not deterministic ones (4xx) — relayed from the provider, or chosen for a
    transport fault."""

    def __init__(self, message: str = "", *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


MODEL_TRANSPORT_ERROR_EXIT_CODE = 97
"""Reserved harness exit code for an exhausted model transport failure.
`Harness._check_result` maps it to `InterceptionError`."""


class HarnessError(RolloutError):
    """The harness failed to install or launch, or its agent process exited unsuccessfully."""


class ToolsetError(RolloutError):
    """A task's `Toolset` could not be built or served."""


class EnvError(RolloutError):
    """The environment's own hooks failed — `run()` or `finalize()` raised (or
    ran no agent at all). Episode-level: per-agent failures stay typed on their
    traces. (Not `EnvironmentError` — that's a builtin alias of OSError.)"""


SandboxErrorKind = Literal[
    "account", "quota", "box_lost", "missing", "disk_full", "io", "exec", "unknown"
]
"""What a `SandboxError` means for the caller, set where the runtime raises it:

- `account`: billing or auth refused account-wide (payment required, unauthorized,
  forbidden). Retrying is pointless until someone fixes the account.
- `quota`: refused for capacity or rate (burst limits, 429, a full creation backlog).
  Retryable later.
- `box_lost`: the box itself is gone (terminated, out of memory, 410, its placement
  lost). Never a path: what may take a missing path as absent may not take this.
- `missing`: a path the operation named does not exist (ENOENT, the SDK's file not
  found, 404). The transfer worked; the box is there.
- `disk_full`: the box's disk is full (ENOSPC, 507). Repeating waits for nothing.
- `io`: a transfer or the platform failed in transit (a dropped connection, a timeout,
  a gateway 5xx, the platform unavailable). Retryable; an outage is held through.
- `exec`: a command could not be run in a live box.
- `unknown`: anything else (a request the box refuses, 413 or 422), and every error
  raised without a kind."""


class SandboxError(RolloutError):
    """A runtime/sandbox operation failed (provisioning, exec, or file I/O). `kind`
    (`SandboxErrorKind`) says what the failure means; the message is the detail."""

    def __init__(
        self, message: str = "", *, kind: SandboxErrorKind = "unknown"
    ) -> None:
        super().__init__(message)
        self.kind: SandboxErrorKind = kind


def error_kind(error: BaseException) -> SandboxErrorKind | None:
    """The kind a recorded `Error` carries: a `SandboxError`'s, unless `unknown`."""
    if isinstance(error, SandboxError) and error.kind != "unknown":
        return error.kind
    return None


_ACCOUNT_STATUSES = frozenset({401, 402, 403})
_ACCOUNT_TYPES = frozenset({"UnauthorizedError", "PaymentRequiredError"})
_ACCOUNT_MARKERS = (
    "payment required",
    "check billing status",
    "unauthorized",
    "invalid api key",
    "forbidden",
)
_QUOTA_MARKERS = ("too many requests", "rate limit", "burst limit", "quota exceeded")
_BOX_LOST_TYPES = frozenset(
    {"SandboxNotRunningError", "SandboxOOMError", "SandboxTimeoutError"}
)
_BOX_LOST_MARKERS = ("sandbox_not_found", "sandbox_terminated")
_STOPPED = frozenset({"ERROR", "TERMINATED", "TIMEOUT"})
"""`SandboxNotRunningError.status` values of a box that stopped for good."""
_IO_TYPES = frozenset(
    {
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "BrokenPipeError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "WriteError",
        "WriteTimeout",
        "RemoteProtocolError",
        "APITimeoutError",
        "UploadTimeoutError",
        "DownloadTimeoutError",
    }
)
_GATEWAY_MARKERS = (
    "bad gateway",
    "upstream unavailable",
    "failed to route",
    "(unavailable)",
)
"""A gateway in front of the provider failed the request, not the box: a repeat can
ride it out."""
_MISSING_FILE_TYPES = frozenset({"SandboxFileNotFoundError", "FileNotFoundError"})
_MISSING_FILE_MARKERS = ("no such file",)
_DISK_FULL_MARKERS = ("no space left on device",)
_HTTP_STATUS = re.compile(r"\bHTTP (\d{3})\b")
_ERRNO_KINDS: dict[int, SandboxErrorKind] = {
    errno.ENOSPC: "disk_full",
    errno.EDQUOT: "disk_full",
    errno.ENOENT: "missing",
    errno.ECONNRESET: "io",
    errno.ECONNREFUSED: "io",
    errno.ECONNABORTED: "io",
    errno.EPIPE: "io",
    errno.ETIMEDOUT: "io",
    errno.EHOSTUNREACH: "io",
    errno.ENETUNREACH: "io",
}
"""What an `OSError`'s errno says about a runtime operation."""


def _leaf(error: BaseException) -> BaseException:
    """The first leaf of an exception group (`error` itself when it is no group)."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return error


def _chain(error: BaseException) -> Iterator[BaseException]:
    """`error`, then what it was raised from or during, each group taken at its leaf."""
    seen: set[int] = set()
    current: BaseException | None = _leaf(error)
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        cause = current.__cause__ or current.__context__
        current = _leaf(cause) if cause is not None else None


def http_status(error: BaseException) -> int | None:
    """The HTTP status an error carries: its own `status_code`, its httpx response's,
    or the SDK's `HTTP <status>: ...` message prefix."""
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    match = _HTTP_STATUS.search(str(error))
    return int(match.group(1)) if match else None


def missing_file_kind(detail: str, default: SandboxErrorKind) -> SandboxErrorKind:
    """`default`, unless `detail` (a failed read's stderr) says the file is absent
    (`missing`) or the disk is full (`disk_full`)."""
    text = detail.lower()
    if any(m in text for m in _MISSING_FILE_MARKERS):
        return "missing"
    if any(m in text for m in _DISK_FULL_MARKERS):
        return "disk_full"
    return default


def _status_kind(status: int) -> SandboxErrorKind | None:
    """The kind an HTTP status from a sandbox provider names."""
    if status in _ACCOUNT_STATUSES:
        return "account"
    if status == 429:
        return "quota"
    if status == 410:
        return "box_lost"
    if status == 404:
        return "missing"
    if status == 507:
        return "disk_full"  # the box's disk, not the platform: a hold waits for nothing
    if status in (413, 422):
        return "unknown"  # the request itself is refused; the same one is again
    if status == 408 or 500 <= status < 600:
        return "io"
    return None


def _is_rpc(error: BaseException) -> bool:
    """Whether `error` is a Connect RPC failure (connectrpc ships with prime-sandboxes;
    without it nothing is one)."""
    try:
        from connectrpc.errors import ConnectError
    except ImportError:
        return False
    return isinstance(error, ConnectError)


def _rpc_kind(error: BaseException) -> SandboxErrorKind | None:
    """The kind a Connect RPC failure's code names."""
    if not _is_rpc(error):
        return None
    from connectrpc.code import Code

    return {
        Code.NOT_FOUND: "missing",
        Code.DEADLINE_EXCEEDED: "io",
        Code.UNAVAILABLE: "io",
        Code.ABORTED: "io",
        Code.RESOURCE_EXHAUSTED: "quota",
        Code.PERMISSION_DENIED: "account",
        Code.UNAUTHENTICATED: "account",
    }.get(error.code)


def _typed_kind(error: BaseException) -> SandboxErrorKind | None:
    """The kind one failure's own typed evidence names: a typed `SandboxError`, the
    SDK's or Python's exception type, an errno, an HTTP status, a Connect code."""
    if isinstance(error, SandboxError):
        return error.kind if error.kind != "unknown" else None
    name = type(error).__name__
    if name in _ACCOUNT_TYPES:
        return "account"
    if name in _BOX_LOST_TYPES:
        # A box that stopped is gone (an OOM kill or the box's own timeout is a stop).
        # One coming up, running but not yet reachable through the gateway, or whose
        # boot wait ran out with no status (the SDK's creation timeouts) is not.
        status = getattr(error, "status", None)
        if name == "SandboxNotRunningError" and status not in _STOPPED:
            return "io"
        return "box_lost"
    if name in _MISSING_FILE_TYPES:
        return "missing"
    if isinstance(error, OSError) and error.errno in _ERRNO_KINDS:
        return _ERRNO_KINDS[error.errno]
    status = http_status(error)
    if status is not None and (kind := _status_kind(status)) is not None:
        return kind
    if _is_rpc(error):
        # Its code is the evidence (httpx has a `ConnectError` of the same name).
        return _rpc_kind(error)
    if name in _IO_TYPES or isinstance(error, TimeoutError):
        return "io"
    return None


def _marked_kind(error: BaseException) -> SandboxErrorKind | None:
    """The kind one failure's message names, when nothing typed did."""
    text = str(error).lower()
    if any(m in text for m in _ACCOUNT_MARKERS):
        return "account"
    if any(m in text for m in _QUOTA_MARKERS):
        return "quota"
    if any(m in text for m in _BOX_LOST_MARKERS):
        return "box_lost"
    if any(m in text for m in _MISSING_FILE_MARKERS):
        return "missing"
    if any(m in text for m in _DISK_FULL_MARKERS):
        return "disk_full"
    if any(m in text for m in _GATEWAY_MARKERS):
        return "io"
    return None


def sandbox_error_kind(
    error: BaseException, default: SandboxErrorKind = "unknown"
) -> SandboxErrorKind:
    """Classify a runtime operation's failure (`SandboxErrorKind`). An exception group
    is classified by its leaf, and the failures it was raised from or during are read
    too. Typed evidence decides first, anywhere in that chain: a typed `SandboxError`,
    the exception's type, an errno, an HTTP status, a Connect RPC code. Only when none
    names a kind does the message text (a marker such as "payment required" or
    "sandbox_not_found") decide. With neither, the operation's `default` stands."""
    chain = list(_chain(error))
    for current in chain:
        if (kind := _typed_kind(current)) is not None:
            return kind
    for current in chain:
        if (kind := _marked_kind(current)) is not None:
            return kind
    return default


class TaskError(RolloutError):
    """Task-authored code raised — `setup`, `finalize`, or a `@reward`/`@metric`."""


class InterceptionError(RolloutError):
    """Communication with the host interception server failed."""


class TunnelError(InterceptionError):
    """The `prime_tunnel` tunnel to the host interception server couldn't be established."""


@contextlib.asynccontextmanager
async def boundary(error_cls: type[RolloutError], what: str) -> AsyncIterator[None]:
    """Run a framework→code boundary, attributing any error escaping it to `error_cls`. An
    already-typed `RolloutError` passes through unchanged — it crossed a more specific boundary
    first (e.g. a `SandboxError` from `runtime.run` inside a reward stays a `SandboxError`). A
    `TimeoutError` the code raises itself (its own I/O) becomes `error_cls` too; a stage
    deadline wraps the boundary from outside, so its expiry is a timeout, not an error.
    `what` names the boundary in the error message."""
    try:
        yield
    except RolloutError:
        raise
    except TimeoutError as e:
        raise error_cls(f"{what} timed out") from e
    except Exception as e:
        wrapped = error_cls(f"{what}: {type(e).__name__}: {e}")
        wrapped.wrapped = type(e).__name__
        raise wrapped from e


def error_cause(error: BaseException) -> str | None:
    """The type name of the exception `boundary` wrapped into `error`, so a reader of
    the record can tell a `ValueError` from a `KeyError` inside one `EnvError`. None for
    an error the boundary did not wrap (a typed error raised `from` another keeps its
    own type)."""
    return getattr(error, "wrapped", None)


def stop_condition(error: BaseException) -> str:
    """The stop condition a recorded error leaves on the trace: `<boundary>_error` for
    a typed rollout error (`SandboxError` -> `sandbox_error`), `error` for any other."""
    if isinstance(error, RolloutError):
        return f"{type(error).__name__.removesuffix('Error').lower()}_error"
    return "error"


def _provider_status(e: OpenAIError | str) -> int:
    """The HTTP status to surface for an SDK error: the provider's own for an HTTP status error, a
    retryable 5xx for a transport/timeout fault, else 502."""
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    if isinstance(e, APIStatusError):
        return e.status_code
    if isinstance(e, APITimeoutError):  # subclass of APIConnectionError — check first
        return 504
    if isinstance(e, APIConnectionError):
        return 503
    return 502


def model_error(
    e: OpenAIError | str, *, status_code: int | None = None
) -> ProviderError:
    """Map a provider failure to a `ProviderError`. `status_code` is the HTTP status surfaced to
    the harness (whose SDK then retries 5xx/429/timeout and not 4xx); derived from an SDK error
    when not given. Accepts an SDK error (the renderer) or the provider's raw error body (the
    httpx proxy)."""
    # Some SDK errors stringify empty; fall back to the type so the message is never blank.
    text = str(e) or (type(e).__name__ if isinstance(e, BaseException) else "")
    return ProviderError(
        text,
        status_code=status_code if status_code is not None else _provider_status(e),
    )
