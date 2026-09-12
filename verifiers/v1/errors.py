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
   A boundary type may be narrowed by a subclass naming the fault (`SandboxUnavailableError`) when
   typed evidence — an exception type, an errno, an HTTP status, an RPC code — says so (`sandbox_error`);
   the subclass's `code` rides onto the trace so a consumer branches on it instead of the message text.
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
from collections.abc import AsyncIterator, Iterator
from typing import ClassVar

import httpx
from openai import OpenAIError


class RolloutError(Exception):
    """Base for a failure recorded onto the trace rather than crashing the rollout."""

    code: ClassVar[str | None] = None
    """Stable machine-readable name of the failure class, recorded as `trace.Error.code`; None on
    the bare boundary types. Set by subclasses that name a fault, never per instance."""


class ProviderError(RolloutError):
    """A model-provider call failed (transport, HTTP status, timeout, or malformed response).
    `status_code` is the HTTP status surfaced to the harness so its SDK retries transient faults
    (5xx/429/timeout) and not deterministic ones (4xx) — relayed from the provider, or chosen for a
    transport fault."""

    def __init__(self, message: str = "", *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class HarnessError(RolloutError):
    """The harness failed to install or launch, or its agent process exited unsuccessfully."""


class HarnessTimeoutError(HarnessError):
    """The rollout ran past its agent time budget (`TimeoutConfig.rollout`)."""

    code = "agent_timeout"


class ToolsetError(RolloutError):
    """A task's `Toolset` could not be built or served."""


class EnvError(RolloutError):
    """The environment's own hooks failed — `run()` or `finalize()` raised (or
    ran no agent at all). Episode-level: per-agent failures stay typed on their
    traces. (Not `EnvironmentError` — that's a builtin alias of OSError.)"""


class SandboxError(RolloutError):
    """A runtime/sandbox operation failed (provisioning, exec, or file I/O) — the bare type when
    nothing typed says more; the subclasses below name the fault (see `sandbox_error`)."""


class SandboxNotFoundError(SandboxError):
    """The path, or the box itself, is gone."""

    code = "not_found"


class SandboxTimeoutError(SandboxError):
    """The operation, or the box's lifetime, ran out."""

    code = "timeout"


class SandboxUnavailableError(SandboxError):
    """The provider or box is transiently unreachable (retry later)."""

    code = "unavailable"


class SandboxDeniedError(SandboxError):
    """The provider refused: auth, billing, permission."""

    code = "denied"


class SandboxDiskFullError(SandboxError):
    """The box's disk is full (ENOSPC)."""

    code = "disk_full"


class SandboxProvisioningError(SandboxError):
    """The box never came up."""

    code = "provisioning"


class TaskError(RolloutError):
    """Task-authored code raised — `setup`, `finalize`, or a `@reward`/`@metric`."""


class InterceptionError(RolloutError):
    """The host interception server (model calls + `/state` + `/task` channels) couldn't be reached."""


class TunnelError(InterceptionError):
    """The `prime_tunnel` tunnel to the host interception server couldn't be established."""


@contextlib.asynccontextmanager
async def boundary(error_cls: type[RolloutError], what: str) -> AsyncIterator[None]:
    """Run a framework→code boundary, attributing any error escaping it to `error_cls`. An
    already-typed `RolloutError` passes through unchanged — it crossed a more specific boundary
    first (e.g. a `SandboxError` from `runtime.run` inside a reward stays a `SandboxError`). A
    `TimeoutError` (the stage exceeded its budget) becomes `error_cls` too. `what` names the
    boundary in the error message."""
    try:
        yield
    except RolloutError:
        raise
    except TimeoutError as e:
        raise error_cls(f"{what} timed out") from e
    except Exception as e:
        raise error_cls(f"{what}: {type(e).__name__}: {e}") from e


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


def failures(e: BaseException) -> Iterator[BaseException]:
    """`e` and the failures it wraps, outermost first, as a traceback prints them: `__cause__`,
    else the implicit `__context__` of an error raised inside another's handler without `from`
    (the prime SDK's `APIError("HTTP 503: ...")` keeps its typed status only there)."""
    current: BaseException | None = e
    while current is not None:
        yield current
        if current.__cause__ is None and current.__suppress_context__:
            return  # `raise ... from None`
        current = current.__cause__ or current.__context__


def _http_class(status: int) -> type[SandboxError] | None:
    """The `SandboxError` subclass an HTTP status from a sandbox provider names."""
    if status == 404:
        return SandboxNotFoundError
    if status in (401, 402, 403):
        return SandboxDeniedError
    if status in (408, 504):
        return SandboxTimeoutError
    if status == 429 or status >= 500:
        return SandboxUnavailableError
    return None


def _rpc_class(e: BaseException) -> type[SandboxError] | None:
    """The `SandboxError` subclass a Connect RPC failure's code names (connectrpc ships with
    prime-sandboxes; without it nothing is an RPC failure)."""
    try:
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError
    except ImportError:
        return None
    if not isinstance(e, ConnectError):
        return None
    if e.code is Code.NOT_FOUND:
        return SandboxNotFoundError
    if e.code is Code.DEADLINE_EXCEEDED:
        return SandboxTimeoutError
    if e.code in (Code.UNAVAILABLE, Code.RESOURCE_EXHAUSTED, Code.ABORTED):
        return SandboxUnavailableError
    if e.code in (Code.PERMISSION_DENIED, Code.UNAUTHENTICATED):
        return SandboxDeniedError
    return None


def _sandbox_class(e: BaseException) -> type[SandboxError] | None:
    """The `SandboxError` subclass one failure's own type, errno, HTTP status or RPC code names."""
    if isinstance(e, FileNotFoundError):
        return SandboxNotFoundError
    if isinstance(e, (TimeoutError, httpx.TimeoutException)):
        return SandboxTimeoutError
    if isinstance(e, OSError) and e.errno == errno.ENOSPC:
        return SandboxDiskFullError
    if isinstance(e, httpx.HTTPStatusError):
        return _http_class(e.response.status_code)
    if isinstance(e, httpx.TransportError):
        return SandboxUnavailableError
    return _rpc_class(e)


def sandbox_error(
    message: str, e: BaseException, *, default: type[SandboxError] = SandboxError
) -> SandboxError:
    """Map a runtime failure to the `SandboxError` subclass its typed evidence names — an
    exception type, an errno, an HTTP status or a Connect RPC code, read off `e` and the failures
    it chains, never its text — else `default`. The runtime counterpart of `model_error`; runtimes
    call it at their mapping seam: `raise sandbox_error("prime exec failed", e) from e`."""
    cls = next((c for c in map(_sandbox_class, failures(e)) if c is not None), default)
    return cls(f"{message}: {e}")
