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
from collections.abc import AsyncIterator

import httpx
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


class HarnessError(RolloutError):
    """The harness failed to install or launch, or its agent process exited unsuccessfully."""


class ToolsetError(RolloutError):
    """A task's `Toolset` could not be built or served."""


class EnvError(RolloutError):
    """The environment's own hooks failed — `run()` or `finalize()` raised (or
    ran no agent at all). Episode-level: per-agent failures stay typed on their
    traces. (Not `EnvironmentError` — that's a builtin alias of OSError.)"""


class SandboxError(RolloutError):
    """A runtime/sandbox operation failed (provisioning, exec, or file I/O). `code` names the
    fault when typed evidence (an exception type, an errno, an HTTP/RPC status — never the
    message text) identifies it, so a consumer branches on it instead of parsing the message:
    `not_found` (the path, or the box itself, is gone), `timeout` (the operation or the box's
    lifetime ran out), `disk_full` (ENOSPC), `unavailable` (the provider was unreachable or
    answered 5xx/429), `provisioning` (the box never came up), `denied` (the provider refused:
    401/402/403). None when nothing typed says. A cancelled task's own cancellation is never a
    `SandboxError`: a runtime whose SDK reports it as an ordinary failure re-raises
    `asyncio.CancelledError`, so a caller that retries on sandbox faults cannot swallow it."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


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


def sandbox_fault_code(e: BaseException) -> str | None:
    """The `SandboxError.code` a Python-level fault identifies — a missing path, a timeout,
    ENOSPC, or an `httpx` status / transport error — or None when nothing typed does."""
    if isinstance(e, FileNotFoundError):
        return "not_found"
    if isinstance(e, (TimeoutError, httpx.TimeoutException)):
        return "timeout"
    if isinstance(e, OSError) and e.errno == errno.ENOSPC:
        return "disk_full"
    if isinstance(e, httpx.HTTPStatusError):
        return _http_fault_code(e.response.status_code)
    if isinstance(e, httpx.TransportError):
        return "unavailable"
    return None


def _http_fault_code(status: int) -> str | None:
    """The `SandboxError.code` an HTTP status from a sandbox provider identifies."""
    if status == 404:
        return "not_found"
    if status in (401, 402, 403):
        return "denied"
    if status in (408, 504):
        return "timeout"
    if status == 429 or status >= 500:
        return "unavailable"
    return None


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
