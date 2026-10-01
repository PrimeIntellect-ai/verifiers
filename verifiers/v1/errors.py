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
from collections.abc import AsyncIterator

from openai import OpenAIError


class RolloutError(Exception):
    """Base for a failure recorded onto the trace rather than crashing the rollout.

    `retryable` says whether rerunning the whole rollout could plausibly succeed —
    a property of the fault, not the policy: `True` for a transient infra fault,
    `False` for a deterministic one a rerun only reproduces, `None` when the type
    alone can't tell (the `--env.retries` include/exclude lists decide). It rides
    onto the recorded `Error` so `trace_should_retry` can honour it without the
    live exception."""

    retryable: bool | None = None


def _transient_status(status_code: int) -> bool:
    """Whether an HTTP status a provider fault carries is worth a rerun: any 5xx
    (incl. Cloudflare 52x) plus 408/409/425/429. A deterministic 4xx (400 bad
    request, 413 context overflow, 422) just re-fails, so a rerun only burns the
    trajectory again."""
    return status_code >= 500 or status_code in {408, 409, 425, 429}


class ProviderError(RolloutError):
    """A model-provider call failed (transport, HTTP status, timeout, or malformed response).
    `status_code` is the HTTP status surfaced to the harness so its SDK retries transient faults
    (5xx/429/timeout) and not deterministic ones (4xx) — relayed from the provider, or chosen for a
    transport fault."""

    def __init__(self, message: str = "", *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        """Transient statuses (5xx/429/timeout) can succeed on a rerun; 4xx can't."""
        return _transient_status(self.status_code)


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


class SandboxError(RolloutError):
    """A runtime/sandbox operation failed (provisioning, exec, or file I/O)."""

    retryable = True  # provisioning/exec hiccups usually clear on a fresh box


class TaskError(RolloutError):
    """Task-authored code raised — `setup`, `finalize`, or a `@reward`/`@metric`."""

    retryable = False  # authored code is deterministic — a rerun reproduces it


class InterceptionError(RolloutError):
    """Communication with the host interception server failed."""

    retryable = True  # a tunnel/interception drop is transient


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
        raise error_cls(f"{what}: {type(e).__name__}: {e}") from e


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
