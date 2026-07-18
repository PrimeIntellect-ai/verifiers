"""The error model — every rollout failure is attributed to one boundary, then recorded once.

Four mechanisms, each in one place:

1. Vocabulary (this module): `RolloutError` and the flat boundary types below. Each names the
   boundary a failure crossed — provider, harness, toolset, user, sandbox, task, or
   interception — so a recorded `trace.error.type` says where the rollout broke.
2. Classification (`boundary`): the one helper that runs a framework→code boundary and attributes
   any escaping error to that boundary's type. Extension code (task hooks, harness subclasses)
   raises plain Python errors — it never constructs a `vf` error type; `boundary` classifies them.
   Infra that fails raises its type at the source (`runtimes` → `SandboxError`, `clients` →
   `ProviderError`, tunnels → `TunnelError`); an already-typed `RolloutError` passes through unchanged.
3. Surfacing (`session.RolloutSession.error`): a model/tool/user call fails behind the harness
   subprocess and comes back as HTTP, so the interception server stashes the real error there and
   the rollout re-raises it once the harness returns — not a secondary `HarnessError`.
4. Capture (`Rollout.run`, mirrored by the env-server): the one place that records a failure (typed
   or not) onto the trace and never lets it cancel sibling rollouts. A bad rollout is data, not a
   crash.

The detail (status code, stderr, ...) comes from the wrapped inner error; we add a type only when
the boundary isn't already clear from it.
"""

import contextlib
from collections.abc import AsyncIterator

from openai import OpenAIError


class RolloutError(Exception):
    """Base for a failure recorded onto the trace rather than crashing the rollout."""


class ProviderError(RolloutError):
    """A model-provider call failed (transport, HTTP status, timeout, or malformed response).
    `status_code` is the HTTP status surfaced to the harness so its SDK retries transient faults
    (5xx/429/timeout) and not deterministic ones (4xx) — relayed from the provider, or chosen for a
    transport fault."""

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int = 502,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers
        """Provider response headers when the failure carried an HTTP response (request
        ids, rate-limit diagnostics) — surfaced on the trace's per-call records."""


class OverlongPromptError(ProviderError):
    """The prompt exceeded the model's context window — a budget limit, ended as a clean
    truncation rather than recorded as an error. Defaults to a 400 (what the interception
    server surfaces for it — deterministic, so an SDK never retries it); `model_error`
    keeps the provider's real status when the failure carried one."""

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int = 400,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, headers=headers)


class HarnessError(RolloutError):
    """The harness failed to install or launch, or its agent process exited unsuccessfully."""


class ToolsetError(RolloutError):
    """A task's `Toolset` could not be built or served."""


class UserError(RolloutError):
    """A task's `User` simulator could not be served, or its `respond` raised."""


class SandboxError(RolloutError):
    """A runtime/sandbox operation failed (provisioning, exec, or file I/O)."""


class TaskError(RolloutError):
    """Task-authored code raised — `setup`, `finalize`, or a `@reward`/`@metric`/`@group_reward`."""


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


_CONTEXT_LENGTH_PHRASES = (
    "this model's maximum context length is",
    "is longer than the model's context length",
    "is longer than the maximum model length",
    "exceeds the model's context length",
    "exceed the configured limit",
    "exceeds the configured limit",
    "exceeded model",
    "prompt_too_long",
    "context length",
    "maximum model length",
)


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
    e: OpenAIError | str,
    *,
    status_code: int | None = None,
    headers: dict[str, str] | None = None,
) -> ProviderError:
    """Map a provider failure to our error type: an overlong prompt (a budget limit the interception
    server turns into a clean truncation) is told apart from any other provider call failure, which
    becomes a plain `ProviderError`. `status_code` is the HTTP status surfaced to the harness (whose
    SDK then retries 5xx/429/timeout and not 4xx); derived from an SDK error when not given. Accepts
    an SDK error (the renderer) or the provider's raw error body (the httpx proxy)."""
    from openai import APIStatusError

    # Some SDK errors stringify empty; fall back to the type so the message is never blank.
    text = str(e) or (type(e).__name__ if isinstance(e, BaseException) else "")
    # An SDK status error carries the provider's HTTP response; keep its diagnostics
    # (request ids, rate limits) when the caller didn't pass them explicitly.
    if headers is None and isinstance(e, APIStatusError):
        headers = dict(e.response.headers)
    if any(phrase in text.casefold() for phrase in _CONTEXT_LENGTH_PHRASES):
        # Keep the provider's real status when the failure carried one; else the class
        # default (the 400 the interception server surfaces for overlong prompts).
        if status_code is None and isinstance(e, APIStatusError):
            status_code = e.status_code
        return OverlongPromptError(
            text,
            **({} if status_code is None else {"status_code": status_code}),
            headers=headers,
        )
    return ProviderError(
        text,
        status_code=status_code if status_code is not None else _provider_status(e),
        headers=headers,
    )
