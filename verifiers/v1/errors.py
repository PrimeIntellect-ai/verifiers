"""The error model — every rollout failure is attributed to one boundary, then recorded once.

Four mechanisms, each in one place:

1. Vocabulary (this module): `RolloutError` and the flat boundary types below. Each names the
   boundary a failure crossed — provider, harness, toolset, user, sandbox, taskset, or
   interception — so a recorded `trace.error.type` says where the rollout broke.
2. Classification (`boundary`): the one helper that runs a framework→code boundary and attributes
   any escaping error to that boundary's type. Infra that fails raises its type at the source
   (`runtimes` → `SandboxError`, `clients` → `ProviderError`, tunnels → `TunnelError`); an
   already-typed `RolloutError` passes through unchanged.
3. Surfacing (`interception.RolloutSession.error`): a model/tool/user call fails behind the harness
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
    """Base for errors recorded into the trace rather than crashing the rollout."""


class ProviderError(RolloutError):
    """A call to the model provider failed: transport, an HTTP error status, a timeout, a
    malformed response, or a prompt the model can't accept. The wrapped SDK/httpx error
    carries the status code and detail."""


class OverlongPromptError(ProviderError):
    """The prompt (plus the requested completion) exceeded the model's context window.
    A budget limit, not a crash: the interception server ends the rollout as a clean,
    truncated trajectory rather than recording it as an error."""


class HarnessError(RolloutError):
    """The harness failed to install or launch, or its agent process exited unsuccessfully."""


class ToolsetError(RolloutError):
    """A task's `Toolset` (tool MCP server) could not be built or served for the rollout."""


class UserError(RolloutError):
    """A task's `User` simulator could not be served, or its `respond` raised while driving the
    conversation."""


class SandboxError(RolloutError):
    """A runtime/sandbox operation failed — provisioning, a non-zero program exit, exec or file
    I/O across the sandbox boundary, or a rollout stage exceeding its timeout. The framework's
    execution boundary, distinct from the model/harness/tool/interception ones."""


class TasksetError(RolloutError):
    """Taskset-authored code raised — `setup`, `finalize`, or a `@reward`/`@metric`/`@group_reward`
    (e.g. a verifier script the reward runs exiting non-zero). The framework wraps any non-
    `RolloutError` escaping a taskset hook as this, so taskset code raises plain Python errors and
    needn't import an error type — the type names the taskset boundary, the wrapped error the detail."""


class InterceptionError(RolloutError):
    """The interception layer — the host server a sandboxed harness/tool reaches for its model
    calls and `/state` + `/task` channels — could not be reached."""


class TunnelError(InterceptionError):
    """The `prime_tunnel` reverse tunnel that makes the host interception server reachable from
    inside a remote runtime could not be established. Retried with backoff before it surfaces;
    tunnel creation is network-bound and globally rate-capped, so transient failures are expected.
    (A runtime publishing its *own* ports — `Runtime.expose` — is a `SandboxError`, not this.)"""


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


def model_error(e: OpenAIError | str) -> ProviderError:
    """Map a provider failure to our error type: an overlong prompt (a budget limit the
    interception server turns into a clean truncation) is told apart from any other provider
    call failure (auth, rate limit, transport, a genuine bad request, ...), which becomes a
    plain `ProviderError`. Accepts either an SDK error (the renderer) or the provider's raw
    error body (the httpx proxy)."""
    # Some SDK errors stringify empty; fall back to the type so the message is never blank.
    text = str(e) or (type(e).__name__ if isinstance(e, BaseException) else "")
    if any(phrase in text.casefold() for phrase in _CONTEXT_LENGTH_PHRASES):
        return OverlongPromptError(text)
    return ProviderError(text)
