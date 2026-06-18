"""The exception tree the rollout catches and records.

Only errors the rollout deliberately catches (and records into `trace.error` as a
`verifiers.v1.trace.Error`) live here. Everything else propagates with its
built-in traceback — we own the code, so we don't wrap internal invariants in
custom messages.

Each type names the *boundary* a failure crossed — provider, model, harness, tool, or
program/runtime — so a recorded `trace.error.type` says where the rollout broke. The
detail (status code, stderr, ...) comes from the wrapped inner error; we add a type only
when the boundary isn't already clear from it.
"""

from openai import OpenAIError


class RolloutError(Exception):
    """Base for errors recorded into the trace rather than crashing the rollout."""


class ModelError(RolloutError):
    """The model could not produce a usable completion — the umbrella for a failed
    provider call (`ProviderError`) and an overlong prompt (`OverlongPromptError`)."""


class ProviderError(ModelError):
    """A call to the model provider failed: transport, an HTTP error status, a timeout, or a
    malformed response. The wrapped SDK/httpx error carries the status code and detail."""


class OverlongPromptError(ModelError):
    """The prompt (plus the requested completion) exceeded the model's context window.
    A budget limit, not a crash: the interception server ends the rollout as a clean,
    truncated trajectory rather than recording it as an error."""


class HarnessError(RolloutError):
    """The harness failed to install or launch, or its agent process exited unsuccessfully."""


class ToolError(RolloutError):
    """A task's tool / MCP servers could not be built or served for the rollout."""


class ProgramError(RolloutError):
    """A program failed (non-zero exit or timeout) — e.g. a runtime/sandbox operation or a
    rollout stage exceeding its timeout."""


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


def model_error(e: OpenAIError | str) -> ModelError:
    """Map a provider failure to our error type: an overlong prompt (a budget limit the
    interception server turns into a clean truncation) is told apart from any other provider
    call failure (auth, rate limit, transport, a genuine bad request, ...), which becomes a
    `ProviderError`. Accepts either an SDK error (the renderer) or the provider's raw error
    body (the httpx proxy)."""
    # Some SDK errors stringify empty; fall back to the type so the message is never blank.
    text = str(e) or (type(e).__name__ if isinstance(e, BaseException) else "")
    if any(phrase in text.casefold() for phrase in _CONTEXT_LENGTH_PHRASES):
        return OverlongPromptError(text)
    return ProviderError(text)
