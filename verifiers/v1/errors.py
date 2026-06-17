"""The exception tree the rollout catches and records.

Only errors the rollout deliberately catches (and records into `trace.error` as a
`verifiers.v1.trace.Error`) live here. Everything else propagates with its
built-in traceback — we own the code, so we don't wrap internal invariants in
custom messages.
"""

from openai import OpenAIError


class RolloutError(Exception):
    """Base for errors recorded into the trace rather than crashing the rollout."""


class ModelError(RolloutError):
    """A model/provider call failed (bad request, auth, ...)."""


class OverlongPromptError(ModelError):
    """The prompt (plus the requested completion) exceeded the model's context window.
    A budget limit, not a crash: the interception server ends the rollout as a clean,
    truncated trajectory rather than recording it as an error."""


class ToolError(RolloutError):
    """A tool invocation failed."""


class ProgramError(RolloutError):
    """A program failed (non-zero exit or timeout)."""


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
    """Map a provider failure to our error type, distinguishing an overlong prompt from any
    other model-call failure (auth, rate limit, a genuine bad request, ...). Accepts either an
    SDK error (the renderer) or the provider's raw error body (the httpx proxy)."""
    # Some SDK errors stringify empty; fall back to the type so the message is never blank.
    text = str(e) or (type(e).__name__ if isinstance(e, BaseException) else "")
    if any(phrase in text.casefold() for phrase in _CONTEXT_LENGTH_PHRASES):
        return OverlongPromptError(text)
    return ModelError(text)
