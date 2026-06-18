"""The exception tree the rollout catches and records.

Only errors the rollout deliberately catches (and records into `trace.error` as a
`verifiers.v1.trace.Error`) live here. Everything else propagates with its
built-in traceback — we own the code, so we don't wrap internal invariants in
custom messages.
"""

from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError as OpenAIAuthenticationError,
    OpenAIError,
)

from verifiers.v1.types import Response


class RolloutError(Exception):
    """Base for errors recorded into the trace rather than crashing the rollout."""


class ModelError(RolloutError):
    """A model/provider call failed (bad request, auth, ...)."""


class ProviderError(ModelError):
    """Base for failures at the model-provider boundary."""


class ProviderHTTPError(ProviderError):
    """The provider returned an unsuccessful HTTP response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        detail = message.strip() or "<empty response body>"
        super().__init__(f"provider returned HTTP {status_code}: {detail}")


class ProviderAuthenticationError(ProviderHTTPError):
    """The provider rejected its API key or credentials."""


class ProviderRateLimitError(ProviderHTTPError):
    """The provider rejected the request because of a rate limit."""


class ProviderTransportError(ProviderError):
    """The provider could not be reached or its connection failed."""


class ProviderTimeoutError(ProviderTransportError):
    """The provider request timed out."""


class InvalidModelResponseError(ProviderError):
    """The provider returned a response Verifiers could not consume."""


class ProviderResponseError(InvalidModelResponseError):
    """The provider response was malformed or failed schema validation."""


class EmptyModelResponseError(InvalidModelResponseError):
    """The provider returned no assistant content or tool calls."""


class HarnessError(RolloutError):
    """The harness implementation failed or its agent process exited unsuccessfully."""


class OverlongPromptError(ModelError):
    """The prompt (plus the requested completion) exceeded the model's context window.
    A budget limit, not a crash: the interception server ends the rollout as a clean,
    truncated trajectory rather than recording it as an error."""


class ToolError(RolloutError):
    """Task-owned tools could not be constructed or served for the rollout."""


class ProgramError(RolloutError):
    """A runtime process, sandbox, tunnel, or command boundary failed."""


class SandboxError(ProgramError):
    """A sandbox reached a terminal lifetime or resource failure state."""


class SandboxTimeoutError(SandboxError):
    """The sandbox lifetime expired while the rollout was still running."""


class SandboxOutOfMemoryError(SandboxError):
    """The sandbox was terminated because it exhausted its memory limit."""


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


def ensure_model_output(response: Response) -> None:
    """Reject a completion that cannot advance a rollout or safely reach scoring."""
    if response.message.tool_calls or (response.message.content or "").strip():
        return
    detail = (
        "reasoning but no content or tool calls"
        if response.message.reasoning_content
        else "no content or tool calls"
    )
    raise EmptyModelResponseError(f"model response contained {detail}")


def provider_http_error(status_code: int, message: str) -> ModelError:
    """Map an upstream HTTP failure, preserving its actionable category and status."""
    overlong = model_error(message)
    if isinstance(overlong, OverlongPromptError):
        return overlong
    if status_code in (401, 403):
        return ProviderAuthenticationError(status_code, message)
    if status_code == 429:
        return ProviderRateLimitError(status_code, message)
    return ProviderHTTPError(status_code, message)


def model_error(e: OpenAIError | str) -> ModelError:
    """Map an SDK/provider failure to the specific rollout error it represents."""
    # Some SDK errors stringify empty; fall back to the type so the message is never blank.
    text = str(e) or (type(e).__name__ if isinstance(e, BaseException) else "")
    if any(phrase in text.casefold() for phrase in _CONTEXT_LENGTH_PHRASES):
        return OverlongPromptError(text)
    if isinstance(e, OpenAIAuthenticationError):
        return ProviderAuthenticationError(e.status_code, text)
    if isinstance(e, APITimeoutError):
        return ProviderTimeoutError(f"provider request timed out: {text}")
    if isinstance(e, APIConnectionError):
        return ProviderTransportError(f"provider transport failed: {text}")
    if isinstance(e, APIResponseValidationError):
        return ProviderResponseError(f"provider response validation failed: {text}")
    if isinstance(e, APIStatusError):
        return provider_http_error(e.status_code, text)
    return ModelError(text)
