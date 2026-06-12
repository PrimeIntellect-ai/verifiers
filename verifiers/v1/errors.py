"""The exception tree the rollout catches and records.

Only errors the rollout deliberately catches (and records into `trace.error` as a
`verifiers.v1.trace.Error`) live here. Everything else propagates with its
built-in traceback — we own the code, so we don't wrap internal invariants in
custom messages.
"""


class RolloutError(Exception):
    """Base for errors recorded into the trace rather than crashing the rollout."""


class ModelError(RolloutError):
    """A model/provider call failed (bad request, auth, ...)."""


class OverlongPromptError(ModelError):
    """The prompt (plus the requested completion) exceeded the model's context window.
    A budget limit, not a crash: the interception server ends the rollout as a clean,
    truncated trajectory rather than recording it as an error."""


# Phrases providers use to say "your prompt exceeded the context window" (OpenAI, vLLM,
# SGLang, DeepSeek, Anthropic, ...). Substrings of the error text, casefolded.
_CONTEXT_LENGTH_PHRASES = (
    "this model's maximum context length is",
    "is longer than the model's context length",
    "is longer than the maximum model length",
    "exceeds the model's context length",
    "exceed the configured limit",
    "exceeds the configured limit",
    "exceeded model",
    "prompt_too_long",
    "prompt is too long",
    "context length",
    "maximum model length",
)


def classify_model_error(text: str) -> ModelError:
    """Map a provider error message to our error type, distinguishing an overlong prompt
    (a budget limit the interception server turns into a clean truncation) from any other
    model-call failure (auth, rate limit, a genuine bad request, ...)."""
    lowered = text.casefold()
    if any(phrase in lowered for phrase in _CONTEXT_LENGTH_PHRASES):
        return OverlongPromptError(text)
    return ModelError(text)


class ToolError(RolloutError):
    """A tool invocation failed."""


class ProgramError(RolloutError):
    """A program failed (non-zero exit or timeout)."""
