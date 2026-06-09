"""The exception tree the rollout catches and records.

Only errors the rollout deliberately catches (and records into `trace.error` as a
`verifiers.v1.trace.Error`) live here. Everything else propagates with its
built-in traceback — we own the code, so we don't wrap internal invariants in
custom messages.
"""


class RolloutError(Exception):
    """Base for errors recorded into the trace rather than crashing the rollout."""


class ModelError(RolloutError):
    """A model/provider call failed (bad request, auth, overlong prompt, ...)."""


class ToolError(RolloutError):
    """A tool invocation failed."""


class ProgramError(RolloutError):
    """A program failed (non-zero exit or timeout)."""
