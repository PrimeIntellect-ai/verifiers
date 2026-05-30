"""V1 state contract exports.

V1 uses the shared top-level ``verifiers.State`` type. V1 tasks opt state
instances into strict runtime/lifecycle-field handling when they are passed to
``State.for_task(...)``.
"""

from verifiers.types import State

__all__ = ["State"]
