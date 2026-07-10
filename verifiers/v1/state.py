"""Mutable state shared within one rollout.

Tool and user servers synchronize it through the interception state channel. It is excluded
from serialized traces; persist artifacts in `Trace.info` instead.
"""

from pydantic import ConfigDict
from typing_extensions import TypeVar

from verifiers.v1.types import StrictBaseModel
from verifiers.v1.utils.generic import generic_type


class State(StrictBaseModel):
    model_config = ConfigDict(ser_json_inf_nan="constants")


StateT = TypeVar("StateT", bound=State, default=State)


def state_cls(cls: type) -> type[State]:
    """Resolve a class's `State` specialization through its MRO, else `State`."""
    return generic_type(cls, State) or State
