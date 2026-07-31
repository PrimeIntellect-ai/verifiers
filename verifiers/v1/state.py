"""Mutable state shared within one rollout.

Tool servers synchronize it through the interception state channel. It is excluded
from serialized traces.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypeVar

from verifiers.v1.utils.generic import concrete_type


class State(BaseModel):
    model_config = ConfigDict(ser_json_inf_nan="constants")
    artifacts: dict[str, bytes] = Field(default_factory=dict)


StateT = TypeVar("StateT", bound=State, default=State)


def state_cls(cls: type) -> type[State]:
    """Resolve a class's `State` specialization through its MRO, else `State`."""
    return concrete_type(cls, State) or State
