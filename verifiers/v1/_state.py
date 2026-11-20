"""Mutable rollout-level state shared across tool server + host."""

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypeVar

from verifiers.v1._utils.generic import concrete_type


class State(BaseModel):
    """Mutable rollout state shared between the program and the host."""

    model_config = ConfigDict(ser_json_inf_nan="constants")

    artifacts: dict[str, bytes | None] = Field(default_factory=dict)
    """Files collected from the runtime for grading, by source path."""


StateT = TypeVar("StateT", bound=State, default=State)


def state_cls(cls: type) -> type[State]:
    """Resolve a class's `State` specialization through its MRO, else `State`."""
    return concrete_type(cls, State) or State
