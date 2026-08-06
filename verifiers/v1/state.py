"""Mutable rollout-level state shared across tool server + host."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypeVar

from verifiers.v1.utils.generic import concrete_type


class State(BaseModel):
    model_config = ConfigDict(ser_json_inf_nan="constants")

    artifacts: dict[str, Path | None] = Field(default_factory=dict)
    """Collected artifact archives, spooled on host disk (see `utils.artifacts`),
    keyed by their source path in the box they came from."""


StateT = TypeVar("StateT", bound=State, default=State)


def state_cls(cls: type) -> type[State]:
    """Resolve a class's `State` specialization through its MRO, else `State`."""
    return concrete_type(cls, State) or State
