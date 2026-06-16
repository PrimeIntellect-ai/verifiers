"""Per-rollout shared runtime state.

`Trace.state` is a typed, mutable `State` that a rollout's tool servers (`@vf.tool`) and user
simulator (`respond`) read+write as `self.state` (synced over the interception server per call), and
that `@reward`/`@metric`/`finalize` read+write directly off the trace. Unlike `Trace.info` — the
free-form artifact bag persisted to `results.jsonl` — `state` is transient runtime scratch (counters,
game state, the `done` end-of-trajectory flag): never written to disk or sent over the wire.

Subclass `State` to declare typed fields and parameterize the taskset (`Taskset[Task, Config,
MyState]`) and any stateful server (`Toolset[Config, MyState]` / `User[Config, MyState]`) to type it.
"""

from typing import get_args

from typing_extensions import TypeVar

from verifiers.v1.types import StrictBaseModel


class State(StrictBaseModel):
    """Per-rollout mutable runtime state shared across a rollout's tool/user servers and its scoring.
    Subclass to declare typed fields, e.g. `class MyState(State): count: int = 0` — fields need
    defaults so the framework can build the initial state. Strict (unknown fields are rejected) and
    transient: never persisted to disk or sent over the wire (use the free-form `Trace.info` for
    artifacts that must persist)."""

    done: bool = False
    """Set True (in a user sim's `respond`, or a tool) to end the trajectory."""


StateT = TypeVar("StateT", bound=State, default=State)


def state_cls(cls: type) -> type[State]:
    """The `State` subclass a class parameterizes — `Taskset[Task, Config, MyState]`,
    `Toolset[Config, MyState]`, `User[Config, MyState]` — read off its generic bases, walking the MRO
    so a further subclass inherits it. Falls back to the base `State` when none is given (the common
    case: an env that doesn't customize state, written without the extra generic param)."""
    for klass in getattr(cls, "__mro__", [cls]):
        for base in getattr(klass, "__orig_bases__", ()):
            for arg in get_args(base):
                if isinstance(arg, type) and issubclass(arg, State):
                    return arg
    return State
