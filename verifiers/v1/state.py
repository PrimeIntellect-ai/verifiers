"""Per-rollout shared runtime state.

`Trace.state` is a typed, mutable `State` that a rollout's tool servers (`@vf.tool`) and user
simulator (`respond`) read+write as `self.state` (synced over the interception server per call), and
that `@reward`/`@metric`/`finalize` read+write directly off the trace. Unlike `Trace.info` — the
free-form artifact bag persisted to `results.jsonl` — `state` is transient runtime scratch (counters,
game progress, an end-of-trajectory flag): never written to disk or sent over the wire.

The base `State` is empty — the framework holds no opinion about its contents. Subclass it to declare
typed fields, then parameterize the task (`Task[MyState]`) and any stateful server
(`Toolset[Config, MyState]` / `User[Config, MyState]`) to type it. To end a trajectory from state,
add your own flag and a `@vf.stop` that checks it (e.g. `user_finished`) — see the user-sim examples.
"""

from typing import get_args

from pydantic import ConfigDict
from typing_extensions import TypeVar

from verifiers.v1.types import StrictBaseModel


class State(StrictBaseModel):
    """Per-rollout mutable runtime state shared across a rollout's tool/user servers and its scoring.
    Empty by default — subclass to declare typed fields, e.g. `class MyState(State): count: int = 0`
    (fields need defaults so the framework can build the initial state). Strict (unknown fields are
    rejected) and transient: never persisted to disk or sent over the wire (use the free-form
    `Trace.info` for artifacts that must persist)."""

    model_config = ConfigDict(ser_json_inf_nan="constants")


StateT = TypeVar("StateT", bound=State, default=State)


def state_cls(cls: type) -> type[State]:
    """The `State` subclass a class parameterizes — `Task[MyState]`,
    `Toolset[Config, MyState]`, `User[Config, MyState]` — read off its generic bases, walking the MRO
    so a further subclass inherits it. Falls back to the base `State` when none is given (the common
    case: an env that doesn't customize state, written without the generic param). A pydantic generic
    (`Task[MyState]`) parametrizes into a real class, not a typing alias, so its args live in
    `__pydantic_generic_metadata__` rather than `__orig_bases__`; check both."""
    for klass in getattr(cls, "__mro__", [cls]):
        meta = getattr(klass, "__pydantic_generic_metadata__", None) or {}
        for base in (*meta.get("args", ()), *getattr(klass, "__orig_bases__", ())):
            for arg in (base, *get_args(base)):
                if isinstance(arg, type) and issubclass(arg, State):
                    return arg
    return State
