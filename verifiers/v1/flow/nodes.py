"""Node declarations — the vocabulary a `Flow` is written in.

A node is work in a runtime that produces a record. `then=` continues
unconditionally (a tuple starts every named node in parallel); `outcomes=` routes
on a name the node emits (`"*"` is the default); `on_error` and `on_exhausted` are
the reserved failure edges. Cycles are legal only through outcome edges and must
pass through a node with `max_visits`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from verifiers.v1.runtimes import RuntimeConfig


class _End:
    def __repr__(self) -> str:
        return "END"


END = _End()
"""Terminal target: the branch stops here."""

Target = str | tuple[str, ...] | _End
RuntimeSpec = str | RuntimeConfig
"""`"fresh"` (provision from the seat's config), `"inherit:<node>"` (the same live
runtime that node ran in), or an explicit runtime config."""


@dataclass(frozen=True)
class Join:
    kind: Literal["all", "any", "at_least"] = "all"
    k: int | Callable[..., int] = 0
    """For `at_least`: the count, or a callable of `Upstream` (a config-dependent K)."""


ALL = Join("all")
ANY = Join("any")


def at_least(k: int | Callable[..., int]) -> Join:
    return Join("at_least", k)


@dataclass(frozen=True, kw_only=True)
class Node:
    name: str = ""  # set by compile from the class attribute name
    then: Target | None = None
    outcomes: dict[str, Target] | None = None
    join: Join = ALL
    max_visits: int | None = None
    """Visits before `on_exhausted` fires. None = unbounded; every cycle must pass
    through a node that sets it."""
    on_error: Target | None = None
    on_exhausted: Target | None = None
    runtime: RuntimeSpec = "fresh"
    retries: int = 0
    pools: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return type(self).__name__.removesuffix("Node").lower()

    @property
    def inherits(self) -> str | None:
        """The node whose live runtime this one runs in, for `"inherit:<node>"`."""
        if isinstance(self.runtime, str) and self.runtime.startswith("inherit:"):
            return self.runtime.removeprefix("inherit:")
        return None

    def edges(self) -> dict[str, list[str]]:
        """Named outgoing edges → target node names (END dropped)."""
        out: dict[str, list[str]] = {}
        if self.then is not None:
            out["then"] = _names(self.then)
        for outcome, target in (self.outcomes or {}).items():
            out[f"outcome:{outcome}"] = _names(target)
        for label in ("on_error", "on_exhausted"):
            target = getattr(self, label)
            if target is not None:
                out[label] = _names(target)
        return out

    def successors(self) -> set[str]:
        return {name for names in self.edges().values() for name in names}


def _names(target: Target) -> list[str]:
    if isinstance(target, _End):
        return []
    if isinstance(target, str):
        return [target]
    return list(target)


@dataclass(frozen=True, kw_only=True)
class AgentNode(Node):
    seat: str
    make_task: Callable[..., Any]
    pools: tuple[str, ...] = ("runtimes",)


@dataclass(frozen=True, kw_only=True)
class RunNode(Node):
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    pools: tuple[str, ...] = ("runtimes",)


@dataclass(frozen=True, kw_only=True)
class FnNode(Node):
    func: Callable[..., Any | Awaitable[Any]]


@dataclass(frozen=True, kw_only=True)
class ExpandNode(Node):
    seat: str
    make_task: Callable[..., Any]
    over: Callable[..., Iterable[Any]]
    max_active: int | None = None
    pools: tuple[str, ...] = ("runtimes",)


def agent(
    seat: str,
    make_task: Callable[..., Any],
    *,
    then: Target | None = None,
    outcomes: dict[str, Target] | None = None,
    **kw: Any,
) -> AgentNode:
    """One verifiers Agent (the config field `seat`) on the Task `make_task(upstream)`
    returns. With `outcomes`, the agent gets a `submit_outcome` tool."""
    return AgentNode(seat=seat, make_task=make_task, then=then, outcomes=outcomes, **kw)


def run(
    argv: list[str],
    *,
    runtime: RuntimeSpec,
    then: Target | None = None,
    outcomes: dict[str, Target] | None = None,
    env: dict[str, str] | None = None,
    **kw: Any,
) -> RunNode:
    """A command in a runtime, no model. Its outcome is the last `Outcome: <name>`
    line on stdout; without one, exit 0 is `completed` and anything else `failed`."""
    return RunNode(
        argv=argv, runtime=runtime, then=then, outcomes=outcomes, env=env or {}, **kw
    )


def fn(
    func: Callable[..., Any],
    *,
    then: Target | None = None,
    outcomes: dict[str, Target] | None = None,
    **kw: Any,
) -> FnNode:
    """Host-side Python (sync or async) over `Upstream`; with `outcomes`, its return
    value routes."""
    return FnNode(func=func, then=then, outcomes=outcomes, **kw)


def expand(
    seat: str,
    make_task: Callable[..., Any],
    *,
    over: Callable[..., Iterable[Any]],
    then: Target | None = None,
    join: Join = ALL,
    max_active: int | None = None,
    **kw: Any,
) -> ExpandNode:
    """The same agent node once per item of `over(upstream)`, built by
    `make_task(upstream, item)` and joined by `join`."""
    return ExpandNode(
        seat=seat,
        make_task=make_task,
        over=over,
        then=then,
        join=join,
        max_active=max_active,
        **kw,
    )
