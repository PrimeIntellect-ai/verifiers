"""What a step runs: a seat on a task, a command in a runtime, or host Python."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Literal, TypeVar

from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

T = TypeVar("T")

WorkKind = Literal["agent", "command", "fn"]


class Work(ABC, Generic[T]):
    """One unit of step work whose value is a `T`; `kind` is what the ledger records."""

    kind: ClassVar[WorkKind]

    @abstractmethod
    def content(self) -> list[Any]:
        """What keys the step besides its place: the work's inputs, JSON-stable."""


@dataclass(frozen=True)
class AgentWork(Work[Trace]):
    kind: ClassVar[WorkKind] = "agent"
    seat: str
    task: Task
    runtime: Runtime | None = None
    """A live box to run in; None provisions one for the step."""

    def content(self) -> list[Any]:
        return ["agent", self.seat, type(self.task).__name__, self.task.data]


@dataclass(frozen=True)
class CommandWork(Work[ProgramResult]):
    kind: ClassVar[WorkKind] = "command"
    argv: list[str]
    runtime: Runtime
    env: dict[str, str] = field(default_factory=dict)

    def content(self) -> list[Any]:
        return ["command", self.argv, self.env]


@dataclass(frozen=True)
class FnWork(Work[T]):
    kind: ClassVar[WorkKind] = "fn"
    func: Callable[..., T | Awaitable[T]]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def content(self) -> list[Any]:
        return ["fn", self.func.__qualname__, self.args, self.kwargs]


def agent(seat: str, task: Task, *, runtime: Runtime | None = None) -> AgentWork:
    """`Agent.run(task)` on the config field `seat`; the value is the `Trace`. The task's
    prompt must be complete at construction (the rollout reads it before `setup` runs),
    and anything a later step reads back must sit on `trace.info`, metrics or rewards:
    `trace.state` is not serialized and does not survive a resume."""
    return AgentWork(seat=seat, task=task, runtime=runtime)


def command(
    argv: list[str], *, runtime: Runtime, env: dict[str, str] | None = None
) -> CommandWork:
    """`runtime.run(argv)`; the value is the `ProgramResult`, whatever the exit code."""
    return CommandWork(argv=list(argv), runtime=runtime, env=dict(env or {}))


def fn(func: Callable[..., T | Awaitable[T]], *args: Any, **kwargs: Any) -> FnWork[T]:
    """`func(*args, **kwargs)` on the host, sync or async; the value is its return. The
    arguments key the step, so they must be JSON-stable: a bound method, closure or live
    object digests to its address and the step never attaches on resume."""
    return FnWork(func=func, args=args, kwargs=kwargs)
