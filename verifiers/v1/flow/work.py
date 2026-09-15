"""What a step runs: a seat on a task, a command in a runtime, or host Python."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task


@dataclass(frozen=True)
class AgentWork:
    seat: str
    task: Task
    runtime: Runtime | None = None
    """A live box to run in; None provisions one for the step."""

    def content(self) -> Any:
        return ["agent", self.seat, type(self.task).__name__, self.task.data]


@dataclass(frozen=True)
class CommandWork:
    argv: list[str]
    runtime: Runtime
    env: dict[str, str] = field(default_factory=dict)

    def content(self) -> Any:
        return ["command", self.argv, self.env]


@dataclass(frozen=True)
class FnWork:
    func: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def content(self) -> Any:
        return ["fn", self.func.__qualname__, self.args, self.kwargs]


Work = AgentWork | CommandWork | FnWork


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


def fn(func: Callable[..., Any], *args: Any, **kwargs: Any) -> FnWork:
    """`func(*args, **kwargs)` on the host, sync or async; the value is its return. The
    arguments key the step, so they must be JSON-stable: a bound method, closure or live
    object digests to its address and the step never attaches on resume."""
    return FnWork(func=func, args=args, kwargs=kwargs)
