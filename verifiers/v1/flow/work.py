"""What a step runs: a seat on a task, a command in a runtime, or host Python. Each
kind knows how to execute against a row's `Ctx`, what of its value the record keeps,
and how the record rebuilds that value on a resume."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import typing
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar

from pydantic import TypeAdapter
from pydantic.errors import PydanticSchemaGenerationError
from pydantic_core import to_jsonable_python

from verifiers.v1.agent import Agent, make_agent
from verifiers.v1.flow.config import RUNTIMES
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

if TYPE_CHECKING:
    from verifiers.v1.flow.ledger import Ledger, StepRecord
    from verifiers.v1.flow.run import Ctx

T = TypeVar("T")

WorkKind = Literal["agent", "command", "fn"]


class Oversized(ValueError):
    """A step value over `payload_cap`: no retry shrinks it, so the step fails at once."""


class RolloutFailed(Exception):
    """An agent step whose trace ended in an error: the step's attempt failed. `type` and
    `status_code` are the trace's last error, for a flow routing on `StepFailed.__cause__`."""

    def __init__(self, message: str, type: str = "", status_code: int | None = None):
        super().__init__(message)
        self.type, self.status_code = type, status_code


def _payload(ctx: Ctx, value: Any) -> Any:
    """A step value as the record keeps it, under `payload_cap`: bulk belongs in traces or files."""
    payload = to_jsonable_python(value)
    if (size := len(json.dumps(payload))) > ctx.config.payload_cap:
        raise Oversized(
            f"step value is {size} bytes, over payload_cap; keep bulk in traces or files"
        )
    return payload


class Work(ABC, Generic[T]):
    """One unit of step work whose value is a `T`; `kind` is what the ledger records."""

    kind: ClassVar[WorkKind]

    @abstractmethod
    def content(self, ctx: Ctx) -> list[Any]:
        """What keys the step besides its place: the work's inputs, JSON-stable."""

    @abstractmethod
    async def execute(self, ctx: Ctx) -> T:
        """Run once; the value."""

    @abstractmethod
    def dump(self, ctx: Ctx, value: T) -> dict[str, Any]:
        """The record fields that carry the value: a `payload`, or a `trace_id`."""

    @abstractmethod
    def load(self, ledger: Ledger, record: StepRecord) -> T:
        """The value a completed record carries; `LookupError` when it cannot be
        rebuilt, and the step runs again."""


@dataclass(frozen=True)
class AgentWork(Work[Trace]):
    kind: ClassVar[WorkKind] = "agent"
    seat: str
    task: Task
    runtime: Runtime | None = None
    """A live box to run in; None provisions one for the step."""

    def content(self, ctx: Ctx) -> list[Any]:
        # The resolved seat — the model and effort the rollout runs under — keys the
        # step too, so a seat change re-runs that seat's steps only.
        seat = ctx.run.seat(self.seat).model_dump(mode="json")
        return ["agent", self.seat, type(self.task).__name__, self.task.data, seat]

    async def execute(self, ctx: Ctx) -> Trace:
        run = ctx.run
        agent = make_agent(run.seat(self.seat), interception=run.interception)
        held = () if self.runtime is not None else (RUNTIMES,)
        async with run.pools.hold(held), agent:
            trace = await self.rollout(agent)
        await run.ledger.append(trace)
        if not trace.ok:
            if (last := trace.last_error) is None:
                raise RolloutFailed("rollout failed")
            raise RolloutFailed(
                f"{last.type}: {last.message}", last.type, last.status_code
            )
        return trace

    async def rollout(self, agent: Agent) -> Trace:
        """One rollout of the task on `agent`. A subclass that drives the seat differently
        (turn by turn, through `agent.interaction`) overrides this alone; the pool, the
        ledger and the failure rule stay here."""
        return await agent.run(self.task, runtime=self.runtime)

    def dump(self, ctx: Ctx, value: Trace) -> dict[str, Any]:
        return {"trace_id": value.id}

    def load(self, ledger: Ledger, record: StepRecord) -> Trace:
        trace = ledger.trace(record.trace_id or "")
        if trace is None:
            raise LookupError(f"trace {record.trace_id} is not in the run's traces")
        return trace


@dataclass(frozen=True)
class CommandWork(Work[ProgramResult]):
    kind: ClassVar[WorkKind] = "command"
    argv: list[str]
    runtime: Runtime
    env: dict[str, str] = field(default_factory=dict)

    def content(self, ctx: Ctx) -> list[Any]:
        return ["command", self.argv, self.env]

    async def execute(self, ctx: Ctx) -> ProgramResult:
        return await self.runtime.run(self.argv, self.env)

    def dump(self, ctx: Ctx, value: ProgramResult) -> dict[str, Any]:
        return {"payload": _payload(ctx, value)}

    def load(self, ledger: Ledger, record: StepRecord) -> ProgramResult:
        return ProgramResult(**record.payload)


@dataclass(frozen=True)
class FnWork(Work[T]):
    kind: ClassVar[WorkKind] = "fn"
    func: Callable[..., T | Awaitable[T]]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def content(self, ctx: Ctx) -> list[Any]:
        name = f"{self.func.__module__}.{self.func.__qualname__}"
        return ["fn", name, self.args, self.kwargs]

    async def execute(self, ctx: Ctx) -> T:
        # A sync function runs off the loop: a build or a shell-out in it must not stall
        # every other row's turns. A thread cannot be cancelled, so a timeout or a cancel
        # waits for it to finish before it propagates: no retry overlaps the work.
        if inspect.iscoroutinefunction(self.func):
            return await self.func(*self.args, **self.kwargs)
        future = asyncio.ensure_future(
            asyncio.to_thread(self.func, *self.args, **self.kwargs)
        )
        try:
            value = await asyncio.shield(
                future
            )  # a cancel must not cancel the thread's task
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await asyncio.shield(future)
            raise
        return await value if inspect.isawaitable(value) else value

    def dump(self, ctx: Ctx, value: T) -> dict[str, Any]:
        return {"payload": _payload(ctx, value)}

    def load(self, ledger: Ledger, record: StepRecord) -> T:
        """The payload, validated against the function's return annotation when it
        has one pydantic can build (a model, a list of them, a dataclass...)."""
        try:
            hint = typing.get_type_hints(self.func).get("return")
        except (NameError, TypeError):
            hint = None
        if hint is None or hint is Any:
            return record.payload
        try:
            adapter = TypeAdapter(hint)
        except PydanticSchemaGenerationError:
            return record.payload
        return adapter.validate_python(record.payload)


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
    arguments key the step, so they must be JSON-stable (anything else is an error): a bound
    method's instance state is not part of the key. A sync function cannot be interrupted:
    a `timeout` on its step fails the step once the function returns, and hard termination
    needs a subprocess with its own timeout."""
    return FnWork(func=func, args=args, kwargs=kwargs)
