"""What a stage calls: a seat on a task, a command in a runtime, host Python. A call with a
`key` is durable: its result is recorded under the key and the content of its work, and a
stage run again finds it instead of running it. A call without a key just runs.

The result of a failed call is typed (`CallFailed`: the error's type, status code and trace
id), never recorded, so a stage rerun redoes only what did not land.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import typing
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar, cast

from pydantic import BaseModel, TypeAdapter
from pydantic.errors import PydanticSchemaGenerationError
from pydantic_core import to_jsonable_python

from verifiers.v1.agent import Agent
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.utils.retries import trace_should_retry

if TYPE_CHECKING:
    from verifiers.v1.flow.flow import Ctx

T = TypeVar("T")
Kind = Literal["agent", "command", "fn"]
LIVE_EVERY_S = 3.0
"""How often at most a live trace snapshot is rewritten while a seat runs."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class CallFailed(Exception):
    """A call that did not land: `<name>: <error>`. For an agent, `type` and `status_code`
    are the trace's last error and `trace_id` the failed trace, for a stage routing on it."""

    def __init__(
        self,
        message: str,
        type: str = "",
        status_code: int | None = None,
        trace_id: str | None = None,
    ):
        super().__init__(message)
        self.type, self.status_code, self.trace_id = type, status_code, trace_id


@dataclass
class Result(Generic[T]):
    """One item of a spread: the value, or the failure, typed."""

    ok: bool
    value: T | None = None
    error: str | None = None
    type: str | None = None
    status_code: int | None = None
    trace_id: str | None = None
    attached: bool = False
    """The value came from an earlier run's record rather than from running the work now: what
    a stage checks when a later call assumed this one's side effects in a box."""


@dataclass(frozen=True)
class Invocation:
    id: str
    key: str | None
    kind: Kind
    cache: str | None
    attempt: int = 0
    parent: str | None = None


INVOCATION: ContextVar[Invocation] = ContextVar("flow_invocation")


class Record(BaseModel):
    """A durable call's record, `calls/<unit>/<digest>.json`."""

    key: str
    unit: str
    stage: str
    kind: Kind
    execution: str
    call: str
    attempt: int
    payload: Any = None
    trace_id: str | None = None
    started_at: str
    finished_at: str


class Work(ABC, Generic[T]):
    kind: ClassVar[Kind]

    @abstractmethod
    def content(self, ctx: Ctx) -> Any:
        """What keys the call besides its key: the work's inputs, JSON-stable."""

    @abstractmethod
    async def execute(self, ctx: Ctx, name: str) -> T: ...

    @abstractmethod
    def dump(self, ctx: Ctx, value: T) -> dict[str, Any]:
        """The record fields that carry the value: a `payload`, or a `trace_id`."""

    @abstractmethod
    def load(self, ctx: Ctx, record: Record) -> T:
        """The value a record carries; `LookupError` when it cannot be rebuilt."""


def _payload(ctx: Ctx, value: Any) -> Any:
    payload = to_jsonable_python(value)
    if (size := len(json.dumps(payload))) > ctx.config.payload_cap:
        raise ValueError(
            f"call value is {size} bytes, over payload_cap; keep bulk in files"
        )
    return payload


def agent_inputs(config: AgentConfig) -> dict[str, Any]:
    """Model behavior, excluding credentials and execution allowances.

    Request sampling (including max_tokens) is semantic; run token/turn ceilings are not.
    Pipelines can extend or replace this selection in Work.content().
    """
    return {
        "model": config.model,
        "sampling": config.sampling,
        "client": config.client.model_dump(
            include={"type", "base_url", "renderer", "renderer_model_name"}
        )
        if config.client
        else None,
        "harness": config.harness.model_dump(
            exclude={
                "forward_env",
                "mcp_header_env",
                "tool_timeout",
                "exec_timeout",
                "max_turns",
                "max_total_turns",
                "max_total_tokens",
                "max_concurrent_subagents",
            }
        )
        if config.harness
        else None,
        "runtime": config.runtime.model_dump(
            include={"type", "image", "workdir", "network_allow", "network_block"}
        ),
    }


@dataclass(frozen=True)
class AgentWork(Work[Trace[Any, Any, Any]]):
    kind: ClassVar[Kind] = "agent"
    seat: str
    task: Task
    inputs: Any
    """Declared task configuration and external dependencies, including grading.
    These supplement task.data and the resolved seat; use {} when neither applies.
    Host paths, credentials and mutable ambient context should not key completed work.
    """
    runtime: Runtime | None = None
    """A live box to run in; None provisions one for the call."""

    def content(self, ctx: Ctx) -> dict[str, Any]:
        return {
            "kind": "agent",
            "seat": self.seat,
            "task_type": f"{type(self.task).__module__}.{type(self.task).__qualname__}",
            "task": self.task.data.model_dump(exclude={"timeout", "resources"}),
            "agent": agent_inputs(ctx.seat(self.seat)),
            "inputs": self.inputs,
        }

    async def execute(self, ctx: Ctx, name: str) -> Trace:
        flow = ctx.flow
        agent = flow.agent(self.seat)
        held = () if self.runtime is not None else ("runtimes",)
        call = INVOCATION.get().id
        watch = flow.live.watch(ctx.unit.id, call)
        traces: list[Trace] = []

        def finished(trace: Trace, attempt: int) -> None:
            status = (
                "succeeded"
                if trace.ok
                else "failed"
                if trace.is_completed
                else "cancelled"
            )
            ctx.event(
                "rollout",
                status,
                rollout=attempt,
                trace_id=trace.id,
                error=trace.last_error.model_dump(mode="json")
                if trace.last_error
                else None,
            )

        def on_trace(trace: Trace) -> None:
            if traces:
                finished(traces[-1], len(traces))
            traces.append(trace)
            ctx.event("rollout", "started", rollout=len(traces), trace_id=trace.id)
            watch(trace)

        try:
            async with flow.pools.hold(held):
                ctx.check_running()
                async with agent:
                    trace = await self.rollout(agent, on_trace=on_trace)
        finally:
            flow.live.drop(ctx.unit.id, call)
            if traces:
                finished(traces[-1], len(traces))
                for recorded in traces:
                    await flow.traces.append(recorded)
        if not trace.ok:
            last = trace.last_error
            raise CallFailed(
                f"{last.type}: {last.message}" if last else "rollout failed",
                last.type if last else "",
                last.status_code if last else None,
                trace.id,
            )
        return trace

    async def rollout(self, agent: Agent, on_trace: Callable[[Trace], None]) -> Trace:
        """One rollout of the task on `agent`, with the agent's own retry policy. A seat
        driven turn by turn (`agent.interaction`) overrides this alone; `should_retry`
        applies the same policy there."""
        return await agent.run(self.task, runtime=self.runtime, on_trace=on_trace)

    def dump(self, ctx: Ctx, value: Trace) -> dict[str, Any]:
        return {"trace_id": value.id}

    def load(self, ctx: Ctx, record: Record) -> Trace[Any, Any, Any]:
        trace = ctx.flow.traces.get(record.trace_id or "")
        if trace is None:
            raise LookupError(f"trace {record.trace_id} is not in the flow's traces")
        return trace


def should_retry(trace: Trace, agent: Agent, attempt: int) -> bool:
    """Whether a driven rollout that ended in `trace` gets another attempt under the
    agent's `RetryConfig` (`attempt` counts from 0): one retry owner for both call styles."""
    retry = agent.config.retries
    return (
        attempt < retry.max_retries
        and not trace.ok
        and trace_should_retry(trace, retry)
    )


@dataclass(frozen=True)
class CommandWork(Work[ProgramResult]):
    kind: ClassVar[Kind] = "command"
    argv: list[str]
    runtime: Runtime
    inputs: Any
    """Explicit filesystem/artifact dependencies for reuse; {} for an unkeyed command."""
    env: dict[str, str] = field(default_factory=dict)

    def content(self, ctx: Ctx) -> list[Any]:
        return ["command", self.argv, self.env, self.inputs]

    async def execute(self, ctx: Ctx, name: str) -> ProgramResult:
        return await self.runtime.run(self.argv, self.env)

    def dump(self, ctx: Ctx, value: ProgramResult) -> dict[str, Any]:
        return {"payload": _payload(ctx, value)}

    def load(self, ctx: Ctx, record: Record) -> ProgramResult:
        return ProgramResult(**record.payload)


@dataclass(frozen=True)
class FnWork(Work[T]):
    kind: ClassVar[Kind] = "fn"
    func: Callable[..., T | Awaitable[T]]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    inputs: Any = None
    """Optional semantic inputs in place of arguments that contain operational settings."""

    def content(self, ctx: Ctx) -> list[Any]:
        return [
            "fn",
            f"{self.func.__module__}.{cast(Any, self.func).__qualname__}",
            self.inputs if self.inputs is not None else [self.args, self.kwargs],
        ]

    async def execute(self, ctx: Ctx, name: str) -> T:
        if inspect.iscoroutinefunction(self.func):
            return await self.func(*self.args, **self.kwargs)
        # A sync function runs off the loop; a thread cannot be cancelled, so a cancel waits
        # for it to finish before it propagates and no retry overlaps the work.
        future = asyncio.ensure_future(
            asyncio.to_thread(self.func, *self.args, **self.kwargs)
        )
        try:
            value = await asyncio.shield(future)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await asyncio.shield(future)
            raise
        return cast(T, await value if inspect.isawaitable(value) else value)

    def dump(self, ctx: Ctx, value: T) -> dict[str, Any]:
        return {"payload": _payload(ctx, value)}

    def load(self, ctx: Ctx, record: Record) -> T:
        try:
            hint = typing.get_type_hints(self.func).get("return")
            return (
                TypeAdapter(hint).validate_python(record.payload)
                if hint not in (None, Any)
                else record.payload
            )
        except (NameError, TypeError, PydanticSchemaGenerationError):
            return record.payload


def agent(
    seat: str, task: Task, *, inputs: Any, runtime: Runtime | None = None
) -> AgentWork:
    """`Agent.run(task)` on the config field `seat`; the value is the `Trace`. What a later
    stage reads back must sit on `trace.info`, metrics or rewards: `trace.state` is not
    serialized. `inputs` declares task configuration and external dependencies affecting
    the returned rollout AND its score; use {} only when neither adds dependencies."""
    return AgentWork(seat=seat, task=task, inputs=inputs, runtime=runtime)


def command(
    argv: list[str], *, runtime: Runtime, inputs: Any, env: dict[str, str] | None = None
) -> CommandWork:
    """`runtime.run(argv)`; the value is the `ProgramResult`, whatever the exit code."""
    return CommandWork(
        argv=list(argv), runtime=runtime, inputs=inputs, env=dict(env or {})
    )


def fn(
    func: Callable[..., T | Awaitable[T]], *args: Any, inputs: Any = None, **kwargs: Any
) -> FnWork[T]:
    """`func(*args, **kwargs)` on the host, sync or async. The arguments key the call, so
    they must be JSON-stable. Supply `inputs` to declare semantic dependencies instead."""
    return FnWork[T](func=func, args=args, kwargs=kwargs, inputs=inputs)


class Live:
    """One snapshot file per seat in flight, `live/<unit>--<name>.json`, rewritten as the
    trace changes (at most every few seconds) and removed when the call ends: what a
    monitor reads to see a running seat turn by turn. Visibility, not durability."""

    def __init__(self, root: Path) -> None:
        self.dir = root / "live"
        self.dir.mkdir(exist_ok=True)
        for stale in self.dir.glob("*.json"):
            stale.unlink()
        self._due: dict[Path, asyncio.TimerHandle] = {}
        self._active: set[Path] = set()

    def _file(self, unit: str, name: str) -> Path:
        return self.dir / f"{unit}--{name.replace('/', '__')}.json"

    def watch(self, unit: str, name: str) -> Callable[[Trace], None]:
        file = self._file(unit, name)
        self._active.add(file)

        def write(trace: Trace) -> None:
            self._due.pop(file, None)
            if file not in self._active:
                return
            tmp = file.with_suffix(".tmp")
            tmp.write_text(trace.model_dump_json())
            os.replace(tmp, file)

        def changed(trace: Trace) -> None:
            if file in self._active and file not in self._due:
                loop = asyncio.get_running_loop()
                self._due[file] = loop.call_later(LIVE_EVERY_S, write, trace)

        def on_trace(trace: Trace) -> None:
            if (due := self._due.pop(file, None)) is not None:
                due.cancel()
            trace.watch(changed)
            changed(trace)

        return on_trace

    def drop(self, unit: str, name: str) -> None:
        file = self._file(unit, name)
        self._active.discard(file)
        if (due := self._due.pop(file, None)) is not None:
            due.cancel()
        file.unlink(missing_ok=True)
