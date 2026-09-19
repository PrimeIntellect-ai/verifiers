"""Agent and host work with explicit reuse inputs and typed persisted results."""

from __future__ import annotations

import asyncio
import inspect
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypedDict, TypeVar

from pydantic import BaseModel, JsonValue, TypeAdapter

from verifiers.v1.agent import Agent
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.retries import trace_should_retry

if TYPE_CHECKING:
    from verifiers.v1.flow.flow import Ctx

T = TypeVar("T")
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
    call: str
    key: str | None
    kind: str
    cache: str | None


INVOCATION: ContextVar[Invocation] = ContextVar("flow_invocation")


class Record(BaseModel):
    """A durable call's record, `calls/<unit>/<digest>.json`."""

    key: str
    execution: str
    call: str
    payload: JsonValue = None
    trace_id: str | None = None


class StoredValue(TypedDict, total=False):
    payload: JsonValue
    trace_id: str


class Work(ABC, Generic[T]):
    kind: ClassVar[str]
    inputs: JsonValue | BaseModel | None = None
    """The complete reuse identity besides the call key. Required for keyed work;
    use {} for no dependencies. Core adds no task, configuration, or code identity."""

    @abstractmethod
    async def execute(self, ctx: Ctx, name: str) -> T: ...

    @abstractmethod
    def dump(self, ctx: Ctx, value: T) -> StoredValue:
        """The record fields that carry the value: a `payload`, or a `trace_id`."""

    @abstractmethod
    def load(self, ctx: Ctx, record: Record) -> T:
        """The value a record carries; `LookupError` when it cannot be rebuilt."""


@dataclass(frozen=True)
class AgentWork(Work[Trace[Any, Any, Any]]):
    kind: ClassVar[str] = "agent"
    seat: str
    task: Task
    runtime: Runtime | None = None
    inputs: JsonValue | BaseModel | None = None

    async def execute(self, ctx: Ctx, name: str) -> Trace:
        flow = ctx.flow
        agent = flow.agent(self.seat)
        held = () if self.runtime is not None else ("runtimes",)
        call = INVOCATION.get().call
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

    def dump(self, ctx: Ctx, value: Trace) -> StoredValue:
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
class FnWork(Work[T]):
    kind: ClassVar[str] = "fn"
    func: Callable[..., T | Awaitable[T]]
    output: TypeAdapter[T]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    inputs: JsonValue | BaseModel | None = None

    async def execute(self, ctx: Ctx, name: str) -> T:
        if inspect.iscoroutinefunction(self.func):
            value = await self.func(*self.args, **self.kwargs)
        else:
            value = await run_shielded(
                asyncio.to_thread(self.func, *self.args, **self.kwargs)
            )
        return self.output.validate_python(value)

    def dump(self, ctx: Ctx, value: T) -> StoredValue:
        return {"payload": self.output.dump_python(value, mode="json")}

    def load(self, ctx: Ctx, record: Record) -> T:
        return self.output.validate_python(record.payload)


def agent(
    seat: str,
    task: Task,
    *,
    inputs: JsonValue | BaseModel | None = None,
    runtime: Runtime | None = None,
) -> AgentWork:
    """A native agent rollout. Cached traces retain durable info, never live state."""
    return AgentWork(seat=seat, task=task, inputs=inputs, runtime=runtime)


def fn(
    func: Callable[..., T | Awaitable[T]],
    *args: Any,
    output: type[T] | TypeAdapter[T],
    inputs: JsonValue | BaseModel | None = None,
    **kwargs: Any,
) -> FnWork[T]:
    """Host work with an explicit output type for both fresh and cached results."""
    adapter = output if isinstance(output, TypeAdapter) else TypeAdapter(output)
    return FnWork[T](func=func, output=adapter, args=args, kwargs=kwargs, inputs=inputs)


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
