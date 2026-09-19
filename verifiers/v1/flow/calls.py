"""Agent and host work with explicit reuse inputs and typed persisted results."""

from __future__ import annotations

import asyncio
import inspect
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypedDict, TypeVar

from pydantic import BaseModel, JsonValue, TypeAdapter

from verifiers.v1.agent import Agent
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Error, Trace
from verifiers.v1.utils.aio import run_shielded

if TYPE_CHECKING:
    from verifiers.v1.flow.flow import Ctx

T = TypeVar("T")
LIVE_EVERY_S = 3.0
"""How often at most a live trace snapshot is rewritten while a seat runs."""


class CallFailed(Exception):
    """A failed call, carrying the native error and its trace when available."""

    def __init__(self, error: Error, trace_id: str | None = None):
        super().__init__(f"{error.type}: {error.message}")
        self.error, self.trace_id = error, trace_id


@dataclass(frozen=True)
class Success(Generic[T]):
    value: T
    attached: bool = False
    """Reuse restores a value, never its sandbox side effects."""
    ok: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True)
class Failure:
    error: Error
    trace_id: str | None = None
    ok: Literal[False] = field(default=False, init=False)


Result = Success[T] | Failure


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
    async def execute(self, ctx: Ctx[Any, Any]) -> T: ...

    @abstractmethod
    def dump(self, value: T) -> StoredValue:
        """The record fields that carry the value: a `payload`, or a `trace_id`."""

    @abstractmethod
    def load(self, ctx: Ctx[Any, Any], record: Record) -> T:
        """The value a record carries; `LookupError` when it cannot be rebuilt."""


@dataclass(frozen=True)
class AgentWork(Work[Trace[Any, Any, Any]]):
    kind: ClassVar[str] = "agent"
    seat: str
    task: Task
    runtime: Runtime | None = None
    inputs: JsonValue | BaseModel | None = None

    async def execute(self, ctx: Ctx[Any, Any]) -> Trace:
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
                error=trace.last_error,
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
        except Exception as exc:
            if traces and traces[-1].last_error is not None:
                raise CallFailed(traces[-1].last_error, traces[-1].id) from exc
            raise
        finally:
            flow.live.drop(ctx.unit.id, call)
            if traces:
                finished(traces[-1], len(traces))
                for recorded in traces:
                    await flow.traces.append(recorded)
        if not trace.ok:
            raise CallFailed(
                trace.last_error
                or Error(type="RolloutError", message="rollout failed"),
                trace.id,
            )
        return trace

    async def rollout(self, agent: Agent, on_trace: Callable[[Trace], None]) -> Trace:
        """Use native retries. Override for caller-driven `agent.interaction` work."""
        return await agent.run(self.task, runtime=self.runtime, on_trace=on_trace)

    def dump(self, value: Trace) -> StoredValue:
        return {"trace_id": value.id}

    def load(self, ctx: Ctx[Any, Any], record: Record) -> Trace[Any, Any, Any]:
        trace = ctx.flow.traces.get(record.trace_id or "")
        if trace is None:
            raise LookupError(f"trace {record.trace_id} is not in the flow's traces")
        return trace


@dataclass(frozen=True)
class FnWork(Work[T]):
    kind: ClassVar[str] = "fn"
    func: Callable[..., T | Awaitable[T]]
    output: TypeAdapter[T]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    inputs: JsonValue | BaseModel | None = None

    async def execute(self, ctx: Ctx[Any, Any]) -> T:
        if inspect.iscoroutinefunction(self.func):
            value = await self.func(*self.args, **self.kwargs)
        else:
            value = await run_shielded(
                asyncio.to_thread(self.func, *self.args, **self.kwargs)
            )
        return self.output.validate_python(value)

    def dump(self, value: T) -> StoredValue:
        return {"payload": self.output.dump_python(value, mode="json")}

    def load(self, ctx: Ctx[Any, Any], record: Record) -> T:
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
