"""Agent and host work with explicit reuse inputs and typed persisted results."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, JsonValue, TypeAdapter

from verifiers.v1.agent import Agent, Interaction
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.flow.events import CallIdentity
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Error, Trace

if TYPE_CHECKING:
    from verifiers.v1.flow.flow import Flow

T = TypeVar("T")
LIVE_EVERY_S = 3.0
"""How often at most a live trace snapshot is rewritten while a call runs."""


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


INVOCATION: ContextVar[CallIdentity] = ContextVar("flow_invocation")


class Record(BaseModel):
    """A durable call's record, `calls/<unit>/<digest>.json`."""

    key: str
    execution: str
    call: str
    payload: JsonValue = None
    trace_id: str | None = None


class _FlowAgent(Agent):
    """Native execution with the owning Flow's call records and traces."""

    def __init__(self, flow: Flow[Any], config: AgentConfig) -> None:
        super().__init__(config, interception=flow.interception)
        self.flow = flow

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
        collect_artifacts: bool = False,
        key: str | None = None,
        inputs: JsonValue | BaseModel | None = None,
        interact: Callable[[Interaction], Awaitable[None]] | None = None,
    ) -> Trace:
        """Run or attach a trace; borrowed runtimes remain owned by their caller."""
        result = await self.attempt(
            task,
            runtime=runtime,
            tools=tools,
            on_trace=on_trace,
            collect_artifacts=collect_artifacts,
            key=key,
            inputs=inputs,
            interact=interact,
        )
        if not result.ok:
            raise CallFailed(result.error, result.trace_id)
        return result.value

    async def attempt(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
        collect_artifacts: bool = False,
        key: str | None = None,
        inputs: JsonValue | BaseModel | None = None,
        interact: Callable[[Interaction], Awaitable[None]] | None = None,
    ) -> Result[Trace]:
        """The same recorded run, returning failure so parallel siblings can finish."""
        if interact is not None and collect_artifacts:
            raise ValueError("interaction does not support collect_artifacts")

        async def execute() -> Trace:
            flow = self.flow
            invocation = INVOCATION.get()
            watch = flow.live.watch(invocation.unit, invocation.call, on_trace)

            try:
                if interact is None:
                    trace = await super(_FlowAgent, self).run(
                        task,
                        runtime=runtime,
                        tools=tools,
                        on_trace=watch,
                        collect_artifacts=collect_artifacts,
                    )
                else:
                    async with super(_FlowAgent, self).interaction(
                        task,
                        runtime=runtime,
                        tools=tools,
                        on_trace=watch,
                    ) as interaction:
                        await interact(interaction)
                    trace = interaction.trace
            except Exception as exc:
                current = flow.live.current.get(invocation.call)
                if current is not None and current.last_error is not None:
                    raise CallFailed(current.last_error, current.id) from exc
                raise
            if not trace.ok:
                raise CallFailed(
                    trace.last_error
                    or Error(type="RolloutError", message="rollout failed"),
                    trace.id,
                )
            return trace

        return await self.flow._record(
            execute, TypeAdapter(Trace), key=key, inputs=inputs, kind="agent"
        )

    @asynccontextmanager
    async def provision(self, task: Task | None = None) -> AsyncIterator[Runtime]:
        self.flow.check_running()
        async with super().provision(task) as runtime:
            yield runtime


class Live:
    """Throttled trace snapshots at `live/<unit>--<call>.json`, removed when calls end.
    These let monitors inspect running calls; completed traces are stored separately."""

    def __init__(self, root: Path) -> None:
        self.dir = root / "live"
        self.dir.mkdir(exist_ok=True)
        for stale in self.dir.glob("*.json"):
            stale.unlink()
        self.current: dict[str, Trace] = {}
        self._active: dict[Path, asyncio.TimerHandle | None] = {}

    def _file(self, unit: str, call: str) -> Path:
        return self.dir / f"{unit}--{call}.json"

    def watch(
        self,
        unit: str,
        call: str,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Callable[[Trace], None]:
        file = self._file(unit, call)
        self._active[file] = None

        def write(trace: Trace) -> None:
            if self.current.get(call) is not trace:
                return
            self._active[file] = None
            tmp = file.with_suffix(".tmp")
            tmp.write_text(trace.model_dump_json())
            os.replace(tmp, file)

        def changed(trace: Trace) -> None:
            if self.current.get(call) is trace and self._active[file] is None:
                loop = asyncio.get_running_loop()
                self._active[file] = loop.call_later(LIVE_EVERY_S, write, trace)

        def started(trace: Trace) -> None:
            if (due := self._active.get(file)) is not None:
                due.cancel()
            self.current[call] = trace
            trace.watch(changed)
            write(trace)
            if on_trace is not None:
                on_trace(trace)

        return started

    def drop(self, unit: str, call: str) -> None:
        file = self._file(unit, call)
        if (due := self._active.pop(file, None)) is not None:
            due.cancel()
        self.current.pop(call, None)
        file.unlink(missing_ok=True)
