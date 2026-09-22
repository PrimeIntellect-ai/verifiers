"""Agent and host work with explicit reuse inputs and typed persisted results."""

from __future__ import annotations

import json
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
from verifiers.v1.interception import Interception
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.runtimes import Runtime
from verifiers.v1.serve.delta import DeltaStreamer
from verifiers.v1.task import Task
from verifiers.v1.trace import Error, Trace

if TYPE_CHECKING:
    from verifiers.v1.flow.flow import Flow

T = TypeVar("T")


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

    def __init__(
        self, flow: Flow[Any], config: AgentConfig, *, interception: Interception
    ) -> None:
        super().__init__(config, interception=interception)
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
            watch = flow.live.watch(invocation, on_trace)

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
    """Current attempts streamed to `live/<trace>.jsonl`; completed traces live separately."""

    def __init__(self, root: Path) -> None:
        self.dir = root / "live"
        self.dir.mkdir(exist_ok=True)
        for stale in self.dir.glob("*.jsonl"):
            stale.unlink()
        self.current: dict[str, Trace] = {}
        self._dispatch: dict[str, dict] = {}
        self.streamer = DeltaStreamer(self.current.values, self._send)

    async def _send(self, delta: dict) -> None:
        file = self.dir / f"{delta['trace']}.jsonl"
        if delta.get("discard"):
            file.unlink(missing_ok=True)
        else:
            if "open" in delta:
                delta = {**delta, "dispatch": self._dispatch[delta["trace"]]}
            with file.open("a") as out:
                out.write(json.dumps(delta, default=str) + "\n")

    def watch(
        self,
        invocation: CallIdentity,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Callable[[Trace], None]:
        def started(trace: Trace) -> None:
            if previous := self.current.get(invocation.call):
                self._dispatch.pop(previous.id)
            self.current[invocation.call] = trace
            self._dispatch[trace.id] = {
                "id": invocation.call,
                "kind": "flow",
                "task": invocation.unit,
                "started": trace.timing.start,
                "invocation": invocation.model_dump(mode="json"),
            }
            self.streamer.watch(trace)
            if on_trace is not None:
                on_trace(trace)

        return started

    async def drop(self, call: str) -> None:
        if trace := self.current.pop(call, None):
            self._dispatch.pop(trace.id)
            await self.streamer.flush()
