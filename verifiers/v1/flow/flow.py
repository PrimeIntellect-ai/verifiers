"""Run admitted, ready jobs until idle or drained. Pipelines own scheduling policy."""

from __future__ import annotations

import asyncio
import fcntl
import inspect
import json
import logging
import os
import signal
import socket
from collections import Counter
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
from contextlib import ExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, Self, cast
from uuid import uuid4

from pydantic import BaseModel, JsonValue, TypeAdapter
from typing_extensions import TypeVar

from verifiers.v1.agent import Agents
from verifiers.v1.configs.agent import (
    agent_config_fields,
    resolve_agent,
)
from verifiers.v1.configs.flow import FlowConfig, PoolLimits
from verifiers.v1.flow.calls import (
    INVOCATION,
    CallFailed,
    Failure,
    Live,
    Record,
    Result,
    Success,
    _FlowAgent,
)
from verifiers.v1.flow.events import (
    TRANSITIONS,
    CallEvent,
    CallIdentity,
    CallKind,
    Event,
    Link,
    LinkEvent,
    RunEvent,
    RunReason,
    StageEvent,
    Status,
    Transition,
    append_event,
    now,
)
from verifiers.v1.flow.job import STATE, D, Execution, Job, JobState
from verifiers.v1.interception import make_interception
from verifiers.v1.runtimes import runtime_is_local
from verifiers.v1.runtimes.base import RUN_LABEL_VAR
from verifiers.v1.trace import Error, Trace
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.generic import concrete_type
from verifiers.v1.utils.trace_store import TraceStore, trim_torn_tail

logger = logging.getLogger("verifiers.flow")
_pool_limits = TypeAdapter(PoolLimits)
_cache_inputs = TypeAdapter(dict[str, JsonValue])

T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound=FlowConfig, default=FlowConfig)
JOBS = "jobs"
DRAIN_FILE = "drain"
"""A file of this name in the root drains the flow, as Ctrl-C once does. Remove it to launch again."""
_LINKS: ContextVar[list[Link] | None] = ContextVar("flow_links", default=None)
"""The other jobs the running stage touched, `{job, label}`: the edges between lanes."""


def job_path(root: Path, name: str) -> Path:
    if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
        raise ValueError(f"unsafe job id: {name!r}")
    path = root / JOBS / name
    if not path.resolve().is_relative_to((root / JOBS).resolve()):
        raise ValueError(f"job path escapes root: {name!r}")
    return path


class RunResult(BaseModel):
    """Execution facts; a pipeline decides which outcomes count as success."""

    reason: RunReason
    jobs: dict[str, JobState[Any]]

    @property
    def counts(self) -> dict[Status, int]:
        return dict(Counter(state.status for state in self.jobs.values()))


class Stopped(Exception):
    """The flow is draining: no call starts; the stage's job stays ready to resume."""


class Pools:
    def __init__(self, sizes: dict[str, int]) -> None:
        self.limits = sizes.copy()
        self._used: Counter[str] = Counter()
        self._changed = asyncio.Condition()

    async def resize(self, sizes: PoolLimits) -> None:
        if sizes.keys() != self.limits.keys():
            raise ValueError("pool names must match the configured pools")
        async with self._changed:
            self.limits = sizes
            self._changed.notify_all()

    @asynccontextmanager
    async def hold(self, names: Iterable[str]) -> AsyncIterator[None]:
        """Hold every named pool for the duration; unknown names are unbounded."""
        names = set(names) & self.limits.keys()
        async with self._changed:
            await self._changed.wait_for(
                lambda: all(self._used[n] < self.limits[n] for n in names)
            )
            self._used.update(names)
        try:
            yield
        finally:
            async with self._changed:
                self._used.subtract(names)
                self._changed.notify_all()


def digest(*parts: JsonValue) -> str:
    return sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


F = TypeVar("F", bound=Callable[..., Awaitable[Transition[Any]]])
_CURRENT: ContextVar[Job[Any, Any]] = ContextVar("flow_job")


def stage(method: F) -> F:
    """Mark a named method as a durable job stage."""
    method.__dict__["_flow_stage"] = True
    return method


class Flow(Generic[ConfigT]):
    """Owns a run root: jobs, call results, traces, events, configuration and launch lock."""

    def __init__(self, config: ConfigT, *, root: Path) -> None:
        self.root, self.config = root, config
        config_type = concrete_type(type(self), FlowConfig, origin=Flow) or FlowConfig
        if not isinstance(config, config_type):
            raise TypeError(f"{type(self).__name__} requires {config_type.__name__}")
        self.stages: dict[
            str, Callable[[Job[Any, Self]], Awaitable[Transition[Any]]]
        ] = {
            name: getattr(self, name)
            for name, method in inspect.getmembers(type(self))
            if getattr(method, "_flow_stage", False)
        }
        self.agents: Agents[_FlowAgent]
        label_root = root.resolve()
        self.label = f"flow-{label_root.name[:32]}-{digest(socket.gethostname(), str(label_root))[:12]}"
        self.pools = Pools(config.pools)
        self._pool_text: str | None = None
        self._draining = False
        self._active: dict[str, Execution] = {}

    async def setup(self) -> None:
        """Prepare the pipeline and seed its initial jobs; safe to repeat on resume."""

    def admit(self, job: Job[Any, Self]) -> bool:
        """Whether a ready job may start; earlier admissions are already active."""
        return True

    def exit_code(self, result: RunResult) -> int:
        """Optional pipeline outcome policy. The engine reports execution facts only."""
        return 0

    @property
    def active(self) -> Mapping[str, Execution]:
        """Reserved executions, including their original stage after a live route."""
        return MappingProxyType(self._active)

    @contextmanager
    def _open(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "flow.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"{self.root} is in use by another launch") from None
            previous_label = os.environ.get(RUN_LABEL_VAR)
            try:
                os.environ[RUN_LABEL_VAR] = self.label
                (self.root / "flow.json").write_text(
                    self.config.model_dump_json(indent=1)
                )
                if not (self.root / "pools.json").exists():
                    (self.root / "pools.json").write_text(json.dumps(self.config.pools))
                (self.root / JOBS).mkdir(exist_ok=True)
                trim_torn_tail(self.root / TRANSITIONS)
                trim_torn_tail(self.root / "traces.jsonl")
                self.traces, self.live = (
                    TraceStore(self.root, env="flow"),
                    Live(self.root),
                )
                yield
            finally:
                if previous_label is None:
                    os.environ.pop(RUN_LABEL_VAR, None)
                else:
                    os.environ[RUN_LABEL_VAR] = previous_label

    # -- jobs -----------------------------------------------------------------------------

    def jobs(self) -> list[Job[Any, Self]]:
        return [
            self.job(p.name)
            for p in sorted((self.root / JOBS).iterdir())
            if (p / STATE).is_file()
        ]

    def job(self, name: str) -> Job[Any, Self]:
        job = Job[Any, Self](job_path(self.root, name))
        job.flow = self
        return job

    def create(self, id: str, *, stage: str, data: D) -> Job[D, Self]:
        """Seed a typed job without resetting an existing checkpoint."""
        job = Job[D, Self]._create(
            job_path(self.root, id),
            stage=stage,
            data=data,
            stages=self.stages,
        )
        job.flow = self
        self.touch(id, "seeded")
        return job

    def apply(
        self, id: str, transition: Transition[D], *, expected: int | None = None
    ) -> int:
        """Apply a control now; replacing data requires a settled job and its revision."""
        return self.job(id)._apply(transition, expected=expected).revision

    def touch(self, job: str, label: str) -> None:
        """Note that the running stage acted on another job (created it, released it, made it
        ready): the transition records the link, so a dashboard can draw the edge between lanes."""
        if (links := _LINKS.get()) is not None:
            links.append(Link(job=job, label=label))

    # -- the loop ---------------------------------------------------------------------------

    async def _refresh_pools(self) -> None:
        try:
            text = (self.root / "pools.json").read_text()
            if text == self._pool_text:
                return
            self._pool_text = text
            await self.pools.resize(_pool_limits.validate_json(text, strict=True))
        except (OSError, ValueError) as exc:
            logger.warning(
                "Rejected pools.json; keeping limits %s: %s", self.pools.limits, exc
            )

    async def run(self) -> RunResult:
        """Prepare and run until idle (unless stay_alive) or drain; close owned resources."""
        with self._open():
            await self.setup()
            return await self._schedule()

    async def _schedule(self) -> RunResult:
        running: dict[str, tuple[asyncio.Task[None], ExitStack]] = {}

        def release(name: str) -> asyncio.Task[None]:
            task, lease = running.pop(name)
            lease.close()
            del self._active[name]
            return task

        async with self._serving(), self.live.streamer:
            self.event(RunEvent(type="run_started", label=self.label))
            try:
                while True:
                    await self._refresh_pools()
                    if not self.draining:
                        self._launch(running)
                    if not running:
                        if self.draining or not self.config.stay_alive:
                            break
                        await asyncio.sleep(2)
                        continue
                    done, _ = await asyncio.wait(
                        [task for task, _ in running.values()],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=2,
                    )
                    for name, (task, _) in list(running.items()):
                        if task in done:
                            release(name).result()
            finally:
                # The loop owns reservations even if a task is cancelled before its first turn.
                for task, _ in running.values():
                    task.cancel()
                await asyncio.gather(
                    *(task for task, _ in running.values()), return_exceptions=True
                )
                for name in list(running):
                    release(name)
        result = RunResult(
            reason="draining" if self.draining else "idle",
            jobs={job.id: job.state() for job in self.jobs()},
        )
        self.event(
            RunEvent(type="run_finished", reason=result.reason, counts=result.counts)
        )
        return result

    def _launch(self, running: dict[str, tuple[asyncio.Task[None], ExitStack]]) -> None:
        limit = self.pools.limits.get("jobs")
        for job in self.jobs():
            if self.draining or (limit is not None and len(running) >= limit):
                break
            if job.id in running or job.state().status != "ready":
                continue
            if not self.admit(job):
                continue
            with ExitStack() as stack:
                stack.enter_context(job.executing())
                if job.before.status != "ready":
                    continue
                self._active[job.id] = job.execution
                running[job.id] = (
                    asyncio.create_task(self._stage(job)),
                    stack.pop_all(),
                )

    async def _stage(self, job: Job[Any, Self]) -> None:
        before, execution = job.before, job.execution
        name = execution.stage
        self.event(
            StageEvent(
                type="started",
                job=job.id,
                stage=name,
                execution=execution.id,
                error=None,
            )
        )
        error = None
        links: list[Link] = []
        token = _LINKS.set(links)
        current = _CURRENT.set(job)
        try:
            transition = await self.stages[name](job)
        except (Stopped, asyncio.CancelledError) as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            self.event(
                StageEvent(
                    type="cancelled" if cancelled else "stopped",
                    job=job.id,
                    stage=name,
                    execution=execution.id,
                    error=None,
                )
            )
            if cancelled:
                raise
            return
        except Exception as exc:
            logger.exception("%s/%s failed", job.id, name)
            error = Error(type=type(exc).__name__, message=str(exc))
            transition = Transition(
                outcome="held", reason=f"{type(exc).__name__}: {exc}", status="held"
            )
        finally:
            _LINKS.reset(token)
            _CURRENT.reset(current)
        committed = job._apply(transition, before=before)
        self.event(
            StageEvent(
                type="transition",
                job=job.id,
                stage=name,
                execution=execution.id,
                error=error,
                outcome=transition.outcome,
                to=committed.stage,
                status=committed.status,
                reason=committed.reason,
                report=transition.report,
                revision=committed.revision,
                links=links,
            )
        )

    def event(self, event: Event) -> None:
        logger.info("%s", append_event(self.root / TRANSITIONS, event))

    # -- drain, serving ----------------------------------------------------------------

    def drain(self) -> None:
        if not self._draining:
            self.event(RunEvent(type="drain"))
            self._draining = True

    @property
    def draining(self) -> bool:
        if not self._draining and (self.root / DRAIN_FILE).exists():
            self.drain()
        return self._draining

    @asynccontextmanager
    async def _serving(self) -> AsyncIterator[None]:
        remote = any(
            not runtime_is_local(agent.runtime)
            for agent in agent_config_fields(self.config).values()
        )
        async with make_interception(
            self.config.interception, requires_tunnel=remote
        ) as interception:
            self.agents = Agents(
                self.config,
                lambda _, config: _FlowAgent(
                    self,
                    resolve_agent(
                        config,
                        model=self.config.model,
                        client=self.config.client,
                        sampling=self.config.sampling,
                    ),
                    interception=interception,
                ),
            )
            yield

    @property
    def _job(self) -> Job[Any, Self]:
        """The executing job in this task; shared Flow instances never store a current cursor."""
        job = _CURRENT.get()
        if job.flow is not self:
            raise RuntimeError("call belongs to another flow")
        return job

    def link_from(self, source_execution: str, *, label: str) -> None:
        self.event(
            LinkEvent(
                source_execution=source_execution,
                target_execution=self._job.execution.id,
                label=label,
            )
        )

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        output: type[T] | TypeAdapter[T],
        key: str | None = None,
        cache_inputs: dict[str, JsonValue] | None = None,
        **kwargs: Any,
    ) -> T:
        """Run typed host work; a key reuses successes for the declared inputs."""
        result = await self.attempt(
            func, *args, output=output, key=key, cache_inputs=cache_inputs, **kwargs
        )
        if not result.ok:
            raise CallFailed(result.error, result.trace_id)
        return result.value

    async def attempt(
        self,
        func: Callable[..., Any],
        *args: Any,
        output: type[T] | TypeAdapter[T],
        key: str | None = None,
        cache_inputs: dict[str, JsonValue] | None = None,
        **kwargs: Any,
    ) -> Result[T]:
        """Typed host work whose failure is returned alongside successful siblings."""

        async def execute() -> T:
            value = (
                func(*args, **kwargs)
                if inspect.iscoroutinefunction(func)
                else await run_shielded(asyncio.to_thread(func, *args, **kwargs))
            )
            if inspect.isawaitable(value):
                value = await value
            return adapter.validate_python(value)

        adapter = output if isinstance(output, TypeAdapter) else TypeAdapter(output)
        return await self._record(
            execute, adapter, key=key, cache_inputs=cache_inputs, kind="fn"
        )

    @staticmethod
    async def gather(*calls: Awaitable[T]) -> list[T]:
        """Join every call, including on drain; cancellation settles owned children."""
        children = []
        try:
            for call in calls:
                children.append(asyncio.ensure_future(call))
            results = await asyncio.gather(*children, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return cast(list[T], results)
        finally:
            for child in children:
                if not child.done():
                    child.cancel()
            await asyncio.gather(*children, return_exceptions=True)

    async def _record(
        self,
        execute: Callable[[], Awaitable[T]],
        output: TypeAdapter[T],
        *,
        key: str | None,
        cache_inputs: dict[str, JsonValue] | None,
        kind: CallKind,
    ) -> Result[T]:
        """A value or failure. Successful keyed work is recorded; failures remain retryable."""
        if (key is None) != (cache_inputs is None):
            raise ValueError(
                "key and cache_inputs must be provided together; use {} for no dependencies"
            )
        if cache_inputs is not None:
            cache_inputs = _cache_inputs.validate_python(cache_inputs, strict=True)
        job = self._job
        name = key if key is not None else kind
        cache = digest(kind, key, cache_inputs)[:24] if key is not None else None
        file = self.root / "calls" / job.id / f"{cache}.json" if cache else None
        call = uuid4().hex
        invocation = CallIdentity(
            job=job.id,
            stage=job.execution.stage,
            execution=job.execution.id,
            call=call,
            key=key,
            kind=kind,
            cache=cache,
        )
        token = INVOCATION.set(invocation)
        event = CallEvent(invocation=invocation, status="started")
        try:
            if file is not None and file.exists():
                record = Record.model_validate_json(file.read_text())
                if kind == "agent":
                    value = self.traces.get(record.trace_id or "")
                    if value is None:
                        raise LookupError(f"trace {record.trace_id} is missing")
                    value = cast(T, value)
                else:
                    value = output.validate_python(record.payload)
                event.status = "attached"
                event.source_call = record.call
                event.source_execution = record.execution
                event.trace_id = record.trace_id
                return Success(value, attached=True)
            self.check_running()
            self.event(CallEvent(invocation=invocation, status="started"))
            try:
                value = await execute()
            except (Stopped, asyncio.CancelledError):
                raise
            except Exception as exc:  # noqa: BLE001 - work failures become typed results
                event.status = "failed"
                if isinstance(exc, CallFailed):
                    event.error, event.trace_id = exc.error, exc.trace_id
                else:
                    event.error = Error(type=type(exc).__name__, message=str(exc))
            finally:
                if trace := self.live.current.get(call):
                    event.trace_id = trace.id
                    await self.traces.append(trace)
            if event.error is not None:
                return Failure(event.error, event.trace_id)
            # Publication failures stop the stage; repeating work is an operator decision.
            if file is not None:
                file.parent.mkdir(parents=True, exist_ok=True)
                record = Record(
                    key=name,
                    execution=job.execution.id,
                    call=call,
                    trace_id=cast(Trace, value).id if kind == "agent" else None,
                    payload=None
                    if kind == "agent"
                    else output.dump_python(value, mode="json"),
                )
                tmp = file.with_suffix(".tmp")
                tmp.write_text(record.model_dump_json(indent=1))
                os.replace(tmp, file)
            event.status = "succeeded"
            event.trace_id = value.id if isinstance(value, Trace) else None
            return Success(value)
        except (Stopped, asyncio.CancelledError) as exc:
            event.status = (
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "stopped"
            )
            event.error = None
            raise
        except Exception as exc:
            event.status = "failed"
            event.error = Error(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            INVOCATION.reset(token)
            try:
                if event.status != "started":
                    event.at = now()
                    self.event(event)
            finally:
                if kind == "agent":
                    await self.live.drop(call)

    def check_running(self) -> None:
        """Refuse new work after drain, including work that waited for a pool."""
        if self.draining:
            raise Stopped(self._job.execution.stage)


@contextmanager
def drain_on_interrupt(flow: Flow[Any]) -> Iterator[None]:
    """First SIGINT/SIGTERM drains, second cancels; restore default handlers on exit."""
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    assert task is not None

    def interrupt() -> None:
        if flow.draining:
            task.cancel()
        else:
            logger.warning("draining: calls in flight finish; Ctrl-C again cancels")
            flow.drain()

    signals = (signal.SIGINT, signal.SIGTERM)
    try:
        for sig in signals:
            loop.add_signal_handler(sig, interrupt)
        yield
    finally:
        for sig in signals:
            loop.remove_signal_handler(sig)
