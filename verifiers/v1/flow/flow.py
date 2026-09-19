"""Run admitted, ready units until quiescence or drain. Pipelines own scheduling policy."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import signal
import socket
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any, Generic, Self, cast
from uuid import uuid4

from pydantic import BaseModel
from pydantic_core import to_jsonable_python
from typing_extensions import TypeVar

from verifiers.v1.configs.agent import (
    AgentConfig,
    agent_config_fields,
    resolve_agent,
)
from verifiers.v1.flow.calls import (
    INVOCATION,
    CallFailed,
    Failure,
    Live,
    Record,
    Result,
    Success,
    Work,
)
from verifiers.v1.flow.config import FlowConfig
from verifiers.v1.flow.events import (
    CallEvent,
    EventRecord,
    Invocation,
    Link,
    RunEvent,
    RunReason,
    StageEvent,
    Status,
    append_event,
)
from verifiers.v1.flow.traces import Traces, trim_torn_tail
from verifiers.v1.flow.unit import D, Execution, Transition, Unit, UnitState
from verifiers.v1.interception import Interception, make_interception
from verifiers.v1.runtimes import (
    Runtime,
    provision_runtime,
    runtime_is_local,
)
from verifiers.v1.runtimes.base import RUN_LABEL_VAR
from verifiers.v1.task import Task
from verifiers.v1.trace import Error, Trace
from verifiers.v1.utils.compile import resolve_runtime_config

logger = logging.getLogger("verifiers.flow")

T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound=FlowConfig, default=FlowConfig)
UNITS = "units"
DRAIN_FILE = "drain"
"""A file of this name in the root drains the flow, as Ctrl-C once does. Remove it to launch again."""
TRANSITIONS = "transitions.jsonl"
"""One line per stage start and per transition: what a dashboard follows."""
_LINKS: ContextVar[list[Link] | None] = ContextVar("flow_links", default=None)
"""The other units the running stage touched, `{unit, label}`: the edges between lanes."""


def unit_path(root: Path, name: str) -> Path:
    if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
        raise ValueError(f"unsafe unit id: {name!r}")
    path = root / UNITS / name
    if not path.resolve().is_relative_to((root / UNITS).resolve()):
        raise ValueError(f"unit path escapes root: {name!r}")
    return path


class RunResult(BaseModel):
    """Execution facts; a pipeline decides which outcomes count as success."""

    reason: RunReason
    units: dict[str, UnitState[Any]]

    @property
    def counts(self) -> dict[Status, int]:
        return dict(Counter(state.status for state in self.units.values()))


class Stopped(Exception):
    """The flow is draining: no call starts; the stage's unit stays ready to resume."""


class Pools:
    def __init__(self, sizes: dict[str, int]) -> None:
        self._pools = {name: asyncio.Semaphore(size) for name, size in sizes.items()}

    @asynccontextmanager
    async def hold(self, names: Iterable[str]) -> AsyncIterator[None]:
        """Hold every named pool for the duration; unknown names are unbounded."""
        async with AsyncExitStack() as stack:
            for name in sorted(set(names)):
                if pool := self._pools.get(name):
                    await stack.enter_async_context(pool)
            yield


def digest(*parts: Any) -> str:
    return sha256(
        json.dumps(to_jsonable_python(parts), sort_keys=True).encode()
    ).hexdigest()


Stage = Callable[["Ctx[Any, ConfigT]"], Awaitable[Transition[Any]]]
"""A stage: `async def stage(ctx) -> Transition`."""


@dataclass(frozen=True)
class Pipeline(Generic[ConfigT]):
    stages: dict[str, Stage[ConfigT]]
    admit: Callable[[Unit[Any], Flow[ConfigT]], bool] | None = None
    """Whether a ready, inactive unit may start. Earlier admissions are already active."""


class Flow(Generic[ConfigT]):
    """Owns a run root: units, call results, traces, events, configuration and launch lock."""

    def __init__(
        self, root: Path, config: ConfigT, pipeline: Pipeline[ConfigT]
    ) -> None:
        self.root, self.config, self.pipeline = root, config, pipeline
        label_root = root.resolve()
        self.label = f"flow-{label_root.name[:32]}-{digest(socket.gethostname(), str(label_root))[:12]}"
        self.pools = Pools(config.pools)
        self.interception: Interception | None = None
        self._draining = False
        self._lock: IO[str] | None = None
        self._active: dict[str, Execution] = {}

    @property
    def active(self) -> Mapping[str, Execution]:
        """Reserved executions, including their original stage after a live route."""
        return MappingProxyType(self._active)

    async def __aenter__(self) -> Self:
        self.root.mkdir(parents=True, exist_ok=True)
        lock = (self.root / "flow.lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError(f"{self.root} is in use by another launch") from None
        self._lock = lock
        self._previous_label = os.environ.get(RUN_LABEL_VAR)
        try:
            os.environ[RUN_LABEL_VAR] = self.label
            (self.root / "flow.json").write_text(self.config.model_dump_json(indent=1))
            (self.root / "calls").mkdir(exist_ok=True)
            (self.root / UNITS).mkdir(exist_ok=True)
            trim_torn_tail(self.root / TRANSITIONS)
            trim_torn_tail(self.root / "traces.jsonl")
            self.traces, self.live = Traces(self.root), Live(self.root)
            return self
        except BaseException:
            await self.__aexit__()
            raise

    async def __aexit__(self, *exc: object) -> None:
        if self._previous_label is None:
            os.environ.pop(RUN_LABEL_VAR, None)
        else:
            os.environ[RUN_LABEL_VAR] = self._previous_label
        assert self._lock is not None
        self._lock.close()
        self._lock = None

    # -- units -----------------------------------------------------------------------------

    def units(self) -> list[Unit]:
        return [
            self.unit(p.name)
            for p in sorted((self.root / UNITS).iterdir())
            if (p / ".git").exists()
        ]

    def unit(self, name: str) -> Unit:
        return Unit(unit_path(self.root, name))

    def create_unit(
        self, name: str, *, stage: str, data: D, files: dict[str, str] | None = None
    ) -> Unit[D]:
        """Seed a typed unit without resetting an existing checkpoint."""
        unit = Unit.create(
            unit_path(self.root, name),
            stage=stage,
            data=data,
            stages=self.pipeline.stages,
            events=self.root / TRANSITIONS,
            files=files,
        )
        self.touch(name, "seeded")
        return unit

    def touch(self, unit: str, label: str) -> None:
        """Note that the running stage acted on another unit (created it, released it, made it
        ready): the transition records the link, so a dashboard can draw the edge between lanes."""
        if (links := _LINKS.get()) is not None:
            links.append(Link(unit=unit, label=label))

    # -- seats ------------------------------------------------------------------------------

    def seat(self, name: str) -> AgentConfig:
        cfg = self.config
        return resolve_agent(
            getattr(cfg, name),
            model=cfg.model,
            client=cfg.client,
            sampling=cfg.sampling,
        )

    # -- the loop ---------------------------------------------------------------------------

    async def run(self) -> RunResult:
        """Run until nothing is runnable, or drain. Call inside `async with Flow(...)`."""
        running: dict[str, tuple[asyncio.Task[None], ExitStack]] = {}

        def release(name: str) -> asyncio.Task[None]:
            task, lease = running.pop(name)
            lease.close()
            del self._active[name]
            return task

        async with self._serving():
            self.event(RunEvent(type="run_started", label=self.label))
            try:
                while True:
                    if not self.draining:
                        self._launch(running)
                    if not running:
                        break
                    done, _ = await asyncio.wait(
                        [task for task, _ in running.values()],
                        return_when=asyncio.FIRST_COMPLETED,
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
            reason="draining" if self.draining else "quiescent",
            units={unit.id: unit.state() for unit in self.units()},
        )
        self.event(
            RunEvent(type="run_finished", reason=result.reason, counts=result.counts)
        )
        return result

    def _launch(self, running: dict[str, tuple[asyncio.Task[None], ExitStack]]) -> None:
        for unit in self.units():
            if len(running) >= self.config.pools.get("units", 4) or self.draining:
                break
            if unit.id in running or unit.state().status != "ready":
                continue
            if self.pipeline.admit is not None and not self.pipeline.admit(unit, self):
                continue
            with ExitStack() as stack:
                before, execution = stack.enter_context(unit.executing())
                if before.status != "ready":
                    continue
                self._active[unit.id] = execution
                running[unit.id] = (
                    asyncio.create_task(self._stage(unit, before, execution)),
                    stack.pop_all(),
                )

    async def _stage(self, unit: Unit, before: UnitState, execution: Execution) -> None:
        name = execution.stage
        self.event(
            StageEvent(type="started", unit=unit.id, stage=name, execution=execution.id)
        )
        links: list[Link] = []
        token = _LINKS.set(links)
        try:
            transition = await self.pipeline.stages[name](
                Ctx(self, unit, before, execution)
            )
        except (Stopped, asyncio.CancelledError) as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            self.event(
                StageEvent(
                    type="cancelled" if cancelled else "stopped",
                    unit=unit.id,
                    stage=name,
                    execution=execution.id,
                )
            )
            if cancelled:
                raise
            return
        except Exception as exc:
            logger.exception("%s/%s failed", unit.id, name)
            transition = Transition(
                "held", f"{type(exc).__name__}: {exc}", status="held"
            )
        finally:
            _LINKS.reset(token)
        sha = unit.apply(transition, before=before)
        committed = unit.state()
        self.event(
            StageEvent(
                type="transition",
                unit=unit.id,
                stage=name,
                execution=execution.id,
                outcome=transition.outcome,
                to=committed.stage,
                status=committed.status,
                reason=committed.reason,
                report=transition.report,
                sha=sha,
                links=links,
            )
        )

    def event(self, event: EventRecord) -> None:
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
            not runtime_is_local(self.seat(n).runtime)
            for n in agent_config_fields(self.config)
        )
        async with make_interception(
            self.config.interception, requires_tunnel=remote
        ) as interception:
            self.interception = interception
            try:
                yield
            finally:
                self.interception = None


class Ctx(Generic[D, ConfigT]):
    """A stage's handle: its unit, the flow, and the calls."""

    def __init__(
        self,
        flow: Flow[ConfigT],
        unit: Unit[D],
        state: UnitState[D],
        execution: Execution,
    ) -> None:
        self.flow, self.unit, self.state = flow, unit, state
        self.execution = execution
        self.stage = execution.stage
        self.data = state.data.model_copy(deep=True)

    def notes(self) -> str:
        """Notes present at stage start; successful transitions acknowledge only these."""
        return "\n\n".join(self.state.notes)

    @property
    def config(self) -> ConfigT:
        return self.flow.config

    async def call(self, work: Work[T], *, key: str | None = None) -> T:
        """Return the work's value; a key opts into reuse of explicitly declared inputs."""
        result = await self.attempt(work, key=key)
        if not result.ok:
            raise CallFailed(result.error, result.trace_id)
        return result.value

    async def spread(
        self,
        works: Iterable[Work[T]],
        *,
        key: Callable[[int], str] | None = None,
    ) -> list[Result[T]]:
        """Every work at once; every item settled, the failures typed beside the values.
        What to make of a partial set is the stage's decision."""
        children = []
        try:
            for i, work in enumerate(works):
                children.append(
                    asyncio.create_task(self.attempt(work, key=key(i) if key else None))
                )
            # Stopped must not detach siblings: admitted work still records its result.
            results = await asyncio.gather(*children, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return cast(list[Result[T]], results)
        finally:
            for child in children:
                if not child.done():
                    child.cancel()
            await asyncio.gather(*children, return_exceptions=True)

    async def attempt(self, work: Work[T], *, key: str | None = None) -> Result[T]:
        """A value or failure. Successful keyed work is recorded; failures remain retryable."""
        if key is not None and work.inputs is None:
            raise ValueError(
                "keyed work requires explicit inputs; use {} for no dependencies"
            )
        flow = self.flow
        name = key if key is not None else work.kind
        cache = digest(key, work.inputs)[:24] if key is not None else None
        file = flow.root / "calls" / self.unit.id / f"{cache}.json" if cache else None
        call = uuid4().hex
        invocation = Invocation(
            unit=self.unit.id,
            stage=self.stage,
            execution=self.execution.id,
            call=call,
            key=key,
            kind=work.kind,
            cache=cache,
        )
        token = INVOCATION.set(invocation)
        try:
            if file is not None and file.exists():
                record = Record.model_validate_json(file.read_text())
                value = work.load(self, record)
                flow.event(
                    CallEvent(
                        invocation=invocation,
                        status="attached",
                        source_call=record.call,
                        source_execution=record.execution,
                        trace_id=record.trace_id,
                    )
                )
                return Success(value, attached=True)
            self.check_running()
            flow.event(CallEvent(invocation=invocation, status="started"))
            try:
                value = await work.execute(self)
            except (Stopped, asyncio.CancelledError):
                raise
            except Exception as exc:  # noqa: BLE001 - work failures become typed results
                result = Failure(
                    exc.error
                    if isinstance(exc, CallFailed)
                    else Error(type=type(exc).__name__, message=str(exc)),
                    exc.trace_id if isinstance(exc, CallFailed) else None,
                )
                flow.event(
                    CallEvent(
                        invocation=invocation,
                        status="failed",
                        error=result.error,
                        trace_id=result.trace_id,
                    )
                )
                return result
            # Publication failures stop the stage; repeating work is an operator decision.
            if file is not None:
                file.parent.mkdir(parents=True, exist_ok=True)
                record = Record(
                    key=name,
                    execution=self.execution.id,
                    call=call,
                    **work.dump(value),
                )
                tmp = file.with_suffix(".tmp")
                tmp.write_text(record.model_dump_json(indent=1))
                os.replace(tmp, file)
            trace_id = value.id if isinstance(value, Trace) else None
            flow.event(
                CallEvent(
                    invocation=invocation,
                    status="succeeded",
                    trace_id=trace_id,
                )
            )
            return Success(value)
        except Stopped:
            flow.event(CallEvent(invocation=invocation, status="stopped"))
            raise
        except asyncio.CancelledError:
            flow.event(CallEvent(invocation=invocation, status="cancelled"))
            raise
        except Exception as exc:
            flow.event(
                CallEvent(
                    invocation=invocation,
                    status="failed",
                    error=Error(type=type(exc).__name__, message=str(exc)),
                )
            )
            raise
        finally:
            INVOCATION.reset(token)

    @asynccontextmanager
    async def runtime(
        self, seat: str, task: Task | None = None
    ) -> AsyncIterator[Runtime]:
        """A box from the seat's runtime policy (resolved for `task` when given), alive for
        the block and always torn down; seats can borrow it with `runtime=box`."""
        config = self.flow.seat(seat).runtime
        if task is not None:
            config = resolve_runtime_config(config, task)
        async with self.flow.pools.hold(("runtimes",)):
            self.check_running()
            async with provision_runtime(config) as box:
                box.env = dict(task.runtime_env()) if task is not None else {}
                yield box

    def check_running(self) -> None:
        """Refuse new work after drain, including work that waited for a pool."""
        if self.flow.draining:
            raise Stopped(self.stage)


def drain_on_interrupt(flow: Flow[Any]) -> None:
    """SIGINT and SIGTERM: the first drains (calls in flight finish, units stay ready), the
    second cancels."""
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    assert task is not None

    def interrupt() -> None:
        if flow.draining:
            task.cancel()
        else:
            logger.warning("draining: calls in flight finish; Ctrl-C again cancels")
            flow.drain()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, interrupt)
