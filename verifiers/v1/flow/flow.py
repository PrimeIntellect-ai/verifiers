"""Run admitted, ready units until quiescence or drain. Pipelines own scheduling policy."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import signal
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any, Generic, Literal, Self, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from verifiers.v1.agent import Agent, make_agent
from verifiers.v1.configs.agent import (
    AgentConfig,
    declared_agent_configs,
    resolve_agent,
)
from verifiers.v1.flow.calls import (
    INVOCATION,
    CallFailed,
    Invocation,
    Live,
    Record,
    Result,
    Work,
    now,
)
from verifiers.v1.flow.config import FlowConfig
from verifiers.v1.flow.traces import Traces, trim_torn_tail
from verifiers.v1.flow.unit import D, Execution, Transition, Unit, UnitState
from verifiers.v1.interception import Interception, make_interception
from verifiers.v1.runtimes import (
    Runtime,
    provision_runtime,
    runtime_is_local,
    set_base_sandbox_labels,
)
from verifiers.v1.runtimes.subprocess import RUN_LABEL_VAR
from verifiers.v1.task import Task
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.retries import backoff

logger = logging.getLogger("verifiers.flow")

T = TypeVar("T")
UNITS = "units"
DRAIN_FILE = "drain"
"""A file of this name in the root drains the flow, as Ctrl-C once does. Remove it to launch again."""
TRANSITIONS = "transitions.jsonl"
"""One line per stage start and per transition: what a dashboard follows."""
_LINKS: ContextVar[list[dict[str, str]] | None] = ContextVar("flow_links", default=None)
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

    reason: Literal["quiescent", "draining"]
    units: dict[str, UnitState[Any]]

    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for state in self.units.values():
            counts[state.status] = counts.get(state.status, 0) + 1
        return counts


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


Stage = Callable[["Ctx"], Awaitable[Transition]]
"""A stage: `async def stage(ctx) -> Transition`."""


@dataclass(frozen=True)
class Pipeline:
    stages: dict[str, Stage]
    admit: Callable[[Unit, Flow], bool] | None = None
    """Whether a ready, inactive unit may start. Earlier admissions are already active."""
    initialize: Callable[[Flow], None] | None = None
    """Entrypoints call this to seed initial units; create_unit preserves existing work."""
    config: type[FlowConfig] = FlowConfig


class Flow:
    """Owns a run root: units, call results, traces, events, configuration and launch lock."""

    def __init__(self, root: Path, config: FlowConfig, pipeline: Pipeline) -> None:
        self.root, self.config, self.pipeline = root, config, pipeline
        self.pools = Pools(config.pools)
        self.label = f"flow-{root.name}-{digest(str(root.resolve()))[:12]}"[:60]
        self.interception: Interception | None = None
        self._draining = asyncio.Event()
        self._dirty: set[str] = set()
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
        try:
            (self.root / "flow.json").write_text(self.config.model_dump_json(indent=1))
            (self.root / "calls").mkdir(exist_ok=True)
            (self.root / UNITS).mkdir(exist_ok=True)
            trim_torn_tail(self.root / TRANSITIONS)
            self.traces, self.live = Traces(self.root), Live(self.root)
            return self
        except BaseException:
            lock.close()
            self._lock = None
            raise

    async def __aexit__(self, *exc: object) -> None:
        assert self._lock is not None
        self._lock.close()
        self._lock = None

    # -- units -----------------------------------------------------------------------------

    def units(self) -> list[Unit]:
        return [
            Unit(unit_path(self.root, p.name))
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
        self.touch(name, "created")
        return unit

    def touch(self, unit: str, label: str) -> None:
        """Note that the running stage acted on another unit (created it, released it, made it
        ready): the transition records the link, so a dashboard can draw the edge between lanes."""
        if (links := _LINKS.get()) is not None:
            links.append({"unit": unit, "label": label})

    # -- seats ------------------------------------------------------------------------------

    def seat(self, name: str) -> AgentConfig:
        cfg = self.config
        return resolve_agent(
            getattr(cfg, name),
            model=cfg.model,
            client=cfg.client,
            sampling=cfg.sampling,
        )

    def agent(self, name: str) -> Agent:
        return make_agent(self.seat(name), interception=self.interception)

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
            self.event("run_started")
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
        self.event("run_finished", reason=result.reason, counts=result.counts)
        return result

    def _launch(self, running: dict[str, tuple[asyncio.Task[None], ExitStack]]) -> None:
        for unit in self.units():
            if len(running) >= self.config.pools.get("units", 4) or self.draining:
                break
            if (
                unit.id in running
                or not self._clean(unit)
                or unit.state().status != "ready"
            ):
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

    def _clean(self, unit: Unit) -> bool:
        """Whether the unit's repository is clean enough to schedule. A dirty one (a crash between
        writing files and committing) is left alone with one `dirty` line for the operator to
        repair, so it parks like a hold instead of taking the run down."""
        try:
            unit.check_clean()
        except RuntimeError as exc:
            if unit.id not in self._dirty:
                self._dirty.add(unit.id)
                self.event("dirty", unit=unit.id, reason=str(exc))
            return False
        self._dirty.discard(unit.id)
        return True

    async def _stage(self, unit: Unit, before: UnitState, execution: Execution) -> None:
        name = execution.stage
        self.event("started", unit=unit.id, stage=name, execution=execution.id)
        links: list[dict[str, str]] = []
        token = _LINKS.set(links)
        try:
            transition = await self.pipeline.stages[name](
                Ctx(self, unit, before, execution)
            )
        except Stopped:
            self.event("stopped", unit=unit.id, stage=name, execution=execution.id)
            return
        except asyncio.CancelledError:
            self.event("cancelled", unit=unit.id, stage=name, execution=execution.id)
            raise
        except Exception as exc:
            logger.exception("%s/%s failed", unit.id, name)
            transition = Transition.hold(f"{type(exc).__name__}: {exc}")
        finally:
            _LINKS.reset(token)
        sha = unit.apply(transition, before=before)
        committed = unit.state()
        self.event(
            "transition",
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

    def event(self, event_type: str, **fields: Any) -> None:
        line = json.dumps({"type": event_type, "at": now(), **fields})
        with (self.root / TRANSITIONS).open("a") as file:
            file.write(line + "\n")
        logger.info("%s", line)

    # -- drain, sweep, serving ----------------------------------------------------------------

    def drain(self) -> None:
        if not self._draining.is_set():
            self.event("drain")
            self._draining.set()

    @property
    def draining(self) -> bool:
        if not self._draining.is_set() and (self.root / DRAIN_FILE).exists():
            self.drain()
        return self._draining.is_set()

    async def sweep(self) -> int:
        """Kill what an earlier launch left behind, by the flow's label; the count. Every
        runtime module registers its own sweeper (`SWEEPERS`) when the package imports."""
        import verifiers.v1.runtimes  # noqa: F401 - registers the sweepers
        from verifiers.v1.runtimes.base import SWEEPERS
        from verifiers.v1.runtimes.subprocess import sweep_subprocesses

        swept = sweep_subprocesses(self.label)
        for sweeper in SWEEPERS:
            swept += await sweeper(self.label)
        return swept

    @asynccontextmanager
    async def _serving(self) -> AsyncIterator[None]:
        from verifiers.v1.runtimes import prime

        labels, run_label = prime.BASE_LABELS, os.environ.get(RUN_LABEL_VAR)
        set_base_sandbox_labels([self.label])
        os.environ[RUN_LABEL_VAR] = self.label
        try:
            remote = any(
                not runtime_is_local(self.seat(n).runtime)
                for n in declared_agent_configs(self.config)
            )
            async with make_interception(
                self.config.interception, requires_tunnel=remote
            ) as interception:
                self.interception = interception
                yield
        finally:
            self.interception = None
            set_base_sandbox_labels(labels)
            if run_label is None:
                os.environ.pop(RUN_LABEL_VAR, None)
            else:
                os.environ[RUN_LABEL_VAR] = run_label


class Ctx(Generic[D]):
    """A stage's handle: its unit, the flow, and the calls."""

    def __init__(
        self, flow: Flow, unit: Unit[D], state: UnitState[D], execution: Execution
    ) -> None:
        self.flow, self.unit, self.state = flow, unit, state
        self.execution = execution
        self.stage = execution.stage
        self.data = state.data.model_copy(deep=True)

    def notes(self) -> str:
        """Notes present at stage start; successful transitions acknowledge only these."""
        return "\n\n".join(note.text for note in self.state.notes)

    def updated(self, **fields: Any) -> D:
        """A validated copy for a transition; core scheduling fields cannot enter data."""
        return self.unit.data_type.model_validate(
            {**self.data.model_dump(mode="json"), **fields}
        )

    @property
    def config(self) -> FlowConfig:
        return self.flow.config

    def seat(self, name: str) -> AgentConfig:
        return self.flow.seat(name)

    async def call(
        self,
        work: Work[T],
        *,
        key: str | None = None,
        retries: int = 0,
        timeout: float | None = None,
    ) -> T:
        """Run `work`; the value. With a `key`, durable: the result is recorded under the key
        and the work's content, and found again by a rerun of the stage. `retries` rerun the
        work after any failure (an agent has its own policy: leave this at 0 for seats)."""
        result = await self.attempt(work, key=key, retries=retries, timeout=timeout)
        if not result.ok:
            raise CallFailed(
                f"{key or work.kind}: {result.error}",
                result.type or "",
                result.status_code,
                result.trace_id,
            )
        return cast(T, result.value)

    async def spread(
        self,
        works: Iterable[Work[T]],
        *,
        key: Callable[[int], str] | None = None,
        retries: int = 0,
        timeout: float | None = None,
    ) -> list[Result[T]]:
        """Every work at once; every item settled, the failures typed beside the values.
        What to make of a partial set is the stage's decision."""
        children = []
        try:
            for i, work in enumerate(works):
                children.append(
                    asyncio.create_task(
                        self.attempt(
                            work,
                            key=key(i) if key else None,
                            retries=retries,
                            timeout=timeout,
                        )
                    )
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

    def event(self, kind: str, status: str, **fields: Any) -> None:
        """Call and native rollout evidence share one invocation and stage execution."""
        invocation = asdict(INVOCATION.get())
        invocation["call"] = invocation.pop("id")
        self.flow.event(
            kind,
            unit=self.unit.id,
            stage=self.stage,
            execution=self.execution.id,
            status=status,
            **invocation,
            **fields,
        )

    async def attempt(
        self, work: Work[T], *, key: str | None, retries: int, timeout: float | None
    ) -> Result[T]:
        flow = self.flow
        name, call = key or work.kind, uuid4().hex
        cache = digest(key, work.content(self))[:24] if key is not None else None
        file = flow.root / "calls" / self.unit.id / f"{cache}.json" if cache else None
        parent = INVOCATION.get(None)
        token = INVOCATION.set(
            Invocation(
                call, key, work.kind, cache, parent=parent.id if parent else None
            )
        )
        try:
            if file is not None and file.exists():
                record = Record.model_validate_json(file.read_text())
                try:
                    value = work.load(self, record)
                except Exception as exc:
                    self.event("call", "failed", error=f"{type(exc).__name__}: {exc}")
                    raise
                self.event(
                    "call",
                    "attached",
                    source_call=record.call,
                    source_execution=record.execution,
                    trace_id=record.trace_id,
                )
                return Result(True, value, trace_id=record.trace_id, attached=True)
            for attempt in range(1, retries + 2):
                INVOCATION.set(
                    Invocation(
                        call,
                        key,
                        work.kind,
                        cache,
                        attempt,
                        parent.id if parent else None,
                    )
                )
                if flow.draining:
                    self.event("call", "stopped")
                    raise Stopped(name)
                started = now()
                self.event("call", "started")
                try:
                    async with asyncio.timeout(timeout):
                        value = await work.execute(self, name)
                        fields = work.dump(self, value)
                except Stopped:
                    self.event("call", "stopped")
                    raise
                except asyncio.CancelledError:
                    self.event("call", "cancelled")
                    raise
                except Exception as exc:  # noqa: BLE001 - a failed work item becomes a typed Result
                    error = f"{type(exc).__name__}: {exc}"
                    failed = exc if isinstance(exc, CallFailed) else None
                    result: Result[T] = Result(
                        False,
                        error=error,
                        type=failed.type if failed else type(exc).__name__,
                        status_code=failed.status_code if failed else None,
                        trace_id=failed.trace_id if failed else None,
                    )
                    self.event(
                        "call",
                        "failed",
                        error=error,
                        error_type=result.type,
                        status_code=result.status_code,
                        trace_id=result.trace_id,
                    )
                    logger.warning(
                        "%s/%s: attempt %d failed: %s",
                        self.unit.id,
                        name,
                        attempt,
                        error,
                    )
                    if attempt > retries:
                        return result
                    try:
                        await asyncio.sleep(backoff(attempt - 1))
                    except asyncio.CancelledError:
                        self.event("call", "cancelled")
                        raise
                    continue
                try:
                    if file is not None:
                        file.parent.mkdir(parents=True, exist_ok=True)
                        record = Record(
                            key=name,
                            unit=self.unit.id,
                            stage=self.stage,
                            kind=work.kind,
                            execution=self.execution.id,
                            call=call,
                            attempt=attempt,
                            started_at=started,
                            finished_at=now(),
                            **fields,
                        )
                        tmp = file.with_suffix(".tmp")
                        tmp.write_text(record.model_dump_json(indent=1))
                        os.replace(tmp, file)
                except Exception as exc:
                    self.event(
                        "call",
                        "failed",
                        error=f"recording result: {type(exc).__name__}: {exc}",
                    )
                    raise
                self.event("call", "succeeded", trace_id=fields.get("trace_id"))
                return Result(True, value, trace_id=fields.get("trace_id"))
            raise ValueError("retries must be nonnegative")
        finally:
            INVOCATION.reset(token)

    @asynccontextmanager
    async def runtime(
        self, seat: str, task: Task | None = None
    ) -> AsyncIterator[Runtime]:
        """A box from the seat's runtime policy (resolved for `task` when given), alive for
        the block and always torn down; seats and commands run in it with `runtime=box`."""
        config = self.seat(seat).runtime
        if task is not None:
            config = resolve_runtime_config(config, task, set())
        env = task.runtime_env() if task is not None else None
        async with self.flow.pools.hold(("runtimes",)):
            self.check_running()
            async with provision_runtime(config, env=env) as box:
                yield box

    def check_running(self) -> None:
        """Refuse new work after drain, including work that waited for a pool."""
        if self.flow.draining:
            raise Stopped(self.stage)


def drain_on_interrupt(flow: Flow) -> None:
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
