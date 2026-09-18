"""The loop: for every unit that is ready, run the stage its state names, commit the
transition, repeat until every unit is terminal, held or waiting, or the flow drains.

    pipeline = Pipeline(stages={"plan": plan, "author": author}, start="plan", data=TaskData)
    async with Flow(root, config, pipeline) as flow:
        await flow.run()

The campaign is a unit too, at `campaign/`; its stages run alone, so a campaign stage
that must see every task parked (a fix, a plan) needs no gate. `pipeline.admit` says which
ready tasks may start. A stage that raises holds its unit with the error as reason; a
drain lets in-flight calls finish and leaves their units ready to resume.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import signal
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import IO, Any, Generic, Self, TypeVar, cast

from pydantic_core import to_jsonable_python

from verifiers.v1.agent import Agent, make_agent
from verifiers.v1.configs.agent import (
    AgentConfig,
    declared_agent_configs,
    resolve_agent,
)
from verifiers.v1.flow.calls import CallFailed, Live, Record, Result, Work, now
from verifiers.v1.flow.config import FlowConfig
from verifiers.v1.flow.traces import Traces, trim_torn_tail
from verifiers.v1.flow.unit import D, Transition, Unit, UnitData, UnitState
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
CAMPAIGN = "campaign"
TASKS = "tasks"
DRAIN_FILE = "drain"
"""A file of this name in the root drains the flow, as Ctrl-C once does. Remove it to launch again."""
TRANSITIONS = "transitions.jsonl"
"""One line per stage start and per transition: what a dashboard follows."""
_LINKS: ContextVar[list[dict[str, str]] | None] = ContextVar("flow_links", default=None)
"""The other units the running stage touched, `{unit, label}`: the edges between lanes."""


def task_path(root: Path, name: str) -> Path:
    if (
        not name
        or name in (".", "..", CAMPAIGN)
        or Path(name).name != name
        or "\\" in name
    ):
        raise ValueError(f"unsafe or reserved task id: {name!r}")
    path = root / TASKS / name
    if not path.resolve().is_relative_to((root / TASKS).resolve()):
        raise ValueError(f"task path escapes root: {name!r}")
    return path


def succeeded(campaign: Unit, tasks: Iterable[Unit]) -> bool:
    """Waiting after planning is normal; a held campaign is not success."""
    return campaign.state().status in ("waiting", "terminal") and all(
        unit.state().status == "terminal" for unit in tasks
    )


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
class Pipeline(Generic[D]):
    stages: dict[str, Stage]
    start: str
    """The campaign's first stage."""
    data: type[D]
    admit: Callable[[Unit, Flow], bool] | None = None
    """Whether a ready task may start now; None admits every one."""
    config: type[FlowConfig] = FlowConfig
    campaign_data: type[UnitData] = UnitData


class Flow(Generic[D]):
    """One campaign root: `campaign/` and `tasks/<id>/` units, `calls/`, `traces.jsonl`,
    `live/`, `transitions.jsonl`, the config and the lock. `async with Flow(...)` owns it."""

    def __init__(self, root: Path, config: FlowConfig, pipeline: Pipeline[D]) -> None:
        self.root, self.config, self.pipeline = root, config, pipeline
        self.pools = Pools(config.pools)
        self.label = f"flow-{root.name}-{digest(str(root.resolve()))[:12]}"[:60]
        self.interception: Interception | None = None
        self._draining = asyncio.Event()
        self._dirty: set[str] = set()
        self._lock: IO[str] | None = None
        self._campaign_lock = asyncio.Lock()

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
            (self.root / TASKS).mkdir(exist_ok=True)
            trim_torn_tail(self.root / TRANSITIONS)
            self.traces, self.live = Traces(self.root), Live(self.root)
            self.campaign = Unit.create(
                self.root / CAMPAIGN,
                stage=self.pipeline.start,
                data=self.pipeline.campaign_data(),
                stages=self.pipeline.stages,
                events=self.root / TRANSITIONS,
            )
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

    def tasks(self) -> list[Unit[D]]:
        return [
            Unit(task_path(self.root, p.name), self.pipeline.data)
            for p in sorted((self.root / TASKS).iterdir())
            if (p / ".git").exists()
        ]

    def unit(self, name: str) -> Unit:
        return self.campaign if name == CAMPAIGN else Unit(task_path(self.root, name))

    def create_task(
        self,
        name: str,
        *,
        stage: str,
        data: D,
        files: dict[str, str] | None = None,
    ) -> Unit[D]:
        """Create an independently scheduled task with pipeline-typed durable data."""
        unit = Unit.create(
            task_path(self.root, name),
            stage=stage,
            data=self.pipeline.data.model_validate(data),
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

    async def run(self) -> dict[str, int]:
        """Run stages until nothing is runnable; the units by status. Call inside
        `async with Flow(...)`, after `sweep` on a resume."""
        running: dict[str, asyncio.Task[None]] = {}
        async with self._serving():
            try:
                while True:
                    launched = False if self.draining else await self._launch(running)
                    if not running:
                        if launched:
                            continue
                        break
                    done, _ = await asyncio.wait(
                        running.values(), return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        task.result()
                    running = {k: t for k, t in running.items() if t not in done}
            finally:
                # Settle stages before closing interception or releasing the launch lock.
                for task in running.values():
                    task.cancel()
                await asyncio.gather(*running.values(), return_exceptions=True)
        counts: dict[str, int] = {}
        for unit in self.tasks():
            status = unit.state().status
            counts[status] = counts.get(status, 0) + 1
        return counts

    async def _launch(self, running: dict[str, asyncio.Task[None]]) -> bool:
        """Start what may run: the campaign's stage, alone, when it is ready; else every
        admitted ready task up to the `units` pool. Whether anything was started."""
        if not self._clean(self.campaign):
            return False
        if self.campaign.state().status == "ready":
            if running:
                return False  # the campaign runs alone: let the tasks in flight finish
            await self._stage(self.campaign)
            return True
        started = 0
        free = self.config.pools.get("units", 4) - len(running)
        for unit in self.tasks():
            if free <= 0:
                break
            if unit.id in running or unit.state().status != "ready":
                continue
            if self.pipeline.admit is not None and not self.pipeline.admit(unit, self):
                continue
            if not self._clean(unit):
                continue
            running[unit.id] = asyncio.create_task(self._stage(unit))
            free -= 1
            started += 1
        return started > 0

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

    async def _stage(self, unit: Unit) -> None:
        if not self._clean(unit):
            return
        with unit.executing() as (before, execution):
            if before.status != "ready" or self.draining:
                return
            name = before.stage
            self.event("started", unit=unit.id, stage=name, execution=execution["id"])
            links: list[dict[str, str]] = []
            token = _LINKS.set(links)
            try:
                transition = await self.pipeline.stages[name](Ctx(self, unit, before))
            except Stopped:
                self.event(
                    "stopped", unit=unit.id, stage=name, execution=execution["id"]
                )
                return
            except asyncio.CancelledError:
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
                execution=execution["id"],
                outcome=transition.outcome,
                to=committed.stage,
                status=committed.status,
                reason=committed.reason,
                report=transition.report,
                sha=sha,
                links=links,
            )

    def event(self, kind: str, **fields: Any) -> None:
        line = json.dumps({"type": kind, "at": now(), **fields})
        with (self.root / TRANSITIONS).open("a") as file:
            file.write(line + "\n")
        logger.info("%s", line)

    @asynccontextmanager
    async def commit_campaign(self) -> AsyncIterator[Unit]:
        """The campaign unit, to write from a task stage: one writer at a time."""
        async with self._campaign_lock:
            yield self.campaign

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

    def __init__(self, flow: Flow, unit: Unit[D], state: UnitState[D]) -> None:
        self.flow, self.unit, self.state = flow, unit, state
        self.stage = state.stage
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

    async def attempt(
        self, work: Work[T], *, key: str | None, retries: int, timeout: float | None
    ) -> Result[T]:
        flow = self.flow
        name = key or work.kind
        file = None
        if key:
            calls = flow.root / "calls" / self.unit.id
            file = calls / (digest(key, work.content(self))[:24] + ".json")
        if file is not None and file.exists():
            record = Record.model_validate_json(file.read_text())
            try:
                return Result(
                    True,
                    work.load(self, record),
                    trace_id=record.trace_id,
                    attached=True,
                )
            except LookupError as exc:
                logger.warning("%s/%s: %s; running again", self.unit.id, name, exc)
        if flow.draining:
            raise Stopped(name)
        started, attempt = now(), 0
        while True:
            if flow.draining:
                raise Stopped(name)
            try:
                async with asyncio.timeout(timeout):
                    value = await work.execute(self, name)
                    fields = work.dump(self, value)
            except Stopped:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "%s/%s: attempt %d failed: %s",
                    self.unit.id,
                    name,
                    attempt + 1,
                    error,
                )
                if attempt >= retries:
                    failed = exc if isinstance(exc, CallFailed) else None
                    return Result(
                        False,
                        error=error,
                        type=failed.type if failed else type(exc).__name__,
                        status_code=failed.status_code if failed else None,
                        trace_id=failed.trace_id if failed else None,
                    )
                if flow.draining:
                    raise Stopped(name) from exc
                await asyncio.sleep(backoff(attempt))
                attempt += 1
                continue
            if file is not None:
                file.parent.mkdir(parents=True, exist_ok=True)
                record = Record(
                    key=name,
                    unit=self.unit.id,
                    stage=self.stage,
                    kind=work.kind,
                    started_at=started,
                    finished_at=now(),
                    **fields,
                )
                tmp = file.with_suffix(".tmp")
                tmp.write_text(record.model_dump_json(indent=1))
                os.replace(tmp, file)
            return Result(True, value, trace_id=fields.get("trace_id"))

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
