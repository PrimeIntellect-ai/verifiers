"""The loop: for every unit that is ready, run the stage its state names, commit the
transition, repeat until every unit is terminal, held or waiting, or the flow drains.

    pipeline = Pipeline(stages={"plan": plan, "author": author, ...}, start="plan")
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
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import IO, Any, Self, TypeVar

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
from verifiers.v1.flow.unit import Transition, Unit
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
    start: str
    """The campaign's first stage."""
    admit: Callable[[Unit, Flow], bool] | None = None
    """Whether a ready task may start now; None admits every one."""
    config: type[FlowConfig] = FlowConfig


class Flow:
    """One campaign root: `campaign/` and `tasks/<id>/` units, `calls/`, `traces.jsonl`,
    `live/`, `transitions.jsonl`, the config and the lock. `async with Flow(...)` owns it."""

    def __init__(self, root: Path, config: FlowConfig, pipeline: Pipeline) -> None:
        self.root, self.config, self.pipeline = root, config, pipeline
        self.pools = Pools(config.pools)
        self.label = f"flow-{root.name}-{digest(str(root.resolve()))[:12]}"[:60]
        self.interception: Interception | None = None
        self._draining = asyncio.Event()
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
        (self.root / "flow.json").write_text(self.config.model_dump_json(indent=1))
        (self.root / "calls").mkdir(exist_ok=True)
        (self.root / TASKS).mkdir(exist_ok=True)
        trim_torn_tail(self.root / TRANSITIONS)
        self.traces, self.live = Traces(self.root), Live(self.root)
        self.campaign = Unit.create(
            self.root / CAMPAIGN, {"stage": self.pipeline.start}
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._lock is not None
        self._lock.close()
        self._lock = None

    # -- units -----------------------------------------------------------------------------

    def tasks(self) -> list[Unit]:
        return [
            Unit(p)
            for p in sorted((self.root / TASKS).iterdir())
            if (p / ".git").exists()
        ]

    def unit(self, name: str) -> Unit:
        return self.campaign if name == CAMPAIGN else Unit(self.root / TASKS / name)

    def create_task(
        self, name: str, state: dict[str, Any], files: dict[str, str] | None = None
    ) -> Unit:
        """A new task unit, ready at `state["stage"]`."""
        return Unit.create(self.root / TASKS / name, state, files)

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
            while True:
                launched = False if self.draining else await self._launch(running)
                if not running:
                    if launched:
                        continue  # a campaign stage ran alone: look again
                    break  # nothing runnable: done, or every unit is held or waiting
                done, _ = await asyncio.wait(
                    running.values(), return_when=asyncio.FIRST_COMPLETED
                )
                running = {k: t for k, t in running.items() if t not in done}
                for task in done:
                    task.result()  # a stage never raises; anything else is a bug worth surfacing
        counts: dict[str, int] = {}
        for unit in self.tasks():
            status = unit.state().get("status", "?")
            counts[status] = counts.get(status, 0) + 1
        return counts

    async def _launch(self, running: dict[str, asyncio.Task[None]]) -> bool:
        """Start what may run: the campaign's stage, alone, when it is ready; else every
        admitted ready task up to the `units` pool. Whether anything was started."""
        if self.campaign.state().get("status") == "ready":
            if running:
                return False  # the campaign runs alone: let the tasks in flight finish
            await self._stage(self.campaign)
            return True
        started = 0
        free = self.config.pools.get("units", 4) - len(running)
        for unit in self.tasks():
            if free <= 0:
                break
            if unit.id in running or unit.state().get("status") != "ready":
                continue
            if self.pipeline.admit is not None and not self.pipeline.admit(unit, self):
                continue
            running[unit.id] = asyncio.create_task(self._stage(unit))
            free -= 1
            started += 1
        return started > 0

    async def _stage(self, unit: Unit) -> None:
        state = unit.state()
        name = state["stage"]
        stage = self.pipeline.stages[name]
        self.event("started", unit=unit.id, stage=name)
        try:
            transition = await stage(Ctx(self, unit, name))
        except Stopped:
            self.event("stopped", unit=unit.id, stage=name)
            return  # the unit stays ready; a relaunch runs the stage again
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("%s/%s failed", unit.id, name)
            transition = Transition.hold(f"{type(exc).__name__}: {exc}")
        sha = unit.apply(transition)
        self.event(
            "transition",
            unit=unit.id,
            stage=name,
            outcome=transition.outcome,
            to=transition.stage or name,
            status=transition.status,
            reason=transition.summary,
            sha=sha,
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
        """Kill what an earlier launch left behind, by the flow's label; the count."""
        from verifiers.v1.runtimes.docker import sweep_containers
        from verifiers.v1.runtimes.prime import sweep_sandboxes
        from verifiers.v1.runtimes.subprocess import sweep_subprocesses

        return (
            sweep_subprocesses(self.label)
            + await sweep_containers(self.label)
            + await sweep_sandboxes([self.label])
        )

    @asynccontextmanager
    async def _serving(self) -> AsyncIterator[None]:
        set_base_sandbox_labels([self.label])
        os.environ[RUN_LABEL_VAR] = self.label
        remote = any(
            not runtime_is_local(self.seat(n).runtime)
            for n in declared_agent_configs(self.config)
        )
        async with make_interception(
            self.config.interception, requires_tunnel=remote
        ) as interception:
            self.interception = interception
            try:
                yield
            finally:
                self.interception = None


class Ctx:
    """A stage's handle: its unit, the flow, and the calls."""

    def __init__(self, flow: Flow, unit: Unit, stage: str) -> None:
        self.flow, self.unit, self.stage = flow, unit, stage

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
        return result.value  # type: ignore[return-value]

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
        return list(
            await asyncio.gather(
                *(
                    self.attempt(
                        work,
                        key=key(i) if key else None,
                        retries=retries,
                        timeout=timeout,
                    )
                    for i, work in enumerate(works)
                )
            )
        )

    async def attempt(
        self, work: Work[T], *, key: str | None, retries: int, timeout: float | None
    ) -> Result[T]:
        flow = self.flow
        name = key or work.kind
        file = (
            flow.root
            / "calls"
            / self.unit.id
            / (digest(key, work.content(self))[:24] + ".json")
            if key
            else None
        )
        if file is not None and file.exists():
            record = Record.model_validate_json(file.read_text())
            try:
                return Result(True, work.load(self, record), trace_id=record.trace_id)
            except LookupError as exc:
                logger.warning("%s/%s: %s; running again", self.unit.id, name, exc)
        if flow.draining:
            raise Stopped(name)
        started, attempt = now(), 0
        while True:
            try:
                async with asyncio.timeout(timeout):
                    value = await work.execute(self, name)
                    fields = work.dump(self, value)
            except Stopped:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — the failure is the result
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
        async with (
            self.flow.pools.hold(("runtimes",)),
            provision_runtime(config, env=env) as box,
        ):
            yield box


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
