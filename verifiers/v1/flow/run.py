"""A run: rows in, results out, every step durable.

    async def flywheel(ctx: Ctx, row: dict) -> dict:
        screen = await ctx.step("prescreen", agent("reviewer", review_request(row)))
        ...

A flow is a function of a `Ctx` and a row. Each `ctx.step` is memoized in the
ledger by its place in the function and the content of its work, so re-running the
function against the same run directory attaches to what finished and runs only
what did not. Between steps the code must depend only on the row, the config, and
earlier step values, so a resume walks the same path. Concurrent branches are
`asyncio.gather` or `create_task`; a step's place is fixed when it is called, and a
`scope` is per task.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
import typing
from collections import deque
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
)
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from verifiers.v1.agent import make_agent
from verifiers.v1.configs.agent import (
    AgentConfig,
    declared_agent_configs,
    resolve_agent,
)
from verifiers.v1.flow.config import RUNTIMES, FlowConfig
from verifiers.v1.flow.ledger import (
    SHORT,
    STEP_KEY,
    EventKind,
    Ledger,
    StepRecord,
    Terminal,
    digest,
    now,
    row_key,
)
from verifiers.v1.flow.pools import Pools
from verifiers.v1.flow.work import AgentWork, CommandWork, FnWork, Work
from verifiers.v1.interception import Interception, make_interception
from verifiers.v1.runtimes import (
    ProgramResult,
    Runtime,
    provision_runtime,
    runtime_is_local,
    set_base_sandbox_labels,
)
from verifiers.v1.runtimes.subprocess import RUN_LABEL_VAR
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import Usage
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.retries import backoff

logger = logging.getLogger("verifiers.flow")

_MISSING = object()  # a record whose value cannot be rebuilt: run the step again
LABEL_MAX = 60  # sandbox label length cap
_SCOPE: ContextVar[tuple[str, ...]] = ContextVar("flow_scope", default=())

T = TypeVar("T")
RowState = Literal["running", "ok", "failed", "stopped"]


class StepFailed(Exception):
    """A step used up its retries, or failed in a way no retry mends."""

    def __init__(self, path: str, error: str) -> None:
        super().__init__(f"{path}: {error}")
        self.path, self.error = path, error


class Stopped(Exception):
    """The run is draining: no step starts; the row returns stopped, to resume later."""


class Oversized(ValueError):
    """A step value over `payload_cap`: no retry shrinks it, so the step fails at once."""


class _RolloutFailed(Exception):
    def __init__(self, trace: Trace) -> None:
        last = trace.last_error
        super().__init__(f"{last.type}: {last.message}" if last else "rollout failed")


@dataclass
class RowResult:
    row: str
    state: RowState
    value: Any = None
    error: str | None = None


@dataclass
class RunStatus:
    rows: dict[str, RowState]
    steps: list[str]
    """`<row>/<path>` in flight."""
    draining: bool
    usage: Usage | None
    """Provider usage summed over every agent step so far."""


class Run:
    """The ledger, the pools, the interception, and the row loop for one run directory.

    The run's identity is its directory name, never its config: operational knobs and
    a pipeline's own policy fields re-key nothing on a resume — an agent step keys on
    its resolved seat, all else on its work — and a launch against records keyed under
    a different identity refuses unless `rekey=True`."""

    def __init__(
        self, run_dir: Path, config: FlowConfig, *, rekey: bool = False
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.config = config
        self.ledger = Ledger(run_dir)
        self.pools = Pools(config.pools)
        self.identity = digest(run_dir.name)[:SHORT]
        self.label = f"flow-{run_dir.name}-{self.identity}"[:LABEL_MAX]
        self._require_identity(rekey)
        self.rows: dict[str, RowState] = {}
        self.steps: set[str] = set()  # `<row>/<path>` in flight
        self.usage: Usage | None = None
        self._draining = asyncio.Event()
        self._inference: Interception | None = None
        (run_dir / "config.json").write_text(config.model_dump_json(indent=1))

    # -- rows -----------------------------------------------------------------------

    async def stream(
        self, flow: Callable[..., Any], rows: Iterable[Any] | AsyncIterable[Any]
    ) -> AsyncIterator[RowResult]:
        """Run `flow(ctx, row)` for every row, at most `max_concurrent_rows` at once,
        yielding each result as its row finishes. A row that raises is a failed row."""
        source = self._note_source(flow)
        self.ledger.event(
            "run", identity=self.identity, source=source, label=self.label
        )
        results: asyncio.Queue[RowResult] = asyncio.Queue()
        gate = asyncio.Semaphore(self.config.max_concurrent_rows)
        tasks: list[asyncio.Task] = []

        async def one(row: Any) -> None:
            async with gate:
                await results.put(await self._row(flow, row))

        async def feed() -> None:
            async for row in _aiter(rows):
                if self._draining.is_set():
                    break
                tasks.append(asyncio.create_task(one(row)))

        async with self._serving():
            feeder = asyncio.create_task(feed())
            try:
                yielded = 0
                while not feeder.done() or yielded < len(tasks):
                    if feeder.done():
                        feeder.result()  # a failed producer raises here
                        result = await results.get()
                    else:
                        getter = asyncio.ensure_future(results.get())
                        await asyncio.wait(
                            {getter, feeder}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if not getter.done():
                            getter.cancel()
                            await asyncio.gather(getter, return_exceptions=True)
                            continue
                        result = getter.result()
                    yielded += 1
                    yield result
            finally:
                feeder.cancel()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(feeder, *tasks, return_exceptions=True)

    async def run(
        self, flow: Callable[..., Any], rows: Iterable[Any] | AsyncIterable[Any]
    ) -> list[RowResult]:
        return [result async for result in self.stream(flow, rows)]

    async def sweep(self) -> int:
        """Kill what an earlier launch of this run left behind -- Prime sandboxes by
        label, host subprocesses by the same label in their environment; the count.
        Call before `stream` at a resume."""
        from verifiers.v1.runtimes.prime import sweep_sandboxes
        from verifiers.v1.runtimes.subprocess import sweep_subprocesses

        return sweep_subprocesses(self.label) + await sweep_sandboxes([self.label])

    def drain(self) -> None:
        """Stop admitting rows and steps; in-flight steps finish and record."""
        if not self._draining.is_set():
            self.ledger.event("drain")
            self._draining.set()

    def status(self) -> RunStatus:
        return RunStatus(
            rows=dict(self.rows),
            steps=sorted(self.steps),
            draining=self._draining.is_set(),
            usage=self.usage,
        )

    def seat(self, name: str) -> AgentConfig:
        """The seat with the run's defaults filled in: the identity its steps key on."""
        cfg = self.config
        return resolve_agent(
            getattr(cfg, name),
            model=cfg.model,
            client=cfg.client,
            sampling=cfg.sampling,
        )

    async def _row(self, flow: Callable[..., Any], row: Any) -> RowResult:
        try:
            key = row_key(row)
        except Exception as exc:  # noqa: BLE001 — the row itself is the problem being reported
            return RowResult(
                row=repr(row)[:80], state="failed", error=f"row cannot be keyed: {exc}"
            )
        self.rows[key] = "running"
        self.ledger.event("row_started", row=key)
        try:
            value = await flow(Ctx(self, key, row), row)
            self.rows[key] = "ok"
            self.ledger.event("row_finished", row=key, state="ok")
            return RowResult(row=key, state="ok", value=value)
        except Stopped:
            self.rows[key] = "stopped"
            self.ledger.event("row_finished", row=key, state="stopped")
            return RowResult(row=key, state="stopped")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a failed row is data, never a failed run
            self.rows[key] = "failed"
            error = f"{type(exc).__name__}: {exc}"
            self.ledger.event("row_finished", row=key, state="failed", error=error)
            return RowResult(row=key, state="failed", error=error)

    @asynccontextmanager
    async def _serving(self) -> AsyncIterator[None]:
        set_base_sandbox_labels([self.label])
        os.environ[RUN_LABEL_VAR] = (
            self.label
        )  # every host subprocess inherits it: what `sweep` finds
        remote = any(
            not runtime_is_local(self.seat(name).runtime)
            for name in declared_agent_configs(self.config)
        )
        interception = make_interception(
            self.config.interception, requires_tunnel=remote
        )
        async with interception:
            self._inference = interception
            try:
                yield
            finally:
                self._inference = None

    def _require_identity(self, rekey: bool) -> None:
        """Refuse a resume against records keyed under another identity: nothing would
        attach, and every step would re-run over the finished ones. `rekey=True`
        accepts the new identity, loudly."""
        file = self.run_dir / "run.json"
        previous = json.loads(file.read_text()) if file.exists() else {}
        recorded = previous.get("identity")
        if recorded in (None, self.identity):
            return
        if not rekey:
            raise ValueError(
                f"{self.run_dir}: run.json records identity {recorded} but this "
                f"launch computes {self.identity} — resuming would re-key every step; "
                "pass Run(..., rekey=True) to accept that"
            )
        logger.warning("%s: rekeying %s -> %s", self.run_dir, recorded, self.identity)

    def _note_source(self, flow: Callable[..., Any]) -> str | None:
        """Record the run's identity and the flow's source hash; a resume under changed
        code attaches to the steps whose inputs still match, so say so once. The hash,
        None if unavailable (an unhashable flow keeps what was recorded)."""
        file = self.run_dir / "run.json"
        previous = json.loads(file.read_text()) if file.exists() else {}
        try:
            source = digest(inspect.getsource(flow))[:SHORT]
        except (OSError, TypeError):
            source = previous.get("source")
        if source is not None and previous.get("source") not in (None, source):
            logger.warning(
                "%s: the flow's source changed since this run was written", self.run_dir
            )
        file.write_text(
            json.dumps(
                {"identity": self.identity, "source": source, "label": self.label},
                indent=1,
            )
        )
        return source


class Ctx:
    """One row's handle on the run: steps, spreads, runtimes, scopes."""

    def __init__(self, run: Run, key: str, row: Any) -> None:
        self.run, self.key, self.row = run, key, row
        self._counts: dict[tuple[str, ...], dict[str, int]] = {}

    @property
    def config(self) -> FlowConfig:
        return self.run.config

    def seat(self, name: str) -> AgentConfig:
        return self.run.seat(name)

    @contextmanager
    def scope(self, label: str) -> Iterator[None]:
        """A namespace for the steps inside: a loop iteration, a phase, a branch.
        Concurrent branches take one each: two branches stepping under the same scope
        race for occurrence numbers and neither attaches on resume."""
        token = _SCOPE.set((*_SCOPE.get(), label))
        try:
            yield
        finally:
            _SCOPE.reset(token)

    def _path(self, name: str) -> str:
        scope = _SCOPE.get()
        counts = self._counts.setdefault(scope, {})
        n = counts.get(name, 0)
        counts[name] = n + 1
        return "/".join([*scope, f"{name}#{n}"])

    def _event(self, kind: EventKind, **fields: Any) -> None:
        """One event for this row: the row key always rides along."""
        self.run.ledger.event(kind, row=self.key, **fields)

    # -- steps ----------------------------------------------------------------------

    def step(
        self,
        name: str,
        work: Work[T],
        *,
        retries: int = 0,
        timeout: float | None = None,
    ) -> Awaitable[T]:
        """Run `work` once, durably. The value is the trace, the program result, or
        the function's return; on resume it comes from the ledger. `retries` re-run the
        whole work after a failure, with the shared backoff, on top of an agent's own
        `AgentConfig.retries` (which names the error types worth a rerun)."""
        return self._step_value(self._path(name), work, retries, timeout)

    async def _step_value(
        self, path: str, work: Work[T], retries: int, timeout: float | None
    ) -> T:
        value, _ = await self._step(path, work, None, retries, timeout)
        return value

    def spread(
        self,
        name: str,
        works: Iterable[Work[T]],
        *,
        at_least: int | None = None,
        max_active: int | None = None,
        within: float | None = None,
        retries: int = 0,
        timeout: float | None = None,
    ) -> Awaitable[dict[int, T]]:
        """`works` as one step each, by item index. Starts only as many as the quorum
        still needs, `max_active` at a time; returns the first `at_least` to land (by
        recorded finish time, so a resume picks the same ones) and cancels the rest.
        `within` seconds after the start, nothing more starts."""
        return self._spread(
            self._path(name),
            list(works),
            at_least,
            max_active,
            within,
            retries,
            timeout,
        )

    async def _spread(
        self,
        path: str,
        works: list[Work[T]],
        at_least: int | None,
        max_active: int | None,
        within: float | None,
        retries: int,
        timeout: float | None,
    ) -> dict[int, T]:
        need = len(works) if at_least is None else at_least
        if not 1 <= need <= len(works):
            raise ValueError(f"{path}: at_least={at_least} over {len(works)} items")
        self._event("spread_started", path=path, need=need)
        done: dict[int, tuple[Any, StepRecord]] = {}
        for i, work in enumerate(works):
            if (attached := self._attached(path, work, i)) is not None:
                self._event("step_attached", path=path, index=i)
                done[i] = attached
        failures: list[str] = []
        pending = deque(i for i in range(len(works)) if i not in done)
        running: dict[asyncio.Task, int] = {}
        deadline = None if within is None else time.monotonic() + within

        async def one(i: int) -> tuple[Any, StepRecord]:
            return await self._step(path, works[i], i, retries, timeout)

        try:
            while len(done) < need:
                while (
                    pending
                    and len(done) + len(running) < need
                    and (max_active is None or len(running) < max_active)
                    and (deadline is None or time.monotonic() < deadline)
                ):
                    i = pending.popleft()
                    running[asyncio.create_task(one(i))] = i
                if not running:
                    break
                wait = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                finished, _ = await asyncio.wait(
                    running, timeout=wait, return_when=asyncio.FIRST_COMPLETED
                )
                if not finished:
                    break  # past the deadline with nothing landing
                for task in finished:
                    i = running.pop(task)
                    try:
                        done[i] = task.result()
                    except StepFailed as exc:
                        failures.append(exc.error)
        finally:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
        if len(done) < need:
            self._event("spread_finished", path=path, need=need, landed=len(done))
            raise StepFailed(path, f"{len(done)}/{need} items landed: {failures[:3]}")
        chosen = sorted(done.items(), key=lambda kv: (kv[1][1].finished_at, kv[0]))[
            :need
        ]
        self._event("spread_finished", path=path, need=need, landed=len(done))
        return {i: value for i, (value, _) in sorted(chosen)}

    @asynccontextmanager
    async def runtime(
        self, seat: str, task: Task | None = None
    ) -> AsyncIterator[Runtime]:
        """A box from the seat's runtime policy (resolved for `task` when given), alive
        for the block and always torn down; steps run in it with
        `agent(..., runtime=box)` or `command(...)`."""
        config = self.seat(seat).runtime
        if task is not None:
            config = resolve_runtime_config(config, task, set())
        async with self.run.pools.hold((RUNTIMES,)), provision_runtime(config) as box:
            box.env = dict(task.runtime_env()) if task is not None else {}
            yield box

    async def _step(
        self,
        path: str,
        work: Work,
        index: int | None,
        retries: int,
        timeout: float | None,
    ) -> tuple[Any, StepRecord]:
        run = self.run
        if (attached := self._attached(path, work, index)) is not None:
            self._event("step_attached", path=path, index=index)
            return attached
        key = self._key(path, work)
        tag = f"{self.key}/{path}" + (f".{index}" if index is not None else "")
        run.steps.add(tag)
        self._event("step_started", path=path, index=index)
        started, attempts = now(), 0
        try:
            while True:
                if run._draining.is_set():
                    raise Stopped(path)
                attempts += 1
                try:
                    async with asyncio.timeout(timeout):
                        value, extra = await self._execute(work)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    logger.warning("%s: attempt %d failed: %s", tag, attempts, error)
                    if attempts > retries or isinstance(exc, Oversized):
                        run.ledger.put(
                            self._record(
                                path,
                                index,
                                work,
                                key,
                                "error",
                                started,
                                attempts,
                                error=error,
                            )
                        )
                        self._event("step_failed", path=path, index=index, error=error)
                        raise StepFailed(path, error) from exc
                    delay = backoff(attempts - 1)
                    self._event(
                        "step_retrying",
                        path=path,
                        index=index,
                        attempt=attempts,
                        error=error,
                        backoff=delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                record = self._record(
                    path, index, work, key, "completed", started, attempts, **extra
                )
                run.ledger.put(record)
                self._event("step_completed", path=path, index=index)
                return value, record
        finally:
            run.steps.discard(tag)

    def _key(self, path: str, work: Work) -> str:
        """The step's key: its place and the content of its work. An agent's work also
        keys on its resolved seat — the model and effort the rollout ran under — so a
        seat change re-runs that seat's steps only; command and fn steps do not
        depend on the run's config."""
        content = work.content()
        if isinstance(work, AgentWork):
            content = [*content, self.run.seat(work.seat).model_dump(mode="json")]
        return digest(self.run.identity, self.key, path, content)[:STEP_KEY]

    def _attached(
        self, path: str, work: Work, index: int | None
    ) -> tuple[Any, StepRecord] | None:
        """The finished record for this step, with its value rebuilt, when the ledger has one."""
        record = self.run.ledger.get(self.key, path, index)
        if (
            record is None
            or record.key != self._key(path, work)
            or record.terminal != "completed"
        ):
            return None
        value = self._rehydrate(work, record)
        return None if value is _MISSING else (value, record)

    async def _execute(self, work: Work) -> tuple[Any, dict[str, Any]]:
        run = self.run
        if isinstance(work, FnWork):
            value = work.func(*work.args, **work.kwargs)
            if inspect.isawaitable(value):
                value = await value
            return value, {"payload": self._payload(value)}
        if isinstance(work, CommandWork):
            result = await work.runtime.run(work.argv, work.env)
            return result, {"payload": to_jsonable_python(result)}
        agent = make_agent(run.seat(work.seat), interception=run._inference)
        held = () if work.runtime is not None else (RUNTIMES,)
        async with run.pools.hold(held), agent:
            trace = await agent.run(work.task, runtime=work.runtime)
        await run.ledger.append(trace)
        run.usage = Usage.aggregate(u for u in (run.usage, trace.usage) if u)
        if not trace.ok:
            raise _RolloutFailed(trace)
        return trace, {"trace_id": trace.id}

    def _payload(self, value: Any) -> Any:
        payload = to_jsonable_python(value)
        size = len(json.dumps(payload))
        if size > self.run.config.payload_cap:
            raise Oversized(
                f"step value is {size} bytes, over payload_cap; keep bulk in traces or files"
            )
        return payload

    def _rehydrate(self, work: Work, record: StepRecord) -> Any:
        if isinstance(work, AgentWork):
            trace = self.run.ledger.trace(record.trace_id or "")
            return _MISSING if trace is None else trace
        if isinstance(work, CommandWork):
            return ProgramResult(**record.payload)
        try:
            hint = typing.get_type_hints(work.func).get("return")
        except (NameError, TypeError):
            hint = None
        if isinstance(hint, type) and issubclass(hint, BaseModel):
            return hint.model_validate(record.payload)
        return record.payload

    def _record(
        self,
        path: str,
        index: int | None,
        work: Work,
        key: str,
        terminal: Terminal,
        started: str,
        attempts: int,
        **fields: Any,
    ) -> StepRecord:
        return StepRecord(
            key=key,
            row=self.key,
            path=path,
            index=index,
            kind=work.kind,
            terminal=terminal,
            attempts=attempts,
            started_at=started,
            finished_at=now(),
            **fields,
        )


async def _aiter(rows: Iterable[Any] | AsyncIterable[Any]) -> AsyncIterable[Any]:
    if isinstance(rows, AsyncIterable):
        async for row in rows:
            yield row
    else:
        for row in rows:
            yield row
