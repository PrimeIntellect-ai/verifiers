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
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from verifiers.v1.agent import make_agent
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.errors import infrastructure, permanent
from verifiers.v1.flow.config import FlowConfig
from verifiers.v1.flow.ledger import Ledger, StepRecord, digest, now, row_key
from verifiers.v1.flow.pools import Pools
from verifiers.v1.flow.work import AgentWork, CommandWork, FnWork, Work
from verifiers.v1.interception import InterceptionServer
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
from verifiers.v1.utils.compile import resolve_runtime_config

logger = logging.getLogger("verifiers.flow")

_MISSING = object()  # a record whose value cannot be rebuilt: run the step again
RUNTIMES = "runtimes"
_SCOPE: ContextVar[tuple[str, ...]] = ContextVar("flow_scope", default=())


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
        self.trace = trace
        self.last = trace.last_error
        super().__init__(
            f"{self.last.type}: {self.last.message}" if self.last else "rollout failed"
        )


@dataclass
class RowResult:
    row: str
    ok: bool
    value: Any = None
    error: str | None = None
    stopped: bool = False


class Run:
    """The ledger, the pools, the inference gate, and the row loop for one run directory.

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
        self.identity = digest(run_dir.name)[:16]
        self.label = f"flow-{run_dir.name}-{self.identity}"[:60]
        self._require_identity(rekey)
        self.rows: dict[str, str] = {}  # row key -> running | ok | failed | stopped
        self.steps: set[str] = set()  # `<row>/<path>` in flight
        self.tokens = {"input": 0, "output": 0}
        self._draining = asyncio.Event()
        self._admissions = (
            asyncio.Event()
        )  # cleared while an infrastructure outage is on
        self._admissions.set()
        self._inference: InterceptionServer | None = None
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

    def status(self) -> dict[str, Any]:
        return {
            "rows": dict(self.rows),
            "steps": sorted(self.steps),
            "holding": not self._admissions.is_set(),
            "draining": self._draining.is_set(),
            "tokens": dict(self.tokens),
        }

    def seat(self, name: str) -> AgentConfig:
        cfg: AgentConfig = getattr(self.config, name)
        update = {}
        if cfg.model is None and self.config.model is not None:
            update["model"] = self.config.model
        if cfg.client is None and self.config.client is not None:
            update["client"] = self.config.client
        return cfg.model_copy(update=update) if update else cfg

    async def _row(self, flow: Callable[..., Any], row: Any) -> RowResult:
        try:
            key = row_key(row)
        except Exception as exc:  # noqa: BLE001 — the row itself is the problem being reported
            return RowResult(
                row=repr(row)[:80], ok=False, error=f"row cannot be keyed: {exc}"
            )
        self.rows[key] = "running"
        self.ledger.event("row_started", row=key)
        try:
            value = await flow(Ctx(self, key, row), row)
            self.rows[key] = "ok"
            self.ledger.event("row_finished", row=key, state="ok")
            return RowResult(row=key, ok=True, value=value)
        except Stopped:
            self.rows[key] = "stopped"
            self.ledger.event("row_finished", row=key, state="stopped")
            return RowResult(row=key, ok=False, stopped=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a failed row is data, never a failed run
            self.rows[key] = "failed"
            error = f"{type(exc).__name__}: {exc}"
            self.ledger.event("row_finished", row=key, state="failed", error=error)
            return RowResult(row=key, ok=False, error=error)

    @asynccontextmanager
    async def _serving(self) -> AsyncIterator[None]:
        set_base_sandbox_labels([self.label])
        os.environ[RUN_LABEL_VAR] = (
            self.label
        )  # every host subprocess inherits it: what `sweep` finds
        async with AsyncExitStack() as stack:
            if self.config.inference_concurrency is not None:
                remote = any(
                    not runtime_is_local(self.seat(name).runtime)
                    for name, field in type(self.config).model_fields.items()
                    if field.annotation is AgentConfig
                )
                self._inference = await stack.enter_async_context(
                    InterceptionServer(
                        requires_tunnel=remote,
                        max_inflight=self.config.inference_concurrency,
                    )
                )
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
            source = digest(inspect.getsource(flow))[:16]
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

    def _outage(self, on: bool) -> None:
        if on and self._admissions.is_set():
            logger.warning(
                "infrastructure failure: holding new steps until one succeeds"
            )
            self._admissions.clear()
            self.ledger.event("holding", on=True)
        elif not on and not self._admissions.is_set():
            self._admissions.set()
            self.ledger.event("holding", on=False)


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

    def _event(self, kind: str, **fields: Any) -> None:
        """One event for this row: the row key always rides along."""
        self.run.ledger.event(kind, row=self.key, **fields)

    # -- steps ----------------------------------------------------------------------

    def step(
        self, name: str, work: Work, *, retries: int = 0, timeout: float | None = None
    ) -> Awaitable[Any]:
        """Run `work` once, durably. The value is the trace, the program result, or
        the function's return; on resume it comes from the ledger. `retries` re-run the
        whole work after a turn failure, on top of an agent's own `AgentConfig.retries`."""
        return self._step_value(self._path(name), work, retries, timeout)

    async def _step_value(
        self, path: str, work: Work, retries: int, timeout: float | None
    ) -> Any:
        value, _ = await self._step(path, work, None, retries, timeout)
        return value

    def spread(
        self,
        name: str,
        works: Iterable[Work],
        *,
        at_least: int | None = None,
        max_active: int | None = None,
        within: float | None = None,
        retries: int = 0,
        timeout: float | None = None,
    ) -> Awaitable[dict[int, Any]]:
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
        works: list[Work],
        at_least: int | None,
        max_active: int | None,
        within: float | None,
        retries: int,
        timeout: float | None,
    ) -> dict[int, Any]:
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
        if run._draining.is_set():
            raise Stopped(path)
        await run._admissions.wait()
        tag = f"{self.key}/{path}" + (f".{index}" if index is not None else "")
        run.steps.add(tag)
        self._event("step_started", path=path, index=index)
        started, attempts, held_since, backoff = (
            now(),
            0,
            None,
            run.config.outage_backoff_s,
        )
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
                except (
                    Exception
                ) as exc:  # classified below: permanent, the world's, or the attempt's
                    error = f"{type(exc).__name__}: {exc}"
                    cause = exc.last if isinstance(exc, _RolloutFailed) else exc
                    if permanent(cause) or isinstance(exc, Oversized):
                        self._event("step_failed", path=path, index=index, error=error)
                        raise StepFailed(path, error) from exc
                    if infrastructure(cause):
                        held_since = held_since or time.monotonic()
                        if time.monotonic() - held_since > run.config.outage_hold_s:
                            self._event(
                                "step_failed",
                                path=path,
                                index=index,
                                error=f"held {run.config.outage_hold_s:g}s: {error}",
                            )
                            raise StepFailed(
                                path, f"held {run.config.outage_hold_s:g}s: {error}"
                            ) from exc
                        run._outage(True)
                        self._event(
                            "step_retrying",
                            path=path,
                            index=index,
                            attempt=attempts,
                            error=error,
                            backoff=backoff,
                        )
                        logger.warning("%s: %s; retrying in %.0fs", tag, error, backoff)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, run.config.outage_hold_s)
                        continue
                    logger.warning("%s: attempt %d failed: %s", tag, attempts, error)
                    if attempts > retries:
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
                    self._event(
                        "step_retrying",
                        path=path,
                        index=index,
                        attempt=attempts,
                        error=error,
                        backoff=0.0,
                    )
                    continue
                record = self._record(
                    path, index, work, key, "completed", started, attempts, **extra
                )
                run.ledger.put(record)
                self._event("step_completed", path=path, index=index)
                return value, record
        finally:
            if held_since is not None:
                run._outage(False)  # this step's outage is over, one way or the other
            run.steps.discard(tag)

    def _key(self, path: str, work: Work) -> str:
        """The step's key: its place and the content of its work. An agent's work also
        keys on its resolved seat — the model and effort the rollout ran under — so a
        seat change re-runs that seat's steps only; command and fn steps do not
        depend on the run's config."""
        content = work.content()
        if isinstance(work, AgentWork):
            content = [*content, self.run.seat(work.seat).model_dump(mode="json")]
        return digest(self.run.identity, self.key, path, content)[:24]

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
        await run.ledger.append(trace, env="flow")
        if usage := trace.usage:
            run.tokens["input"] += usage.prompt_tokens + (
                usage.cached_input_tokens or 0
            )
            run.tokens["output"] += usage.completion_tokens
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
        terminal: str,
        started: str,
        attempts: int,
        **fields: Any,
    ) -> StepRecord:
        return StepRecord(
            key=key,
            row=self.key,
            path=path,
            index=index,
            kind=type(work).__name__.removesuffix("Work").lower(),
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
