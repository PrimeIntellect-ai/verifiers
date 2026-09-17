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
import fcntl
import inspect
import logging
import os
import re
import resource
import signal
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, Self, TypeVar

from verifiers.v1.configs.agent import (
    AgentConfig,
    declared_agent_configs,
    resolve_agent,
)
from verifiers.v1.flow.config import ROWS, RUNTIMES, FlowConfig
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
from verifiers.v1.flow.work import Oversized, Work
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

LABEL_MAX = 60
"""Chars the run's label is cut to: the sandbox label a sweep finds its boxes by."""
DRAIN_FILE = "drain"
"""A file of this name in the run directory drains the run, as Ctrl-C once does: for a
supervisor or an operator without the process's terminal. Remove it to launch again."""
RSS_STEP_MB = 256
"""An `rss` event each time the process's peak resident memory grows by this much."""
_SCOPE: ContextVar[tuple[str, ...]] = ContextVar("flow_scope", default=())

T = TypeVar("T")
RowState = Literal["running", "ok", "failed", "stopped"]
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
"""What a scope label or step name may be: a path segment, so records never collide on disk."""


class StepFailed(Exception):
    """A step used up its retries, or failed in a way no retry mends: `<path>: <error>`,
    the cause chained. What a flow catches to route on a failed step. A failed spread
    carries `failures`, the exception per item index that did not land."""

    def __init__(self, message: str, failures: dict[int, BaseException] | None = None):
        super().__init__(message)
        self.failures = failures or {}


def _name(name: str) -> str:
    if not NAME.fullmatch(name):
        raise ValueError(
            f"{name!r}: a scope label or step name must match {NAME.pattern}"
        )
    return name


class Stopped(Exception):
    """The run is draining: no step starts; the row returns stopped, to resume later."""


@dataclass
class RowResult:
    row: str
    state: RowState
    value: Any = None
    error: str | None = None


class Run:
    """The ledger, the pools, the interception, and the row loop for one run directory.

    A resume is the same launch against the same directory: a step attaches to its
    record when its place and content still match, and the config re-keys nothing
    (an agent step keys on its resolved seat, all else on its work alone).

    `async with Run(run_dir, config) as run` owns the directory for the block: the lock
    is taken before anything is read or repaired, and `sweep` and `stream` run inside.
    `run.run(flow, rows)` on its own takes ownership for the duration of the call."""

    def __init__(self, run_dir: Path, config: FlowConfig) -> None:
        self.run_dir = run_dir
        self.config = config
        self.pools = Pools(config.pools)
        where = digest(str(run_dir.resolve()))[:SHORT]
        self.label = f"flow-{run_dir.name}-{where}"[:LABEL_MAX]
        self.rows: dict[str, RowState] = {}
        """Every row seen so far, by key, and where it stands."""
        self.interception: Interception | None = None  # live inside `_serving`
        self._draining = asyncio.Event()
        self._rss_mark = 0
        self._lock: IO[str] | None = None

    async def __aenter__(self) -> Self:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        lock = (self.run_dir / "run.lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError(f"{self.run_dir} is in use by another launch") from None
        self._lock = lock
        self.ledger = Ledger(self.run_dir)  # repairs torn tails: only under the lock
        (self.run_dir / "config.json").write_text(self.config.model_dump_json(indent=1))
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._lock is not None
        self._lock.close()  # releases the flock
        self._lock = None

    @property
    def owned(self) -> bool:
        return self._lock is not None

    # -- rows -----------------------------------------------------------------------

    async def stream(self, flow: Flow, rows: Iterable[Any]) -> AsyncIterator[RowResult]:
        """Run `flow(ctx, row)` for every row, `pools["rows"]` at once, yielding each
        result as its row finishes. A row that raises is a failed row."""

        async def one(row: Any) -> RowResult:
            async with self.pools.hold((ROWS,)):
                return await self._row(flow, row)

        async with nullcontext() if self.owned else self, self._serving():
            self.ledger.event("run", source=self._source(flow), label=self.label)
            tasks = [asyncio.create_task(one(row)) for row in rows]
            try:
                for done in asyncio.as_completed(tasks):
                    yield await done
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self, flow: Flow, rows: Iterable[Any]) -> list[RowResult]:
        return [result async for result in self.stream(flow, rows)]

    async def sweep(self) -> int:
        """Kill what an earlier launch of this run left behind -- Prime sandboxes and Docker
        containers by label, host subprocesses by the same label in their environment; the
        count. Call before `stream` at a resume, inside `async with Run(...)`."""
        if not self.owned:
            raise RuntimeError(
                "sweep needs ownership of the run: `async with Run(...) as run`"
            )
        from verifiers.v1.runtimes.docker import sweep_containers
        from verifiers.v1.runtimes.prime import sweep_sandboxes
        from verifiers.v1.runtimes.subprocess import sweep_subprocesses

        return (
            sweep_subprocesses(self.label)
            + await sweep_containers(self.label)
            + await sweep_sandboxes([self.label])
        )

    def drain(self) -> None:
        """Stop admitting steps; in-flight steps finish and record, rows return stopped."""
        if not self._draining.is_set():
            self.ledger.event("drain")
            self._draining.set()

    @property
    def draining(self) -> bool:
        if not self._draining.is_set() and (self.run_dir / DRAIN_FILE).exists():
            self.drain()
        return self._draining.is_set()

    def _watch_rss(self) -> None:
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        if mb >= self._rss_mark + RSS_STEP_MB:
            self._rss_mark = mb - mb % RSS_STEP_MB
            self.ledger.event("rss", mb=mb)

    def seat(self, name: str) -> AgentConfig:
        """The seat with the run's defaults filled in: the identity its steps key on."""
        cfg = self.config
        return resolve_agent(
            getattr(cfg, name),
            model=cfg.model,
            client=cfg.client,
            sampling=cfg.sampling,
        )

    async def _row(self, flow: Flow, row: Any) -> RowResult:
        try:
            key = row_key(row)
        except Exception as exc:  # noqa: BLE001 — the row itself is the problem being reported
            return RowResult(
                row=repr(row)[:80], state="failed", error=f"row cannot be keyed: {exc}"
            )
        if key in self.rows:
            return RowResult(row=key, state="failed", error="duplicate row key")
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
            self.interception = interception
            try:
                yield
            finally:
                self.interception = None

    def _source(self, flow: Flow) -> str | None:
        """The flow's source hash, for the `run` event; a resume under changed code
        attaches to the steps whose inputs still match, so say so once. None when the
        source is unavailable."""
        try:
            source = digest(inspect.getsource(flow))[:SHORT]
        except (OSError, TypeError):
            return None
        runs = [e for e in self.ledger.events() if e["type"] == "run"]
        if runs and runs[-1].get("source") not in (None, source):
            logger.warning(
                "%s: the flow's source changed since this run was written", self.run_dir
            )
        return source


class Ctx:
    """One row's handle on the run: steps, spreads, runtimes, scopes."""

    def __init__(self, run: Run, key: str, row: Any) -> None:
        self.run, self.key, self.row = run, key, row
        self._counts: dict[tuple[str, ...], dict[str, int]] = {}
        self._attached_paths: set[str] = set()

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
        token = _SCOPE.set((*_SCOPE.get(), _name(label)))
        try:
            yield
        finally:
            _SCOPE.reset(token)

    def _path(self, name: str) -> str:
        _name(name)
        scope = _SCOPE.get()
        counts = self._counts.setdefault(scope, {})
        n = counts.get(name, 0)
        counts[name] = n + 1
        return "/".join([*scope, f"{name}#{n}"])

    def _event(self, kind: EventKind, **fields: Any) -> None:
        """One event for this row: the row key always rides along."""
        self.run.ledger.event(kind, row=self.key, **fields)

    def attached(self, name: str) -> bool:
        """Whether the latest step called `name` in this scope came from the ledger rather
        than ran: what a flow checks when a later step assumed that step's side effects
        (a world it brought up in a shared box) and must redo them on a resume."""
        scope = _SCOPE.get()
        n = self._counts.get(scope, {}).get(name, 0) - 1
        return "/".join([*scope, f"{name}#{n}"]) in self._attached_paths

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
        return self._step(self._path(name), work, None, retries, timeout)

    def spread(
        self,
        name: str,
        works: Iterable[Work[T]],
        *,
        retries: int = 0,
        timeout: float | None = None,
    ) -> Awaitable[dict[int, T]]:
        """`works` as one step each, by item index, all at once. Every item must land,
        else the spread fails with the first few errors."""
        return self._spread(self._path(name), list(works), retries, timeout)

    async def _spread(
        self, path: str, works: list[Work[T]], retries: int, timeout: float | None
    ) -> dict[int, T]:
        self._event("spread_started", path=path, items=len(works))
        results = await asyncio.gather(
            *(
                self._step(path, work, i, retries, timeout)
                for i, work in enumerate(works)
            ),
            return_exceptions=True,
        )
        failed = {i: r for i, r in enumerate(results) if isinstance(r, BaseException)}
        self._event(
            "spread_finished",
            path=path,
            items=len(works),
            landed=len(works) - len(failed),
        )
        for exc in failed.values():
            if isinstance(exc, Stopped):
                raise exc
        if failed:
            errors = [str(exc) for exc in failed.values()]
            raise StepFailed(
                f"{path}: {len(failed)}/{len(works)} items failed: {errors[:3]}", failed
            )
        return dict(enumerate(results))  # type: ignore[arg-type]

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
        env = task.runtime_env() if task is not None else None
        async with (
            self.run.pools.hold((RUNTIMES,)),
            provision_runtime(config, env=env) as box,
        ):
            yield box

    async def _step(
        self,
        path: str,
        work: Work[T],
        index: int | None,
        retries: int,
        timeout: float | None,
    ) -> T:
        run = self.run
        key = self._key(path, work)
        record = run.ledger.get(self.key, path, index)
        # Why the step runs rather than attaches, on the `step_started` event: the first
        # thing to read when a resume re-runs work it should have found.
        reason = (
            "no_record"
            if record is None
            else "previous_failed"
            if record.terminal != "completed"
            else "key_changed"
            if record.key != key
            else None
        )
        if reason is None:
            try:
                value = work.load(run.ledger, record)  # type: ignore[arg-type]
            except LookupError as exc:
                logger.warning("%s/%s: %s; running the step again", self.key, path, exc)
                reason = "load_failed"
            else:
                self._attached_paths.add(path)
                self._event("step_attached", path=path, index=index)
                return value
        if run.draining:
            raise Stopped(path)
        tag = f"{self.key}/{path}" + (f".{index}" if index is not None else "")
        self._event("step_started", path=path, index=index, reason=reason)
        started, attempts = now(), 0
        try:
            while True:
                attempts += 1
                try:
                    async with asyncio.timeout(timeout):
                        value = await work.execute(self)
                        fields = work.dump(self, value)
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
                        raise StepFailed(f"{path}: {error}") from exc
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
                run.ledger.put(
                    self._record(
                        path, index, work, key, "completed", started, attempts, **fields
                    )
                )
                self._event("step_completed", path=path, index=index)
                run._watch_rss()
                return value
        except asyncio.CancelledError:
            self._event("step_cancelled", path=path, index=index)
            raise

    def _key(self, path: str, work: Work) -> str:
        """The step's key: its place in this row and the content of its work."""
        return digest(self.key, path, work.content(self))[:STEP_KEY]

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


Flow = Callable[[Ctx, Any], Awaitable[Any]]
"""A flow: `async def flow(ctx, row) -> value`."""


def drain_on_interrupt(run: Run) -> None:
    """Route SIGINT and SIGTERM to `run` from inside its loop: the first drains (steps
    in flight finish and record, rows return stopped), the second cancels the task."""
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    assert task is not None

    def interrupt() -> None:
        if run.draining:
            task.cancel()
        else:
            logger.warning("draining: in-flight steps finish; Ctrl-C again cancels")
            run.drain()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, interrupt)
