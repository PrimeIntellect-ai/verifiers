"""The walker: one row at a time, ready nodes run under pools, every node instance
lands in the ledger, and a node may run in the live runtime an earlier node used."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import typing
from collections.abc import AsyncIterable, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from verifiers.v1.agent import make_agent
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.flow.compile import FlowError, Graph
from verifiers.v1.flow.flow import Flow
from verifiers.v1.flow.ledger import Ledger, NodeRecord, digest, now, row_key
from verifiers.v1.flow.nodes import (
    AgentNode,
    ExpandNode,
    FnNode,
    Join,
    Node,
    RunNode,
    Target,
    _End,
    _names,
)
from verifiers.v1.flow.outcome import (
    OutcomeTools,
    OutcomeToolsConfig,
    outcome_of,
    parse_outcome,
)
from verifiers.v1.flow.pools import Pools
from verifiers.v1.interception import InterceptionServer
from verifiers.v1.mcp import SharedToolServer, serve_shared
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    provision_runtime,
    runtime_is_local,
)
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

logger = logging.getLogger("verifiers.flow")

_MISSING = object()  # a ledger record whose value cannot be rebuilt: run it again


class RunResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    outcome: str


class Upstream:
    """What a node sees of its row: finished results by node name, their outcomes,
    the input row and the flow config."""

    def __init__(
        self,
        row: Any,
        config: Any,
        results: dict[str, Any],
        outcomes: dict[str, str | None],
        items: dict[str, dict[int, Any]] | None = None,
    ) -> None:
        self.row = row
        self.config = config
        self._results = results
        self._outcomes = outcomes
        self._items = items or {}

    def __getattr__(self, name: str) -> Any:
        results = self.__dict__.get("_results") or {}
        if name in results:
            return results[name]
        raise AttributeError(
            f"no upstream result named {name!r}; have {sorted(results)}"
        )

    def outcome(self, name: str) -> str | None:
        return self._outcomes.get(name)

    def items(self, name: str) -> dict[int, Any]:
        """A fan-out node's results by original item index; unlike the positional
        `up.<name>` list, indices hold when a join drops items."""
        return dict(self._items.get(name, {}))


@dataclass
class RowResult:
    row: str
    ok: bool
    records: dict[str, NodeRecord] = field(default_factory=dict)
    error: str | None = None


@dataclass
class _Done:
    node: str
    record: NodeRecord
    value: Any
    failed: bool = False

    @property
    def outcome(self) -> str | None:
        return self.record.outcome


class Engine:
    def __init__(self, flow: Flow, run_dir: Path) -> None:
        self.graph: Graph = flow.graph
        self.config = flow.config
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(run_dir)
        self.pools = Pools(self.config.pools)
        unknown = {
            pool
            for node in self.graph.nodes.values()
            for pool in node.pools
            if pool not in self.config.pools
        }
        if unknown:
            raise FlowError(
                [
                    f"unknown pools {sorted(unknown)}; config.pools has {sorted(self.config.pools)}"
                ]
            )
        self._stack = AsyncExitStack()
        self._outcome_tools: dict[str, dict[str, SharedToolServer]] = {}
        self._inference: InterceptionServer | None = None
        # Any change to the graph shape or the config invalidates every ledger key.
        self.identity = digest(
            self.graph.structural_hash(), self.config.model_dump(mode="json")
        )[:16]
        (run_dir / "graph.json").write_text(
            json.dumps(self.graph.to_json(), indent=1, sort_keys=True)
        )
        (run_dir / "config.json").write_text(self.config.model_dump_json(indent=1))

    async def run(self, rows: Iterable[Any] | AsyncIterable[Any]) -> list[RowResult]:
        """Run every row; rows from an async iterable start as they arrive, so a
        producer (a miner, a queue) can feed the flow while it runs. A row that
        crashes is a failed row, never a failed run."""
        gate = asyncio.Semaphore(self.config.max_concurrent_rows)

        async def one(row: Any) -> RowResult:
            async with gate:
                try:
                    return await _Row(self, row).run()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — reported on the row
                    return RowResult(
                        row=_row_key_or_repr(row),
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )

        async with self._stack:
            self._outcome_tools.clear()  # a previous run's servers exited with its stack
            self._inference = None
            if self.config.inference_concurrency is not None:
                self._inference = await self._stack.enter_async_context(
                    InterceptionServer(
                        requires_tunnel=self._any_remote_seat(),
                        max_inflight=self.config.inference_concurrency,
                    )
                )
            tasks: list[asyncio.Task] = []
            try:
                async for row in _aiter(rows):
                    tasks.append(asyncio.create_task(one(row)))
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            return list(await asyncio.gather(*tasks))

    async def outcome_tools(
        self, node: AgentNode | ExpandNode
    ) -> dict[str, SharedToolServer]:
        """One `submit_outcome` server per node that declares outcomes, started on
        first use and shared by every rollout of that node."""
        if node.name not in self._outcome_tools:
            allowed = [name for name in node.outcomes or {} if name != "*"]
            toolset = OutcomeTools(OutcomeToolsConfig(allowed=allowed))
            local = runtime_is_local(self.seat(node.seat).runtime)
            servers = await self._stack.enter_async_context(
                serve_shared([toolset], harness_is_local=local)
            )
            self._outcome_tools[node.name] = servers
        return self._outcome_tools[node.name]

    def seat(self, name: str) -> AgentConfig:
        cfg: AgentConfig = getattr(self.config, name)
        update = {}
        if cfg.model is None and self.config.model is not None:
            update["model"] = self.config.model
        if cfg.client is None and self.config.client is not None:
            update["client"] = self.config.client
        return cfg.model_copy(update=update) if update else cfg

    def _any_remote_seat(self) -> bool:
        return any(
            not runtime_is_local(self.seat(node.seat).runtime)
            for node in self.graph.nodes.values()
            if isinstance(node, (AgentNode, ExpandNode)) and node.seat
        )


class _Row:
    def __init__(self, engine: Engine, row: Any) -> None:
        self.e = engine
        self.g = engine.graph
        self.row = row
        self.key = row_key(row)
        self.visits: dict[str, int] = {}
        self.fired: dict[str, dict[str, str | None]] = {}
        self.pending: set[str] = set()
        self.running: dict[asyncio.Task, str] = {}
        self.done: dict[str, _Done] = {}
        self.runtimes: dict[str, Runtime] = {}  # live runtimes other nodes inherit
        self.attached: set[str] = set()  # nodes restored from the ledger, not re-run
        self.items: dict[str, dict[int, Any]] = {}  # fan-out results by item index
        self.stack = AsyncExitStack()
        self.error: str | None = None

    # -- the loop -----------------------------------------------------------------

    async def run(self) -> RowResult:
        async with self.stack:
            try:
                self.pending.add(self.g.entry)
                while self.pending or self.running:
                    self._start_ready()
                    if not self.running:
                        if self.pending and self.error is None:
                            self.error = f"stuck: {sorted(self.pending)} can never fire"
                        break
                    finished, _ = await asyncio.wait(
                        self.running, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in finished:
                        name = self.running.pop(task)
                        self._complete(name, task.result())
                    if self.error:
                        break
            finally:
                # Nothing may run on past the row: it would write the ledger and
                # touch runtimes after the caller has moved on.
                for task in self.running:
                    task.cancel()
                await asyncio.gather(*self.running, return_exceptions=True)
        records = {d.node: d.record for d in self.done.values()}
        return RowResult(
            row=self.key, ok=self.error is None, records=records, error=self.error
        )

    def _start_ready(self) -> None:
        """Start every ready pending node; an exhausted node routes synchronously and
        may make new nodes ready, so scan until nothing more starts."""
        started = True
        while started and not self.error:
            started = False
            for name in sorted(self.pending):
                if name in self.running.values() or not self._ready(name):
                    continue
                self.pending.discard(name)
                fired = self.fired.pop(name, {})
                visit = self.visits[name] = self.visits.get(name, 0) + 1
                node = self.g.nodes[name]
                started = True
                if node.max_visits is not None and visit > node.max_visits:
                    record = self._record(
                        node, visit, None, [], terminal="exhausted", outcome="exhausted"
                    )
                    self.e.ledger.put(record)
                    self.done[name] = _Done(name, record, None)
                    self._route(
                        node, node.on_exhausted, "exhausted", "exhausted max_visits"
                    )
                    continue
                task = asyncio.create_task(
                    self._execute(node, visit, fired),
                    name=f"{self.key}/{name}@{visit}",
                )
                self.running[task] = name

    def _ready(self, name: str) -> bool:
        """A join fires per its policy over predecessors that have fired; a
        predecessor no live node can still reach is dead, not awaited. A cycle's
        bounded node (`max_visits`) is re-entered by any predecessor it cycles
        through; its join is over the others."""
        preds = self.g.preds.get(name, set())
        if not preds:
            return True
        if not self.fired.get(name) and not self.visits.get(name):
            return True  # the entry: its predecessors are back edges
        fired = set(self.fired.get(name, {}))
        if self.g.nodes[name].max_visits is not None:
            back = {p for p in preds if p in self.g.reachable(name)}
            if fired & back:
                return True
            preds, fired = preds - back, fired - back
        live = set(self.running.values()) | (self.pending - {name})
        reach: set[str] = set()
        for source in live:
            reach |= self.g.reachable(source)
        alive = {p for p in preds - fired if p in reach}
        join = self.g.nodes[name].join
        if join.kind == "any":
            return bool(fired)
        if join.kind == "at_least":
            return bool(fired) and (
                len(fired) >= self._quorum(name, join, self._upstream()) or not alive
            )
        return bool(fired) and not alive

    @staticmethod
    def _quorum(name: str, join: Join, up: Upstream) -> int:
        k = join.k(up) if callable(join.k) else join.k
        if k < 1:
            raise FlowError([f"{name}: at_least({k}) needs k >= 1"])
        return k

    def _upstream(self) -> Upstream:
        return Upstream(
            self.row,
            self.e.config,
            {n: d.value for n, d in self.done.items()},
            {n: d.outcome for n, d in self.done.items()},
            self.items,
        )

    def _complete(self, name: str, done: _Done) -> None:
        self.done[name] = done
        node = self.g.nodes[name]
        if done.failed:
            self._route(node, node.on_error, "error", done.record.error or "failed")
        elif node.outcomes is not None:
            target = node.outcomes.get(done.outcome or "") or node.outcomes.get("*")
            self._route(
                node, target, done.outcome, f"unmatched outcome {done.outcome!r}"
            )
        else:
            self._activate(node.then, name, done.outcome)

    def _route(
        self, node: Node, target: Target | None, outcome: str | None, failure: str
    ) -> None:
        if target is None:
            self.error = f"{node.name}: {failure}"
            return
        self._activate(target, node.name, outcome)

    def _activate(
        self, target: Target | None, source: str, outcome: str | None
    ) -> None:
        if target is None or isinstance(target, _End):
            return
        for name in _names(target):
            self.fired.setdefault(name, {})[source] = outcome
            self.pending.add(name)

    # -- one node instance ---------------------------------------------------------

    async def _execute(
        self, node: Node, visit: int, fired: dict[str, str | None]
    ) -> _Done:
        up = self._upstream()
        upstream_keys = sorted(
            _attach_material(self.done[p]) for p in fired if p in self.done
        )
        key = digest(self.e.identity, self.key, node.name, visit, upstream_keys)[:24]
        existing = self.e.ledger.get(self.key, node.name, visit)
        if (
            existing
            and existing.key == key
            and existing.terminal == "completed"
            and (value := self._rehydrate(node, existing)) is not _MISSING
        ):
            logger.info("%s: %s@%s attached to ledger", self.key, node.name, visit)
            self.attached.add(node.name)
            return _Done(node.name, existing, value)

        started = now()
        error: str | None = None
        for attempt in range(1, node.retries + 2):
            try:
                outcome, value, extra = await self._run_kind(
                    node, visit, up, upstream_keys
                )
                record = self._record(
                    node,
                    visit,
                    key,
                    upstream_keys,
                    terminal="completed",
                    outcome=outcome,
                    started=started,
                    attempt=attempt,
                    payload=_payload(value),
                    **extra,
                )
                self.e.ledger.put(record)
                logger.info("%s: %s@%s -> %s", self.key, node.name, visit, outcome)
                return _Done(node.name, record, value)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — a node failure is data: recorded, then routed or retried
                error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "%s: %s@%s attempt %s failed: %s",
                    self.key,
                    node.name,
                    visit,
                    attempt,
                    error,
                )
        record = self._record(
            node,
            visit,
            key,
            upstream_keys,
            terminal="error",
            started=started,
            attempt=node.retries + 1,
            error=error,
        )
        self.e.ledger.put(record)
        return _Done(node.name, record, None, failed=True)

    def _record(
        self,
        node: Node,
        visit: int,
        key: str | None,
        upstream_keys: list[str],
        *,
        terminal: str,
        outcome: str | None = None,
        started: str | None = None,
        **fields: Any,
    ) -> NodeRecord:
        return NodeRecord(
            key=key
            or digest(self.e.identity, self.key, node.name, visit, upstream_keys)[:24],
            row=self.key,
            node=node.name,
            visit=visit,
            kind=node.kind,
            terminal=terminal,
            outcome=outcome,
            started_at=started or now(),
            finished_at=now(),
            **fields,
        )

    def _rehydrate(self, node: Node, record: NodeRecord) -> Any:
        if isinstance(node, ExpandNode):
            per = {
                i: value
                for i, rec in self.e.ledger.item_records(
                    self.key, node.name, record.visit
                ).items()
                if rec.terminal == "completed"
                and (value := self._rehydrate_item(node, rec)) is not _MISSING
            }
            self.items.setdefault(node.name, {}).update(per)
            if node.seat is None:
                return record.payload
            traces = [self.e.ledger.trace(tid) for tid in record.trace_ids]
            return _MISSING if any(t is None for t in traces) else traces
        if isinstance(node, AgentNode):
            trace = self.e.ledger.trace(record.trace_id or "")
            return _MISSING if trace is None else trace
        if isinstance(node, RunNode):
            return RunResult.model_validate(record.payload)
        if isinstance(node, FnNode):
            try:
                hint = typing.get_type_hints(node.func).get("return")
            except NameError:
                hint = None  # a local type annotation is not resolvable on resume
            if isinstance(hint, type) and issubclass(hint, BaseModel):
                return hint.model_validate(record.payload)
        return record.payload

    def _rehydrate_item(self, node: ExpandNode, record: NodeRecord) -> Any:
        if node.seat is None:
            return record.payload
        trace = self.e.ledger.trace(record.trace_id or "")
        return _MISSING if trace is None else trace

    async def _run_kind(
        self, node: Node, visit: int, up: Upstream, upstream_keys: list[str]
    ) -> tuple[str | None, Any, dict]:
        if isinstance(node, AgentNode):
            return await self._run_agent(node, up)
        if isinstance(node, ExpandNode):
            return await self._run_expand(node, visit, up, upstream_keys)
        if isinstance(node, RunNode):
            return await self._run_command(node)
        if isinstance(node, FnNode):
            return await self._run_fn(node, up)
        raise FlowError([f"{node.name}: unknown node kind {node.kind}"])

    # -- kinds ----------------------------------------------------------------------

    async def _run_agent(
        self, node: AgentNode, up: Upstream
    ) -> tuple[str | None, Trace, dict]:
        trace = await self._rollout(node, node.make_task(up))
        if not trace.ok:
            raise FlowError(
                [f"{node.name}: rollout failed: {[e.message for e in trace.errors]}"]
            )
        outcome, summary = outcome_of(trace) if node.outcomes else ("completed", "")
        return (
            outcome,
            trace,
            {"summary": summary, "trace_id": trace.id, "reward": trace.reward},
        )

    async def _run_expand(
        self, node: ExpandNode, visit: int, up: Upstream, upstream_keys: list[str]
    ) -> tuple[str | None, list[Any], dict]:
        """Every item under `max_active`, joined by `join`; each item is its own
        ledger record, keyed by the upstream material and the item."""
        items = list(node.over(up))
        gate = asyncio.Semaphore(node.max_active or max(len(items), 1))
        if node.join.kind == "all":
            need = len(items)
        elif node.join.kind == "any":
            need = 1
        else:
            need = self._quorum(node.name, node.join, up)

        async def one(i: int, item: Any) -> tuple[int, Any]:
            async with gate:
                key = digest(
                    self.e.identity, self.key, node.name, visit, upstream_keys, item
                )[:24]
                existing = self.e.ledger.get(self.key, node.name, visit, i)
                if (
                    existing
                    and existing.key == key
                    and existing.terminal == "completed"
                    and (value := self._rehydrate_item(node, existing)) is not _MISSING
                ):
                    self.items.setdefault(node.name, {})[i] = value
                    return i, value
                value, extra = await self._expand_item(node, up, item, i)
                record = self._record(
                    node,
                    visit,
                    key,
                    upstream_keys,
                    terminal="completed",
                    outcome="completed",
                    index=i,
                    **extra,
                )
                self.e.ledger.put(record)
                self.items.setdefault(node.name, {})[i] = value
                return i, value

        tasks = [asyncio.create_task(one(i, item)) for i, item in enumerate(items)]
        got: dict[int, Any] = {}
        failures: list[str] = []
        try:
            for fut in asyncio.as_completed(tasks):
                try:
                    i, value = await fut
                    got[i] = value
                except Exception as exc:  # noqa: BLE001 — one item's failure is counted, not fatal
                    failures.append(str(exc))
                if len(got) >= need:
                    break
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if len(got) < need:
            raise FlowError(
                [f"{node.name}: {len(got)}/{need} items succeeded: {failures[:3]}"]
            )
        values = [got[i] for i in sorted(got)]
        extra = {"trace_ids": [t.id for t in values]} if node.seat else {}
        return "completed", values, extra

    async def _expand_item(
        self, node: ExpandNode, up: Upstream, item: Any, i: int
    ) -> tuple[Any, dict]:
        if node.seat is None:
            async with self.e.pools.hold(node.pools):
                value = node.each(up, item)
                if inspect.isawaitable(value):
                    value = await value
            return value, {"payload": _payload(value)}
        trace = await self._rollout(node, node.each(up, item))
        if not trace.ok:
            raise FlowError([f"{node.name}[{i}]: rollout failed"])
        return trace, {"trace_id": trace.id, "reward": trace.reward}

    async def _run_command(self, node: RunNode) -> tuple[str | None, RunResult, dict]:
        async with self.e.pools.hold(node.pools), AsyncExitStack() as local:
            runtime = await self._runtime(node, local)
            result = await runtime.run(node.argv, node.env)
        outcome, _ = parse_outcome(result.stdout)
        if outcome is None:
            outcome = "completed" if result.exit_code == 0 else "failed"
        if node.outcomes is None and result.exit_code != 0:
            raise FlowError(
                [
                    f"{node.name}: exit {result.exit_code}: {result.stderr.strip()[-500:]}"
                ]
            )
        value = RunResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            outcome=outcome,
        )
        return outcome, value, {}

    async def _run_fn(self, node: FnNode, up: Upstream) -> tuple[str | None, Any, dict]:
        async with self.e.pools.hold(node.pools):
            value = node.func(up)
            if inspect.isawaitable(value):
                value = await value
        routes = node.outcomes is not None and isinstance(value, str)
        return (value if routes else "completed"), value, {}

    # -- runtimes -------------------------------------------------------------------

    async def _rollout(self, node: AgentNode | ExpandNode, task: Task) -> Trace:
        """Run the seat on `task`, in its own runtime unless the node inherits one
        or a later node inherits this one's. A node with outcomes gets the
        `submit_outcome` tool when its harness speaks MCP and its task's state
        carries `outcome`; otherwise the task sets `state.outcome` itself (in
        `finalize`) or the last reply ends with an `Outcome:` line."""
        agent = make_agent(self.e.seat(node.seat), interception=self.e._inference)
        tools = None
        if node.outcomes and agent.harness.SUPPORTS_MCP:
            if "outcome" in state_cls(type(task)).model_fields:
                tools = await self.e.outcome_tools(node)
            else:
                logger.warning(
                    "%s: %s has no `outcome` state field; routing falls back to an "
                    "`Outcome:` line in the last reply",
                    node.name,
                    type(task).__name__,
                )
        async with agent, self.e.pools.hold(node.pools), AsyncExitStack() as local:
            if node.inherits is None and node.name not in self.g.held:
                trace = await agent.run(task, tools=tools)
            else:
                runtime = await self._runtime(node, local, agent=agent, task=task)
                trace = await agent.run(task, runtime=runtime, tools=tools)
        await self.e.ledger.append(trace, env=self.g.name)
        return trace

    async def _runtime(
        self,
        node: Node,
        local: AsyncExitStack,
        *,
        agent: Any = None,
        task: Task | None = None,
    ) -> Runtime:
        """The runtime a node runs in: an inherited live one, or a fresh one that
        lives for the node, or for the row when a later node inherits it."""
        if node.inherits is not None:
            runtime = self.runtimes.get(node.inherits)
            if runtime is None:
                why = (
                    "was attached from the ledger; a live runtime does not survive "
                    "resume (delete its record to run it again)"
                    if node.inherits in self.attached
                    else "has no live runtime"
                )
                raise FlowError([f"{node.name}: inherit:{node.inherits} but it {why}"])
            return runtime
        held = node.name in self.g.held
        stack = self.stack if held else local
        if agent is not None:
            runtime = await stack.enter_async_context(agent.provision(task))
        else:
            runtime = await stack.enter_async_context(
                provision_runtime(self._runtime_config(node))
            )
        if held:
            self.runtimes[node.name] = runtime
        return runtime

    def _runtime_config(self, node: Node) -> RuntimeConfig:
        if not isinstance(node.runtime, str):
            return node.runtime
        seat = getattr(node, "seat", None)
        if seat:
            return self.e.seat(seat).runtime
        raise FlowError([f"{node.name}: no runtime config to provision from"])


async def _aiter(rows: Iterable[Any] | AsyncIterable[Any]) -> AsyncIterable[Any]:
    if isinstance(rows, AsyncIterable):
        async for row in rows:
            yield row
    else:
        for row in rows:
            yield row


def _row_key_or_repr(row: Any) -> str:
    try:
        return row_key(row)
    except Exception:  # noqa: BLE001 — the row itself is the problem being reported
        return repr(row)[:80]


def _attach_material(done: _Done) -> str:
    """What binds a downstream key to an upstream instance: its record key and what
    it produced (trace ids for rollouts; traces are referenced, not embedded)."""
    record = done.record
    if record.trace_id:
        return f"{record.key}:{record.trace_id}"
    if record.trace_ids:
        return f"{record.key}:{digest(record.trace_ids)}"
    return f"{record.key}:{digest(done.value)[:16]}"


def _payload(value: Any) -> Any:
    if isinstance(value, Trace) or (
        isinstance(value, list) and value and isinstance(value[0], Trace)
    ):
        return None  # traces live in traces.jsonl, referenced by id
    return to_jsonable_python(value)
