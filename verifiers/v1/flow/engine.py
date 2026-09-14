"""The walker: one row at a time, ready nodes run under pools, every node instance
lands in the ledger, and a node may run in the live runtime an earlier node used."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import typing
from collections.abc import Iterable
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
    EvalNode,
    ExpandNode,
    FnNode,
    Node,
    RunNode,
    Target,
    _End,
    _names,
)
from verifiers.v1.flow.outcome import FlowTaskConfig, outcome_of
from verifiers.v1.flow.pools import Pools
from verifiers.v1.runtimes import Runtime, RuntimeConfig, provision_runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

logger = logging.getLogger("verifiers.flow")


class RunResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    outcome: str


class EvalResult(BaseModel):
    episodes: int
    ok: int
    mean_reward: float | None
    output_dir: str


class Upstream:
    """What a node sees of its row: finished results by node name, their outcomes,
    the input row and the flow config."""

    def __init__(
        self,
        row: Any,
        config: Any,
        results: dict[str, Any],
        outcomes: dict[str, str | None],
    ) -> None:
        self.row = row
        self.config = config
        self._results = results
        self._outcomes = outcomes

    def __getattr__(self, name: str) -> Any:
        results = self.__dict__.get("_results") or {}
        if name in results:
            return results[name]
        raise AttributeError(
            f"no upstream result named {name!r}; have {sorted(results)}"
        )

    def outcome(self, name: str) -> str | None:
        return self._outcomes.get(name)


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
        # Any change to the graph shape or the config invalidates every ledger key.
        self.identity = digest(
            self.graph.structural_hash(), self.config.model_dump(mode="json")
        )[:16]
        (run_dir / "graph.json").write_text(
            json.dumps(self.graph.to_json(), indent=1, sort_keys=True)
        )
        (run_dir / "config.json").write_text(self.config.model_dump_json(indent=1))

    async def run(self, rows: Iterable[Any]) -> list[RowResult]:
        gate = asyncio.Semaphore(self.config.max_concurrent_rows)

        async def one(row: Any) -> RowResult:
            async with gate:
                return await _Row(self, row).run()

        return list(await asyncio.gather(*(one(row) for row in rows)))

    def seat(self, name: str) -> AgentConfig:
        cfg: AgentConfig = getattr(self.config, name)
        update = {}
        if cfg.model is None and self.config.model is not None:
            update["model"] = self.config.model
        if cfg.client is None and self.config.client is not None:
            update["client"] = self.config.client
        return cfg.model_copy(update=update) if update else cfg


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
        self.stack = AsyncExitStack()
        self.error: str | None = None

    # -- the loop -----------------------------------------------------------------

    async def run(self) -> RowResult:
        async with self.stack:
            self.pending.add(self.g.entry)
            while self.pending or self.running:
                self._start_ready()
                if not self.running:
                    logger.info(
                        "%s: %s can never fire; ending row",
                        self.key,
                        sorted(self.pending),
                    )
                    break
                finished, _ = await asyncio.wait(
                    self.running, return_when=asyncio.FIRST_COMPLETED
                )
                for task in finished:
                    name = self.running.pop(task)
                    self._complete(name, task.result())
                if self.error:
                    for task in self.running:
                        task.cancel()
                    await asyncio.gather(*self.running, return_exceptions=True)
                    break
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
        predecessor that no live node can still reach counts as dead, not awaited."""
        preds = self.g.preds.get(name, set())
        if not preds:
            return True
        fired = set(self.fired.get(name, {}))
        live = set(self.running.values()) | (self.pending - {name})
        reach: set[str] = set()
        for source in live:
            reach |= self.g.reachable(source)
        alive = {p for p in preds - fired if p in reach}
        join = self.g.nodes[name].join
        if join.kind == "any":
            return bool(fired)
        if join.kind == "at_least":
            return len(fired) >= join.k or (bool(fired) and not alive)
        return bool(fired) and not alive

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
        up = Upstream(
            self.row,
            self.e.config,
            {n: d.value for n, d in self.done.items()},
            {n: d.outcome for n, d in self.done.items()},
        )
        upstream_keys = sorted(self.done[p].record.key for p in fired if p in self.done)
        key = digest(self.e.identity, self.key, node.name, visit, upstream_keys)[:24]
        existing = self.e.ledger.get(self.key, node.name, visit)
        if existing and existing.key == key and existing.terminal == "completed":
            logger.info("%s: %s@%s attached to ledger", self.key, node.name, visit)
            return _Done(node.name, existing, self._rehydrate(node, existing))

        started = now()
        error: str | None = None
        for attempt in range(1, node.retries + 2):
            try:
                outcome, value, extra = await self._run_kind(node, visit, up)
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
            return [self.e.ledger.trace(tid) for tid in record.trace_ids]
        if isinstance(node, AgentNode):
            return self.e.ledger.trace(record.trace_id or "")
        if isinstance(node, RunNode):
            return RunResult.model_validate(record.payload)
        if isinstance(node, EvalNode):
            return EvalResult.model_validate(record.payload)
        if isinstance(node, FnNode):
            hint = typing.get_type_hints(node.func).get("return")
            if isinstance(hint, type) and issubclass(hint, BaseModel):
                return hint.model_validate(record.payload)
        return record.payload

    async def _run_kind(
        self, node: Node, visit: int, up: Upstream
    ) -> tuple[str | None, Any, dict]:
        if isinstance(node, AgentNode):
            return await self._run_agent(node, up)
        if isinstance(node, ExpandNode):
            return await self._run_expand(node, visit, up)
        if isinstance(node, RunNode):
            return await self._run_command(node, visit)
        if isinstance(node, FnNode):
            return await self._run_fn(node, up)
        if isinstance(node, EvalNode):
            return await self._run_eval(node, visit, up)
        raise FlowError([f"{node.name}: unknown node kind {node.kind}"])

    # -- kinds ----------------------------------------------------------------------

    async def _run_agent(
        self, node: AgentNode, up: Upstream
    ) -> tuple[str | None, Trace, dict]:
        allowed = [name for name in node.outcomes or {} if name != "*"]
        task = node.make_task(up)
        self._declare_outcomes(node, task, allowed)
        trace = await self._rollout(node, task)
        if not trace.ok:
            raise FlowError(
                [f"{node.name}: rollout failed: {[e.message for e in trace.errors]}"]
            )
        outcome, summary = outcome_of(trace, allowed) if allowed else ("completed", "")
        return (
            outcome,
            trace,
            {"summary": summary, "trace_id": trace.id, "reward": trace.reward},
        )

    async def _run_expand(
        self, node: ExpandNode, visit: int, up: Upstream
    ) -> tuple[str | None, list[Trace], dict]:
        items = list(node.over(up))
        gate = asyncio.Semaphore(node.max_active or max(len(items), 1))
        need = {"all": len(items), "any": 1, "at_least": node.join.k}[node.join.kind]

        async def one(i: int, item: Any) -> tuple[int, Trace]:
            async with gate:
                existing = self.e.ledger.get(self.key, node.name, visit, i)
                if (
                    existing
                    and existing.terminal == "completed"
                    and (trace := self.e.ledger.trace(existing.trace_id or ""))
                ):
                    return i, trace  # type: ignore[return-value]
                trace = await self._rollout(node, node.make_task(up, item))
                if not trace.ok:
                    raise FlowError([f"{node.name}[{i}]: rollout failed"])
                record = self._record(
                    node,
                    visit,
                    None,
                    [],
                    terminal="completed",
                    outcome="completed",
                    trace_id=trace.id,
                    reward=trace.reward,
                )
                self.e.ledger.put(record.model_copy(update={"index": i}))
                return i, trace

        tasks = [asyncio.create_task(one(i, item)) for i, item in enumerate(items)]
        got: dict[int, Trace] = {}
        failures: list[str] = []
        try:
            for fut in asyncio.as_completed(tasks):
                try:
                    i, trace = await fut
                    got[i] = trace
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
        traces = [got[i] for i in sorted(got)]
        return "completed", traces, {"trace_ids": [t.id for t in traces]}

    async def _run_command(
        self, node: RunNode, visit: int
    ) -> tuple[str | None, RunResult, dict]:
        outcome_file = f"/tmp/flow-outcome-{self.key}-{node.name}-{visit}.json"
        async with self.e.pools.hold(node.pools), AsyncExitStack() as local:
            runtime = await self._runtime(node, local)
            result = await runtime.run(
                node.argv, {**node.env, "FLOW_OUTCOME": outcome_file}
            )
            outcome = await self._command_outcome(
                runtime, node, result.exit_code, outcome_file
            )
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

    async def _command_outcome(
        self, runtime: Runtime, node: RunNode, exit_code: int, outcome_file: str
    ) -> str:
        try:
            written = json.loads((await runtime.read(outcome_file)).decode())
        except Exception:  # noqa: BLE001 — no outcome file: the exit-code map decides
            written = None
        if isinstance(written, dict) and isinstance(written.get("outcome"), str):
            return written["outcome"]
        codes = node.exit_codes
        fallback = "completed" if exit_code == 0 else "failed"
        return (
            codes.get(exit_code)
            or codes.get(str(exit_code))
            or codes.get("*")
            or fallback
        )

    async def _run_fn(self, node: FnNode, up: Upstream) -> tuple[str | None, Any, dict]:
        value = node.func(up)
        if inspect.isawaitable(value):
            value = await value
        routes = node.outcomes is not None and isinstance(value, str)
        return (value if routes else "completed"), value, {}

    async def _run_eval(
        self, node: EvalNode, visit: int, up: Upstream
    ) -> tuple[str | None, EvalResult, dict]:
        from verifiers.v1.cli.eval.runner import run_eval
        from verifiers.v1.configs.cli.eval import EvalConfig

        cfg = node.config(up) if callable(node.config) else node.config
        if isinstance(cfg, dict):
            cfg = EvalConfig.model_validate(cfg)
        output_dir = str(self.e.run_dir / "eval" / f"{node.name}@{visit}")
        cfg = cfg.model_copy(update={"output_dir": output_dir})
        async with self.e.pools.hold(node.pools):
            try:
                episodes = await run_eval(cfg)
            except Exception:  # infra failure is a routable outcome when declared
                if node.outcomes and "infra_failed" in node.outcomes:
                    empty = EvalResult(
                        episodes=0, ok=0, mean_reward=None, output_dir=output_dir
                    )
                    return "infra_failed", empty, {}
                raise
        rewards = [t.reward for ep in episodes for t in ep.traces if t.ok]
        value = EvalResult(
            episodes=len(episodes),
            ok=sum(ep.ok for ep in episodes),
            mean_reward=sum(rewards) / len(rewards) if rewards else None,
            output_dir=output_dir,
        )
        return "completed", value, {"reward": value.mean_reward}

    # -- runtimes -------------------------------------------------------------------

    def _declare_outcomes(self, node: Node, task: Task, allowed: list[str]) -> None:
        if not allowed:
            return
        if not isinstance(task.config, FlowTaskConfig):
            raise FlowError(
                [
                    f"{node.name}: outcomes need a FlowTask (its config carries the outcome tool)"
                ]
            )
        task.config = task.config.model_copy(update={"outcomes": allowed})

    async def _rollout(self, node: AgentNode | ExpandNode, task: Task) -> Trace:
        """Run the seat on `task`, in its own runtime unless the node inherits one
        or a later node inherits this one's."""
        agent = make_agent(self.e.seat(node.seat))
        async with agent, self.e.pools.hold(node.pools), AsyncExitStack() as local:
            if node.inherits is None and node.name not in self.g.held:
                trace = await agent.run(task)
            else:
                runtime = await self._runtime(node, local, agent=agent, task=task)
                trace = await agent.run(task, runtime=runtime)
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
                raise FlowError(
                    [
                        f"{node.name}: inherit:{node.inherits} but its runtime is not live"
                    ]
                )
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


def _payload(value: Any) -> Any:
    if isinstance(value, Trace) or (
        isinstance(value, list) and value and isinstance(value[0], Trace)
    ):
        return None  # traces live in traces.jsonl, referenced by id
    return to_jsonable_python(value)
