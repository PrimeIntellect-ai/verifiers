"""The walker: one row at a time, ready nodes run under pools, every instance lands in
the ledger, sandbox state moves between nodes as git snapshots or a held live box."""

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
from verifiers.v1.flow.snapshot import GitBus, SnapshotRef
from verifiers.v1.runtimes import Runtime, RuntimeConfig, provision_runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

logger = logging.getLogger("verifiers.flow")


class RunResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    outcome: str
    files: dict[str, str] = {}


class EvalResult(BaseModel):
    episodes: int
    ok: int
    mean_reward: float | None
    output_dir: str


class Upstream:
    """What a node sees of the row: prior results by node name, their outcomes, the row, the config."""

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
    def __init__(
        self, flow: Flow, run_dir: Path, *, pools: Pools | None = None
    ) -> None:
        self.flow = flow
        self.graph: Graph = flow.graph
        self.config = flow.config
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(run_dir)
        self.pools = pools or Pools(self.config.pools)
        self.bus = GitBus(run_dir / "repo.git")
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
        self.boxes: dict[str, Runtime] = {}
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
            self._route(
                node, node.on_error, "error", done.record.error or "node failed"
            )
        elif node.outcomes is not None:
            target = (
                node.outcomes.get(done.outcome or "")
                or node.outcomes.get("*")
                or node.on_unmatched
            )
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
                outcome, value, extra = await self._run_kind(node, visit, up, key)
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
        self, node: Node, visit: int, up: Upstream, key: str
    ) -> tuple[str | None, Any, dict]:
        if isinstance(node, AgentNode):
            return await self._run_agent(node, visit, up)
        if isinstance(node, ExpandNode):
            return await self._run_expand(node, visit, up)
        if isinstance(node, RunNode):
            return await self._run_command(node, visit, up, key)
        if isinstance(node, FnNode):
            return await self._run_fn(node, up)
        if isinstance(node, EvalNode):
            return await self._run_eval(node, visit, up)
        raise FlowError([f"{node.name}: unknown node kind {node.kind}"])

    # -- kinds ----------------------------------------------------------------------

    async def _run_agent(
        self, node: AgentNode, visit: int, up: Upstream
    ) -> tuple[str | None, Trace, dict]:
        allowed = list(node.outcomes or {})
        allowed = [name for name in allowed if name != "*"]
        task = node.make_task(up)
        self._declare_outcomes(node, task, allowed)
        trace, snapshot, box_id = await self._rollout(node, task, visit=visit)
        if allowed:
            outcome, summary = outcome_of(trace, allowed)
        else:
            outcome, summary = "completed", ""
        return (
            outcome,
            trace,
            {
                "summary": summary,
                "trace_id": trace.id,
                "reward": trace.reward,
                "snapshot": snapshot,
                "box_id": box_id,
            },
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
                    and (t := self.e.ledger.trace(existing.trace_id or ""))
                ):
                    return i, t  # type: ignore[return-value]
                task = node.make_task(up, item)
                trace, _, box_id = await self._rollout(node, task, visit=visit, index=i)
                if not trace.ok:
                    raise FlowError(
                        [
                            f"{node.name}[{i}]: rollout failed: {[e.message for e in trace.errors]}"
                        ]
                    )
                self.e.ledger.put(
                    self._record(
                        node,
                        visit,
                        None,
                        [],
                        terminal="completed",
                        outcome="completed",
                        trace_id=trace.id,
                        reward=trace.reward,
                        box_id=box_id,
                    ).model_copy(update={"index": i})
                )
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
        self, node: RunNode, visit: int, up: Upstream, key: str
    ) -> tuple[str | None, RunResult, dict]:
        outcome_file = f"/tmp/flow-outcome-{key[:12]}.json"
        async with self.e.pools.hold(node.pools), AsyncExitStack() as local:
            box, workdir = await self._box(node, local, task=None)
            env = {**node.env, "FLOW_OUTCOME": outcome_file}
            argv = node.argv
            if workdir:
                argv = ["sh", "-c", 'cd "$0" && exec "$@"', workdir, *argv]
            result = await box.run(argv, env)
            outcome = await self._command_outcome(
                box, node, result.exit_code, outcome_file
            )
            files = {}
            for path in node.collect:
                try:
                    files[path] = (await box.read(path)).decode(errors="replace")
                except Exception as exc:  # noqa: BLE001 — a missing declared file is reported in the record
                    files[path] = f"<unreadable: {exc}>"
            if node.outcomes is None and result.exit_code != 0:
                raise FlowError(
                    [
                        f"{node.name}: exit {result.exit_code}: {result.stderr.strip()[-500:]}"
                    ]
                )
            snapshot = (
                await self._snapshot(node, box, workdir, visit)
                if node.snapshot
                else None
            )
            box_id = self._hold(node, box, local)
        value = RunResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            outcome=outcome,
            files=files,
        )
        return outcome, value, {"snapshot": snapshot, "box_id": box_id}

    async def _command_outcome(
        self, box: Runtime, node: RunNode, exit_code: int, outcome_file: str
    ) -> str:
        try:
            written = json.loads((await box.read(outcome_file)).decode())
        except Exception:  # noqa: BLE001 — no outcome file: the exit-code map decides
            written = None
        if isinstance(written, dict) and isinstance(written.get("outcome"), str):
            return written["outcome"]
        codes = node.exit_codes
        return (
            codes.get(exit_code)
            or codes.get(str(exit_code))
            or codes.get("*")
            or ("completed" if exit_code == 0 else "failed")
        )

    async def _run_fn(self, node: FnNode, up: Upstream) -> tuple[str | None, Any, dict]:
        value = node.func(up)
        if inspect.isawaitable(value):
            value = await value
        outcome = (
            value
            if node.outcomes is not None and isinstance(value, str)
            else "completed"
        )
        return outcome, value, {}

    async def _run_eval(
        self, node: EvalNode, visit: int, up: Upstream
    ) -> tuple[str | None, EvalResult, dict]:
        from verifiers.v1.cli.eval.runner import run_eval
        from verifiers.v1.configs.cli.eval import EvalConfig

        cfg = node.config(up) if callable(node.config) else node.config
        if isinstance(cfg, dict):
            cfg = EvalConfig.model_validate(cfg)
        cfg = cfg.model_copy(
            update={"output_dir": str(self.e.run_dir / "eval" / f"{node.name}@{visit}")}
        )
        async with self.e.pools.hold(node.pools):
            try:
                episodes = await run_eval(cfg)
            except Exception:  # infra failure is a routable outcome when declared
                if node.outcomes and "infra_failed" in node.outcomes:
                    return (
                        "infra_failed",
                        EvalResult(
                            episodes=0,
                            ok=0,
                            mean_reward=None,
                            output_dir=cfg.output_dir,
                        ),
                        {},
                    )
                raise
        rewards = [t.reward for ep in episodes for t in ep.traces if t.ok]
        value = EvalResult(
            episodes=len(episodes),
            ok=sum(ep.ok for ep in episodes),
            mean_reward=sum(rewards) / len(rewards) if rewards else None,
            output_dir=str(cfg.output_dir),
        )
        return "completed", value, {"reward": value.mean_reward}

    # -- boxes, rollouts, snapshots ---------------------------------------------------

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

    async def _rollout(
        self,
        node: AgentNode | ExpandNode,
        task: Task,
        *,
        visit: int,
        index: int | None = None,
    ) -> tuple[Trace, SnapshotRef | None, str | None]:
        agent = make_agent(self.e.seat(node.seat))
        held = node.name in self.g.held
        async with agent, self.e.pools.hold(node.pools), AsyncExitStack() as local:
            if node.runtime_ref is None and not node.snapshot and not held:
                trace = await agent.run(task)
                box, workdir = None, None
            else:
                box, workdir = await self._box(node, local, task=task, agent=agent)
                trace = await agent.run(task, runtime=box)
            if not trace.ok and index is None:
                raise FlowError(
                    [
                        f"{node.name}: rollout failed: {[e.message for e in trace.errors]}"
                    ]
                )
            snapshot = (
                await self._snapshot(node, box, workdir, visit, index)
                if node.snapshot and box
                else None
            )
            box_id = self._hold(node, box, local) if box else None
        await self.e.ledger.append(trace, env=self.g.name)
        return trace, snapshot, box_id

    async def _box(
        self, node: Node, local: AsyncExitStack, *, task: Task | None, agent: Any = None
    ) -> tuple[Runtime, str | None]:
        """The box a node runs in: a held live box, a fresh box restored from a
        snapshot, or a fresh box from the node's or seat's runtime config."""
        ref = node.runtime_ref
        if ref and ref[0] == "inherit":
            box = self.boxes.get(ref[1])
            if box is None:
                raise FlowError(
                    [
                        f"{node.name}: inherit:{ref[1]} but its box is not live (resume needs attach)"
                    ]
                )
            return box, self._workdir(node, box, task)
        stack = self.stack if node.name in self.g.held else local
        if agent is not None:
            box = await stack.enter_async_context(agent.provision(task))
        else:
            box = await stack.enter_async_context(
                provision_runtime(self._runtime_config(node))
            )
        workdir = self._workdir(node, box, task)
        if ref and ref[0] == "fork":
            snap = self.done[ref[1]].record.snapshot
            if snap is None:
                raise FlowError(
                    [f"{node.name}: fork:{ref[1]} but it recorded no snapshot"]
                )
            if workdir is None:
                raise FlowError(
                    [
                        f"{node.name}: fork needs a workdir (node.workdir or the runtime's)"
                    ]
                )
            await self.e.bus.restore(box, snap, workdir=workdir)
        return box, workdir

    def _runtime_config(self, node: Node) -> RuntimeConfig:
        if not isinstance(node.runtime, str):
            return node.runtime
        ref = node.runtime_ref
        if ref:
            return self._runtime_config(self.g.nodes[ref[1]])
        seat = getattr(node, "seat", None)
        if seat:
            return self.e.seat(seat).runtime
        raise FlowError([f"{node.name}: no runtime config to provision from"])

    def _workdir(self, node: Node, box: Runtime, task: Task | None) -> str | None:
        return (
            node.workdir
            or (task.data.workdir if task else None)
            or getattr(box.config, "workdir", None)
        )

    async def _snapshot(
        self,
        node: Node,
        box: Runtime,
        workdir: str | None,
        visit: int,
        index: int | None = None,
    ) -> SnapshotRef:
        if workdir is None:
            raise FlowError(
                [
                    f"{node.name}: snapshot needs a workdir (node.workdir or the runtime's)"
                ]
            )
        ref = f"refs/heads/flow/{self.key}/{node.name}@{visit}" + (
            f".{index}" if index is not None else ""
        )
        base = None
        if (
            (r := node.runtime_ref)
            and (prev := self.done.get(r[1]))
            and prev.record.snapshot
        ):
            base = prev.record.snapshot.head_sha
        return await self.e.bus.snapshot(box, workdir=workdir, ref=ref, base_sha=base)

    def _hold(self, node: Node, box: Runtime, local: AsyncExitStack) -> str | None:
        if node.name in self.g.held:
            self.boxes[node.name] = (
                box  # provisioned on the row stack; lives until the row ends
            )
        return getattr(box.info, "id", None)


def _payload(value: Any) -> Any:
    if isinstance(value, Trace) or (
        isinstance(value, list) and value and isinstance(value[0], Trace)
    ):
        return None  # traces live in traces.jsonl, referenced by id
    return to_jsonable_python(value)
