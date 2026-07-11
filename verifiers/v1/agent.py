"""The Agent: a reusable (harness x model x runtime) value with one executable arrow.

An `Agent` bundles WHO does the work — the harness (the program), the model context
(model + client + sampling, the same `ModelContext` every rollout consumes), and
a runtime policy (where a run's box comes from by default). `agent.run(task)` executes one
rollout and returns its `Trace`. Everything else is a parameter, not a concept:

  - placement: `runtime=` borrows a live box (creator owns teardown) instead of
    provisioning a fresh one — put a judge into a solver's sandbox, or two agents into
    one world. `agent.provision(task)` hands you a box to place runs into.
  - judgement: the task carries it. A `Task` subclass's hooks (`setup`/`finalize`)
    and signals (`@reward`/`@metric`) run as in any eval; a plain base `Task` has
    no-op hooks and no signals, so the run is unscored — a pure `Task -> Trace` arrow.
  - chaining: plain functions. Mint the next task's `TaskData` from earlier traces
    (stamp `sources`/`relation` for lineage) and hand it to the next agent.

The Agent is an async context manager: entered, it owns an `InterceptionPool` so N
concurrent runs share interception servers (and tunnels, behind remote runtimes) like an
eval does. Un-entered, each run brings up its own per-rollout interception server — fine
for scripts and small programs.

The execution machinery is unchanged: every run is a standard `Rollout` (staged lifecycle,
typed error attribution, token-true trace capture). The Agent only decides what goes into
the rollout.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from verifiers.v1.clients import (
    ModelContext,
)
from verifiers.v1.env import (
    TimeoutConfig,
    cap_remote_harness_timeout,
    resolve_runtime_config,
    validate_pairing,
)
from verifiers.v1.harness import Harness
from verifiers.v1.interception import InterceptionPool, RolloutLimits
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    make_runtime,
    runtime_is_local,
)
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


def _merge(agent_timeout: float | None, task_timeout: float | None) -> float | None:
    """Agent-level timeout wins; else the task's; else no limit (`Environment.episode`'s
    precedence, with the agent standing in for cli/toml)."""
    return agent_timeout if agent_timeout is not None else task_timeout


class Agent:
    """A harness + model context + runtime policy, runnable on any task.

    `harness` is a concrete `Harness` object (v1 construction is explicit), e.g.
    `DefaultHarness(DefaultHarnessConfig())`; harnesses are stateless, so one instance
    can back any number of agents. `load_harness(config)` resolves hub/local ids.

    `ctx` is the `ModelContext` (model + client + sampling) — an agent IS a model in
    a harness, bound at construction. The client is
    yours to build (`resolve_client(EvalClientConfig())`) and to share: agents on the
    same endpoint should share one `Client` (one connection pool); prime-rl hands every
    agent its renderer client the same way.

    `runtime` here is a *policy* (a `RuntimeConfig`): each `run` provisions a fresh box
    from it, resolved per task (image / workdir / resources); it defaults to the harness
    config's own `runtime`. To place a run into an existing box instead, pass a live
    `Runtime` to `run(runtime=...)` — borrowed boxes are never started or torn down by
    the run; their creator owns their lifecycle."""

    def __init__(
        self,
        harness: Harness,
        ctx: ModelContext,
        runtime: RuntimeConfig | None = None,
        *,
        limits: RolloutLimits | None = None,
        timeout: TimeoutConfig | None = None,
        multiplex: int = 32,
    ) -> None:
        self.harness = harness
        self.ctx = ctx
        self.runtime_config: RuntimeConfig = (
            runtime if runtime is not None else harness.config.runtime
        )
        self.limits = limits or RolloutLimits()
        self.timeout = timeout or TimeoutConfig()
        self.multiplex = multiplex
        self._pool: InterceptionPool | None = None
        self._warned_resources: set[tuple[str, str]] = set()

    async def __aenter__(self) -> "Agent":
        if self._pool is not None:
            raise RuntimeError("Agent is already entered; enter it once and share it")
        self._pool = InterceptionPool(self.runtime_config, self.multiplex)
        await self._pool.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.__aexit__(*exc)

    def _interception_for(self, run_is_local: bool) -> InterceptionPool | None:
        """The shared pool, when its endpoint is reachable from this run's box: always for a
        local run (a tunnel URL works from anywhere, localhost works locally), and for a
        remote run only if the pool tunnels (was built for a remote runtime). Otherwise the
        rollout brings up its own per-run interception server."""
        if self._pool is None:
            return None
        if run_is_local or not self._pool.is_local:
            return self._pool
        return None

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        ctx: ModelContext | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return the trace.

        The task carries its own judgement (its hooks + `@reward`/`@metric` run as in
        any eval); a plain base `Task` makes the run unscored. `runtime` places the run
        into a live box (borrowed — not started or torn down here) instead of
        provisioning a fresh one from the agent's runtime policy. `ctx` replaces the
        agent's model context for this run — `dataclasses.replace(agent.ctx, model=...)`
        for a judge sweeping models."""
        ctx = ctx if ctx is not None else self.ctx
        if runtime is not None:
            runtime_config = runtime.config
            run_is_local = runtime.is_local
        else:
            runtime_config = resolve_runtime_config(
                self.runtime_config, task, self._warned_resources
            )
            run_is_local = runtime_is_local(runtime_config)
        validate_pairing(self.harness, type(task), runtime_config)
        rollout = Rollout(
            task=task,
            harness=self.harness,
            ctx=ctx,
            runtime_config=runtime_config,
            setup_timeout=_merge(self.timeout.setup, task.data.timeout.setup),
            harness_timeout=cap_remote_harness_timeout(
                _merge(self.timeout.rollout, task.data.timeout.harness),
                runtime_config,
                task,
            ),
            finalize_timeout=_merge(self.timeout.finalize, task.data.timeout.finalize),
            scoring_timeout=_merge(self.timeout.scoring, task.data.timeout.scoring),
            limits=self.limits,
            interception=self._interception_for(run_is_local),
            runtime=runtime,
        )
        trace = await rollout.run()
        # Who produced this trace — so a program's traces stay attributable after the
        # Agent objects are gone. Resolved per run: overrides and the live box win.
        trace.info["agent"] = {
            "harness": self.harness.config.id,
            "model": ctx.model,
            "runtime": {
                "type": runtime_config.type,
                "descriptor": rollout.runtime.descriptor if rollout.runtime else None,
                "borrowed": runtime is not None,
            },
        }
        return trace

    @asynccontextmanager
    async def provision(self, task: Task | None = None) -> AsyncIterator[Runtime]:
        """Provision (and on exit tear down) a box from this agent's runtime policy —
        resolved for `task` when given (image / workdir / resources). Place runs into it
        via `run(..., runtime=box)`: the program that provisions a box owns it, so several
        runs (by this or other agents) can share one world."""
        config = (
            resolve_runtime_config(self.runtime_config, task, self._warned_resources)
            if task is not None
            else self.runtime_config
        )
        runtime = make_runtime(config)
        try:
            # start inside the try: a failed start may already hold a remote sandbox,
            # so it must reach `stop()` (safe on a partially-started runtime) like in
            # `Rollout.run`.
            await runtime.start()
            yield runtime
        finally:
            await runtime.stop()
