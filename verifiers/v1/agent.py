"""The Agent: a reusable (harness x model x runtime) value with one executable arrow.

An `Agent` bundles WHO does the work — the harness (the program), the model (a name on an
endpoint, bound at construction), and a runtime policy (where a run's box comes from by
default): `Agent("codex", "z-ai/glm-5.2", PrimeConfig())`. `agent.run(task)` executes one
rollout and returns its `Trace`. Everything else is a parameter, not a concept:

  - placement: `runtime=` borrows a live box (creator owns teardown) instead of
    provisioning a fresh one — put a judge into a solver's sandbox, or two agents into
    one world. `agent.provision(task)` hands you a box to place runs into.
  - judgement: `taskset=` attaches data-plane scoring (`@reward`/`@metric`, `setup`/
    `finalize`) to a run; omitted, the run is unscored — a pure `Task -> Trace` arrow.
  - chaining: plain functions. Mint the next `Task` from earlier traces (stamp
    `sources`/`relation` for lineage) and hand it to the next agent.

The Agent is an async context manager: entered, it owns an `InterceptionPool` so N
concurrent runs share interception servers (and tunnels, behind remote runtimes) like an
eval does. Un-entered, each run brings up its own per-rollout interception server — fine
for scripts and small programs.

The execution machinery is unchanged: every run is a standard `Rollout` (staged lifecycle,
typed error attribution, token-true trace capture). The Agent only decides what goes into
the five-tuple.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import AsyncIterator

from verifiers.v1.clients import (
    BaseClientConfig,
    EvalClientConfig,
    RolloutContext,
    resolve_client,
)
from verifiers.v1.env import TimeoutConfig, resolve_runtime_config
from verifiers.v1.harness import Harness
from verifiers.v1.interception import InterceptionPool, RolloutLimits
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    make_runtime,
    runtime_is_local,
)
from verifiers.v1.state import State
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.types import Sampling

logger = logging.getLogger(__name__)


def _merge(agent_timeout: float | None, task_timeout: float | None) -> float | None:
    """Agent-level timeout wins; else the task's; else no limit (`Environment.episode`'s
    precedence, with the agent standing in for cli/toml)."""
    return agent_timeout if agent_timeout is not None else task_timeout


class NullTaskset(Taskset[Task, TasksetConfig, State]):
    """No data, no judgement: the taskset behind an unscored `agent.run`. Every world hook
    is the inherited no-op; `load_tasks` is never called (the program supplies the task)."""


class Agent:
    """A harness + model + runtime policy, runnable on any task.

    `harness` is a concrete `Harness` object (v1 construction is explicit), e.g.
    `DefaultHarness(DefaultHarnessConfig())`; harnesses are stateless, so one instance
    can back any number of agents. `load_harness(config)` resolves hub/local ids.

    `model` is bound at construction — an agent IS a model in a harness. Pass the model
    name (served from the default Prime inference endpoint, or the endpoint in `client=`)
    or, for callers that manage their own `Client` (e.g. prime-rl's orchestrator, sharing
    one renderer client across agents), a prebuilt `RolloutContext`.

    `runtime` here is a *policy* (a `RuntimeConfig`): each `run` provisions a fresh box
    from it, resolved per task (image / workdir / resources); it defaults to the harness
    config's own `runtime`. To place a run into an existing box instead, pass a live
    `Runtime` to `run(runtime=...)` — borrowed boxes are never started or torn down by
    the run; their creator owns their lifecycle."""

    def __init__(
        self,
        harness: Harness,
        model: str | RolloutContext,
        runtime: RuntimeConfig | None = None,
        *,
        client: BaseClientConfig | None = None,
        sampling: Sampling | None = None,
        limits: RolloutLimits | None = None,
        timeout: TimeoutConfig | None = None,
        multiplex: int = 32,
    ) -> None:
        if isinstance(model, str):
            ctx = RolloutContext(
                model=model,
                client=resolve_client(client or EvalClientConfig()),
                sampling=sampling or Sampling(),
            )
        else:
            if client is not None or sampling is not None:
                raise ValueError(
                    "pass client=/sampling= only with a model name; a RolloutContext "
                    "already carries both"
                )
            ctx = model
        self.harness = harness
        self.ctx = ctx
        self.runtime_config: RuntimeConfig = (
            runtime if runtime is not None else harness.config.runtime
        )
        self.limits = limits or RolloutLimits()
        self.timeout = timeout or TimeoutConfig()
        self.multiplex = multiplex
        self._taskset = NullTaskset(TasksetConfig())
        self._pool: InterceptionPool | None = None
        self._warned_resources: set[tuple[str, str]] = set()

    @property
    def fingerprint(self) -> dict:
        """Who this agent is — stamped onto every trace it produces (`trace.info["agent"]`),
        so a program's traces stay attributable after the Agent objects are gone."""
        return {
            "harness": self.harness.config.id,
            "model": self.ctx.model,
            "runtime": self.runtime_config.type,
        }

    async def __aenter__(self) -> "Agent":
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
        taskset: Taskset | None = None,
        runtime: Runtime | None = None,
        model: str | None = None,
        sampling: Sampling | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return the trace.

        `taskset` attaches judgement (its world hooks + `@reward`/`@metric` run as in any
        eval); omitted, the run is unscored. `runtime` places the run into a live box
        (borrowed — not started or torn down here) instead of provisioning a fresh one
        from the agent's runtime policy. `model`/`sampling` override the agent's context
        for this run (e.g. a judge sweeping several models)."""
        ctx = self.ctx
        if model is not None:
            ctx = replace(ctx, model=model)
        if sampling is not None:
            ctx = replace(ctx, sampling=sampling)
        taskset = taskset if taskset is not None else self._taskset
        if runtime is not None:
            runtime_config = runtime.config
            run_is_local = type(runtime).is_local
        else:
            runtime_config = resolve_runtime_config(
                self.runtime_config, task, self._warned_resources
            )
            run_is_local = runtime_is_local(runtime_config)
        rollout = Rollout(
            task=task,
            taskset=taskset,
            harness=self.harness,
            ctx=ctx,
            runtime_config=runtime_config,
            setup_timeout=_merge(self.timeout.setup, task.timeout.setup),
            harness_timeout=_merge(self.timeout.rollout, task.timeout.harness),
            finalize_timeout=_merge(self.timeout.finalize, task.timeout.finalize),
            scoring_timeout=_merge(self.timeout.scoring, task.timeout.scoring),
            limits=self.limits,
            interception=self._interception_for(run_is_local),
            runtime=runtime,
        )
        trace = await rollout.run()
        trace.info["agent"] = {
            **self.fingerprint,
            "model": ctx.model,  # per-run override wins over the agent default
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
        await runtime.start()
        try:
            yield runtime
        finally:
            await runtime.stop()
