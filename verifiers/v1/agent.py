"""The Agent: a reusable (harness x model x runtime) value with one executable arrow.

An `Agent` bundles WHO does the work — the harness (the program), the model leg
(model + client + sampling, grouped internally as the `ModelContext` every rollout
consumes), and a runtime policy (where a run's box comes from by default).
`agent.run(task)` executes one rollout and returns its `Trace`. Everything else is a
parameter, not a concept:

  - placement: `runtime=` borrows a live box (creator owns teardown) instead of
    provisioning a fresh one — put a judge into a solver's sandbox, or two agents into
    one world. `agent.provision(task)` hands you a box to place runs into.
  - judgement: the task carries it. A `Task` subclass's hooks (`setup`/`finalize`)
    and signals (`@reward`/`@metric`) run as in any eval; a plain base `Task` has
    no-op hooks and no signals, so the run is unscored — a pure `Task -> Trace` arrow.
  - chaining: plain functions. Mint the next task's `TaskData` from earlier traces
    (stamp `sources`/`relation` for lineage) and hand it to the next agent.

Interception follows the runtime story: a live `Interception` can be injected at
construction (borrowed — whoever entered it owns its lifecycle; that's how several
agents share one pool of servers and tunnels). Without one, entering the agent
(`async with`) owns an elastic pool sized to its runtime policy, so N concurrent runs
share interception servers like an eval does. Un-entered, each run brings up its own
per-rollout interception server — fine for scripts and small programs.

The execution machinery is the standard rollout engine (`RolloutRun`: staged lifecycle,
typed error attribution, token-true trace capture). The Agent only decides what goes into
the rollout.
"""

import logging
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from typing import AsyncIterator

from verifiers.v1.clients import Client, ModelContext
from verifiers.v1.env import (
    TimeoutConfig,
    cap_remote_harness_timeout,
    resolve_runtime_config,
    validate_pairing,
)
from verifiers.v1.harness import Harness
from verifiers.v1.interception import ElasticInterceptionPool, Interception
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.rollout import RolloutRun
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
    make_runtime,
    runtime_is_local,
)
from verifiers.v1.session import RolloutLimits
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import Sampling

logger = logging.getLogger(__name__)


def _check_borrowed_placement(task: Task, runtime: Runtime) -> None:
    """A borrowed box is never re-provisioned, so a task's placement fields can't be
    honored. Parity with the provisioning path where it refuses: a task `image` on a
    subprocess box raises (`resolve_runtime_config` raises the same on a subprocess
    policy — a lifetime/wiring bug in the borrowing program, so it goes to the caller,
    not the trace). A container box whose image differs only warns: placing a run into
    an existing world is the point of borrowing (e.g. a judge in a solver's box)."""
    if task.data.image is None:
        return
    if isinstance(runtime.config, SubprocessConfig):
        raise ValueError(
            f"task {task.data.idx!r} requires image {task.data.image!r}, but the "
            "borrowed runtime is subprocess-backed (no container); borrow a container "
            "box (e.g. agent.provision(task)) or drop the task's image"
        )
    box_image = getattr(runtime.config, "image", None)
    if box_image != task.data.image:
        logger.warning(
            "task %r requires image %r, but borrowed box %r runs %r; a borrowed box "
            "is never re-provisioned, so the run proceeds in the box's world",
            task.data.idx,
            task.data.image,
            runtime.name,
            box_image,
        )


class Agent:
    """A harness + model + runtime policy, runnable on any task.

    `harness` is a concrete `Harness` object (v1 construction is explicit), e.g.
    `DefaultHarness(DefaultHarnessConfig())`; harnesses are stateless, so one instance
    can back any number of agents. `load_harness(config)` resolves hub/local ids.

    `model`, `client`, and `sampling` are the model leg — an agent IS a model in a
    harness, bound at construction (they group into the `ModelContext` every rollout
    consumes, on `self.ctx`; `sampling` omitted means the provider's defaults). The
    client is yours to build (`resolve_client(EvalClientConfig())`) and to share:
    agents on the same endpoint should share one `Client` (one connection pool);
    prime-rl hands every agent its renderer client the same way.

    `runtime` here is a *policy* (a `RuntimeConfig`): each `run` provisions a fresh box
    from it, resolved per task (image / workdir / resources); it defaults to the harness
    config's own `runtime`. To place a run into an existing box instead, pass a live
    `Runtime` to `run(runtime=...)` — borrowed boxes are never started or torn down by
    the run; their creator owns their lifecycle.

    `interception` is the same story for the model boundary: a live, already-entered
    `Interception` to borrow — whoever entered it owns its lifecycle, this agent only
    acquires slots. Pass one pool to several agents so they share servers (and tunnels,
    behind remote runtimes). Without it, an entered agent owns an elastic pool; an
    un-entered agent's runs each bring up their own per-rollout server.

    `shared_tools` completes the borrowing set: live `SharedToolServer`s (taskset-scoped
    MCP, served once by their owner — an eval's `serving()`, or a program via
    `serve_shared`) that every run of this agent reuses. Borrowed like the others: never
    started or torn down here."""

    def __init__(
        self,
        harness: Harness,
        model: str,
        client: Client,
        runtime: RuntimeConfig | None = None,
        *,
        sampling: Sampling | None = None,
        interception: Interception | None = None,
        shared_tools: Mapping[str, SharedToolServer] | None = None,
        limits: RolloutLimits | None = None,
        timeout: TimeoutConfig | None = None,
    ) -> None:
        self.harness = harness
        self.ctx = ModelContext(
            model=model,
            client=client,
            sampling=sampling if sampling is not None else Sampling(),
        )
        self.runtime_config: RuntimeConfig = (
            runtime if runtime is not None else harness.config.runtime
        )
        self.interception = interception
        self.shared_tools = dict(shared_tools) if shared_tools else {}
        self.limits = RolloutLimits() if limits is None else limits
        self.timeout = TimeoutConfig() if timeout is None else timeout
        self._entered = False
        self._pool: ElasticInterceptionPool | None = None
        self._warned_resources: set[tuple[str, str]] = set()

    async def __aenter__(self) -> "Agent":
        if self._entered:
            raise RuntimeError("Agent is already entered; enter it once and share it")
        self._entered = True
        if self.interception is None:
            # Sized to the runtime policy: a remote policy needs the tunnel. Runs the
            # pool can't serve fall back per run (`_interception_for`).
            self._pool = ElasticInterceptionPool(
                requires_tunnel=not runtime_is_local(self.runtime_config)
            )
            await self._pool.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        self._entered = False
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.__aexit__(*exc)

    def _interception_for(self, run_is_local: bool, task: Task) -> Interception | None:
        """Which interception this run rides. An injected one always — its owner sized
        its reach over its consumers, like an eval injecting into every rollout. The
        owned pool only when provably reachable from all of this run's consumers: always
        when it tunnels (a tunnel URL works from anywhere), else for a local run whose
        task brings no tool servers and this agent no shared ones (any such server
        may sit in its own remote runtime and must reach `/state`). Otherwise `None` —
        the rollout brings up a per-run server sized to the task."""
        if self.interception is not None:
            return self.interception
        if self._pool is None:
            return None
        if self._pool.requires_tunnel or (
            run_is_local and not self.shared_tools and not type(task).tools
        ):
            return self._pool
        return None

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return the trace: the program runs on
        the task's prompt until it exits.

        The task carries its own judgement (its hooks + `@reward`/`@metric` run as in
        any eval); a plain base `Task` makes the run unscored. `runtime` places the run
        into a live box (borrowed — not started or torn down here) instead of
        provisioning a fresh one from the agent's runtime policy. `on_trace`
        observes the run's trace the moment it's minted (before any I/O) — how a
        caller watches the run live (the eval dashboard reads stage, tokens, and
        turns off it)."""
        params = self._rollout_params(task, runtime)
        run = RolloutRun(task=task, on_trace=on_trace, **params)
        if await run.open():
            await run.step()
        trace = await run.close()
        self._stamp_agent(trace, params["runtime_config"], borrowed=runtime is not None)
        return trace

    def _rollout_params(self, task: Task, runtime: Runtime | None) -> dict:
        """Resolve one run's execution parameters — runtime config, pairing checks,
        stage timeouts, interception."""
        if runtime is not None:
            _check_borrowed_placement(task, runtime)
            runtime_config = runtime.config
            run_is_local = runtime.is_local
        else:
            runtime_config = resolve_runtime_config(
                self.runtime_config, task, self._warned_resources
            )
            run_is_local = runtime_is_local(runtime_config)
        validate_pairing(
            self.harness, type(task), runtime_config, shared_tools=self.shared_tools
        )
        # Timeout precedence as in an eval's env-rollouts, with the agent standing in
        # for cli/toml: agent-level wins, else the task's, else no limit.
        harness_timeout = (
            self.timeout.rollout
            if self.timeout.rollout is not None
            else task.data.timeout.harness
        )
        return dict(
            harness=self.harness,
            ctx=self.ctx,
            runtime_config=runtime_config,
            setup_timeout=(
                self.timeout.setup
                if self.timeout.setup is not None
                else task.data.timeout.setup
            ),
            harness_timeout=cap_remote_harness_timeout(
                harness_timeout, runtime_config, task
            ),
            finalize_timeout=(
                self.timeout.finalize
                if self.timeout.finalize is not None
                else task.data.timeout.finalize
            ),
            scoring_timeout=(
                self.timeout.scoring
                if self.timeout.scoring is not None
                else task.data.timeout.scoring
            ),
            limits=self.limits,
            shared_tools=self.shared_tools,
            interception=self._interception_for(run_is_local, task),
            runtime=runtime,
        )

    def _stamp_agent(
        self, trace: Trace, runtime_config: RuntimeConfig, *, borrowed: bool
    ) -> None:
        # Who produced this trace — so a program's traces stay attributable after the
        # Agent objects are gone. Resolved per run: a borrowed box wins over the policy.
        trace.info["agent"] = {
            "harness": self.harness.config.id,
            "model": self.ctx.model,
            "runtime": {
                "type": runtime_config.type,
                "descriptor": trace.runtime.id if trace.runtime is not None else None,
                "borrowed": borrowed,
            },
        }

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
            # `RolloutRun.open`.
            await runtime.start()
            yield runtime
        finally:
            await runtime.stop()
