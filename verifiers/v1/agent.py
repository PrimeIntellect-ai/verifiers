"""The Agent: a reusable (harness x model x runtime) value with one executable arrow.

An `Agent` bundles the harness (the program), the model context (`agent.ctx`), and a
runtime policy (where a run's box comes from). `agent.run(task)` executes one
rollout and returns its `Trace`; `runtime=` borrows a live box, `provision(task)`
hands you one.

Interception follows the runtime story: inject a live `Interception` at
construction to share servers and tunnels across agents (a pool belongs to the
thing that spans agents — an env, a script — never to one agent); an entered
agent (`async with`) owns one interception server; un-entered, each run brings
up its own per-rollout server.
"""

import asyncio
import logging
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager, nullcontext
from typing import AsyncIterator

from verifiers.v1.clients import Client, ModelContext
from verifiers.v1.env import (
    TimeoutConfig,
    cap_remote_harness_timeout,
    resolve_runtime_config,
    validate_pairing,
)
from verifiers.v1.harness import Harness
from verifiers.v1.interception import Interception, InterceptionServer
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
    honored. A task `image` on a subprocess box raises (a wiring bug in the
    borrowing program — it goes to the caller, not the trace); a container box whose
    image differs only warns, since placing a run into an existing world is the
    point of borrowing."""
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

    Harnesses are stateless, so one instance can back any number of agents.
    `model`/`client`/`sampling` are the model context, bound at construction; agents on
    the same endpoint should share one `Client` (one connection pool).

    `runtime` is a *policy* (a `RuntimeConfig`, default the harness config's own):
    each `run` provisions a fresh box from it, resolved per task. Pass a live
    `Runtime` to `run(runtime=...)` to place a run into an existing box instead —
    borrowed boxes are never started or torn down by the run. `interception` is the
    same story for the model boundary: a live one is borrowed; without it, an
    entered agent owns one interception server (a server multiplexes concurrent
    runs — one agent never needs a pool) and an un-entered one brings up a per-run
    server."""

    def __init__(
        self,
        harness: Harness,
        model: str,
        client: Client,
        runtime: RuntimeConfig | None = None,
        *,
        sampling: Sampling | None = None,
        interception: Interception | None = None,
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
        self.limits = RolloutLimits() if limits is None else limits
        self.timeout = TimeoutConfig() if timeout is None else timeout
        # Env-owned standing, not config: `Environment.brief` marks fixed seats
        # (a frozen judge) untrainable and the role wrapper stamps traces from
        # here. Inert in bespoke scripts — nothing outside an env reads it.
        self.trainable: bool = True
        self._entered = False
        self._server: InterceptionServer | None = None
        self._warned_resources: set[tuple[str, str]] = set()

    async def __aenter__(self) -> "Agent":
        if self._entered:
            raise RuntimeError("Agent is already entered; enter it once and share it")
        self._entered = True
        if self.interception is None:
            # Sized to the runtime policy: a remote policy needs the tunnel. Runs the
            # server can't serve fall back per run (`_interception_for`).
            self._server = InterceptionServer(
                requires_tunnel=not runtime_is_local(self.runtime_config)
            )
            try:
                await self._server.__aenter__()
            except BaseException:
                # A failed __aenter__ gets no __aexit__ from `async with`: unwind
                # here, or the agent stays "already entered" forever.
                self._entered, self._server = False, None
                raise
        return self

    async def __aexit__(self, *exc) -> None:
        self._entered = False
        server, self._server = self._server, None
        if server is not None:
            await server.__aexit__(*exc)

    def _interception_for(
        self, run_is_local: bool, task: Task, shared_tools: Mapping
    ) -> Interception | None:
        """Which interception this run rides. An injected one always — its owner
        sized its reach. The owned server only when provably reachable from all of
        this run's consumers: always when it tunnels, else for a local run with no
        tool or user servers in play (any such server may sit in a remote runtime
        and must reach `/state`). Otherwise `None` — the rollout brings up a per-run
        server sized to the task."""
        if self.interception is not None:
            return self.interception
        if self._server is None:
            return None
        if self._server.tunnel is not None or (
            run_is_local
            and not shared_tools
            and not type(task).tools
            and type(task).user is None
        ):
            return self._server
        return None

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        shared_tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return the trace: the program runs on
        the task's prompt until it exits.

        The task carries its own judgement; a plain base `Task` makes the run
        unscored. `runtime` places the run into a live borrowed box instead of
        provisioning one from the agent's policy. `shared_tools` are live servers
        borrowed from their owner, counted in the pairing check. `on_trace` observes
        the run's trace the moment it's minted, before any I/O."""
        params = self._rollout_params(task, runtime, dict(shared_tools or {}))
        run = RolloutRun(task=task, on_trace=on_trace, **params)
        try:
            if await run.open():
                await run.step()
            trace = await run.close()
        except BaseException:
            # A cancellation mid-run (or a lifetime bug raised to the caller) means
            # close() never runs — free the run's servers and owned runtime first.
            await run.abort()
            raise
        self._stamp_agent(trace, params["runtime_config"], borrowed=runtime is not None)
        return trace

    def _rollout_params(
        self, task: Task, runtime: Runtime | None, shared_tools: dict
    ) -> dict:
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
            self.harness, type(task), runtime_config, shared_tools=shared_tools
        )
        # Timeout precedence: agent-level wins, else the task's, else no limit.
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
            shared_tools=shared_tools,
            interception=self._interception_for(run_is_local, task, shared_tools),
            runtime=runtime,
        )

    def _stamp_agent(
        self, trace: Trace, runtime_config: RuntimeConfig, *, borrowed: bool
    ) -> None:
        # Keeps traces attributable after the Agent objects are gone; a borrowed
        # box wins over the policy.
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
        """Provision (and on exit tear down) a box from this agent's runtime policy,
        resolved for `task` when given. Place runs into it via `run(..., runtime=box)`:
        the provisioning program owns the box, so several runs can share one world."""
        config = (
            resolve_runtime_config(self.runtime_config, task, self._warned_resources)
            if task is not None
            else self.runtime_config
        )
        runtime = make_runtime(config)
        try:
            # start() inside the try: a failed start may already hold a remote
            # sandbox, so it must reach stop() (safe on a partially-started runtime).
            await runtime.start()
            yield runtime
        finally:
            await runtime.stop()


class _EpisodeAgent(Agent):
    """One role's `Agent` for one env-rollout. An `Environment` builds these fresh
    per episode (an Agent is a cheap bundle of references — the expensive resources
    are env-owned and borrowed), so the episode's own capture lives right here with
    no state shared across concurrent episodes: every trace's `agent` info gets its
    episode standing (seat name, trainable, episode id, env id) written the moment
    it's created, finished traces land in `completed` (the episode's trace list),
    and each run takes the eval's concurrency gate. The taskset's shared tool servers ride only its own
    tasks — an env-minted task carries its own needs, and handing shared servers
    to its run would wrongly put MCP in play (pass `shared_tools=` explicitly to
    override either way)."""

    def __init__(
        self,
        harness: Harness,
        model: str,
        client: Client,
        *,
        sampling: Sampling | None,
        interception: Interception | None,
        limits: RolloutLimits,
        timeout: TimeoutConfig,
        name: str,
        role: str | None,
        episode: str,
        env: str,
        shared_tools: Mapping[str, SharedToolServer],
        task_cls: type[Task],
        gate: asyncio.Semaphore | None,
        completed: list[Trace],
        on_trace: Callable[[Trace], None] | None,
        warned_resources: set,
    ) -> None:
        super().__init__(
            harness,
            model,
            client,
            sampling=sampling,
            interception=interception,
            limits=limits,
            timeout=timeout,
        )
        # Resource warnings dedupe env-wide, not per episode.
        self._warned_resources = warned_resources
        self._name = name
        self._role = role
        self._episode = episode
        self._env = env
        self._shared_tools = shared_tools
        self._task_cls = task_cls
        self._gate = gate
        self._completed = completed
        self._on_trace = on_trace

    def _shared_for(self, task: Task) -> Mapping[str, SharedToolServer]:
        return self._shared_tools if isinstance(task, self._task_cls) else {}

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        shared_tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        def watch(trace: Trace) -> None:
            # The trace's episode standing, self-contained on its agent info.
            if trace.agent is not None:
                trace.agent.name = self._role
                trace.agent.trainable = self.trainable
                trace.agent.episode = self._episode
                trace.agent.env = self._env
            if self._on_trace is not None:
                self._on_trace(trace)
            if on_trace is not None:
                on_trace(trace)

        async with self._gate or nullcontext():
            trace = await super().run(
                task,
                runtime=runtime,
                shared_tools=shared_tools
                if shared_tools is not None
                else self._shared_for(task),
                on_trace=watch,
            )
        self._completed.append(trace)
        return trace
