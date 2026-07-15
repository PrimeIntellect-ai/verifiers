"""Explicit executable agents.

An `Agent` is the reusable execution primitive under evals and topologies: harness + model
context + runtime policy, with one arrow, `run(task) -> Trace`. A bare agent provisions a
runtime per run and brings up its own per-rollout interception server:

    agent = vf.Agent(DirectHarness(DirectHarnessConfig()), ctx)
    trace = await agent.run(MyTask(vf.TaskData(idx=0, prompt="...")))

Shared serving resources are injected, never owned: an eval or topology brings up one
`Interception` and passes it in (`interception=`), and taskset-scoped shared tool servers
arrive pre-served (`shared_tools=`). `TopologyRunner` is the config-driven packaging of
agents (CLI-addressable, graph-recorded, trainer-integrated); reach for a bare `Agent`
when scripting.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from verifiers.v1.clients import ModelContext
from verifiers.v1.env import (
    TimeoutConfig,
    resolve_runtime_config,
    resolve_stage_timeouts,
    validate_task_pairing,
)
from verifiers.v1.harness import Harness
from verifiers.v1.interception import Interception
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.retries import RolloutRetryConfig, run_with_retry
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import Runtime, RuntimeConfig, make_runtime
from verifiers.v1.session import RolloutLimits
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.utils.memory import trim_memory_periodically

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


Parent = Trace | str
"""An upstream lineage link: the parent trace itself, or just its id."""


def _parent_ids(parents: Sequence[Parent]) -> list[str]:
    return [parent.id if isinstance(parent, Trace) else parent for parent in parents]


class _AgentAttempt:
    """One retryable attempt (see `retries.Runnable`): a fresh `Rollout` each try, reusing
    the borrowed box when the agent didn't provision its own."""

    def __init__(
        self,
        agent: "Agent",
        task: Task,
        *,
        runtime_config: RuntimeConfig,
        runtime: Runtime | None,
        interception: Interception | None,
        parents: Sequence[Parent],
    ) -> None:
        self.agent = agent
        self.task = task
        self.runtime_config = runtime_config
        self.runtime = runtime
        self.interception = interception
        self.parents = parents

    async def run(self) -> Trace:
        agent = self.agent
        timeouts = resolve_stage_timeouts(agent.timeout, self.task, self.runtime_config)
        rollout = Rollout(
            task=self.task,
            harness=agent.harness,
            ctx=agent.ctx,
            runtime_config=self.runtime_config,
            setup_timeout=timeouts.setup,
            harness_timeout=timeouts.rollout,
            finalize_timeout=timeouts.finalize,
            scoring_timeout=timeouts.scoring,
            limits=agent.limits,
            shared_tools=agent.shared_tools,
            interception=self.interception,
            runtime=self.runtime,
        )
        if agent._on_rollout is not None:
            agent._on_rollout(rollout)
        trace = await rollout.run()
        agent.stamp(
            trace,
            parents=self.parents,
            runtime=rollout.runtime,
            borrowed=self.runtime is not None,
        )
        return trace


class Agent:
    """A harness + model context + runtime policy, runnable on any compatible task.

    `run(task)` provisions a runtime and tears it down. To share a world, `provision()`
    it and pass the box to `run(..., runtime=box)` calls; borrowed rollouts never start or
    stop it. `interception` and `shared_tools` are injected serving resources (module
    docstring). The framework-only knobs thread run-wide policy through every agent the
    topology runner builds — `retry` (whole-rollout retry), `semaphore` (the concurrency
    gate, held across retries), `on_trace` (graph recording), `on_rollout` (the live
    dashboard); a hand-built agent leaves them unset."""

    def __init__(
        self,
        harness: Harness,
        ctx: ModelContext,
        runtime: RuntimeConfig | None = None,
        *,
        name: str | None = None,
        seat: int | None = None,
        trainable: bool = True,
        limits: RolloutLimits | None = None,
        timeout: TimeoutConfig | None = None,
        interception: Interception | None = None,
        shared_tools: dict[str, SharedToolServer] | None = None,
        retry: RolloutRetryConfig | None = None,
        semaphore: asyncio.Semaphore | None = None,
        on_trace: Callable[[Trace], None] | None = None,
        on_rollout: Callable[[Rollout], None] | None = None,
    ) -> None:
        self.harness = harness
        self.ctx = ctx
        self.runtime_config = runtime if runtime is not None else harness.config.runtime
        self.name = name
        self.seat = seat
        """Index within a list role (None for a scalar agent)."""
        self.trainable = trainable
        self.limits = limits or RolloutLimits()
        self.timeout = timeout or TimeoutConfig()
        self.shared_tools = shared_tools or {}
        self._interception = interception
        self._retry = retry
        self._semaphore = semaphore
        self._on_trace = on_trace
        self._on_rollout = on_rollout
        self._warned_resources: set[tuple[str, str]] = set()
        self._validated: set[tuple[type[Task], str, str]] = set()

    def runtime_for(self, task: Task, runtime: Runtime | None = None) -> RuntimeConfig:
        """The runtime placement this run will actually use."""
        if runtime is not None:
            return runtime.config
        return resolve_runtime_config(self.runtime_config, task, self._warned_resources)

    def _validate_pairing(self, task: Task, runtime_config: RuntimeConfig) -> None:
        key = (type(task), runtime_config.type, runtime_config.__class__.__name__)
        if key not in self._validated:
            validate_task_pairing(
                self.harness,
                type(task),
                runtime_config,  # where the run actually lands — a borrowed box may differ
                shared_tools=tuple(self.shared_tools),
            )
            self._validated.add(key)

    async def run(
        self,
        task: Task,
        *,
        parents: Sequence[Parent] = (),
        runtime: Runtime | None = None,
    ) -> Trace:
        """Run once, returning the trace stamped with lineage (`parents`: the upstream
        traces this task was derived from) and provenance. `runtime` borrows a live box
        instead of provisioning one. A framework-built agent also applies its injected
        policy here: retry, the concurrency gate, and trace recording."""
        runtime_config = self.runtime_for(task, runtime)
        self._validate_pairing(task, runtime_config)
        attempt = _AgentAttempt(
            self,
            task,
            runtime_config=runtime_config,
            runtime=runtime,
            interception=self._interception,
            parents=parents,
        )
        async with self._semaphore or contextlib.nullcontext():
            if self._retry is not None:
                trace = await run_with_retry(attempt, self._retry)
            else:
                trace = await attempt.run()
        if self._on_trace is not None:
            self._on_trace(trace)
        await trim_memory_periodically()
        return trace

    def stamp(
        self,
        trace: Trace,
        *,
        parents: Sequence[Parent],
        runtime: Runtime | None,
        borrowed: bool,
    ) -> None:
        """Record agent provenance on a finished trace."""
        if self.name is not None:
            trace.agent = self.name
        trace.parents = _parent_ids(parents)
        trace.trainable = self.trainable
        trace.sampling = self.ctx.sampling
        trace.info["agent"] = {
            "name": self.name,
            "seat": self.seat,
            "harness": self.harness.config.id,
            "model": self.ctx.model,
            "runtime": {
                "type": runtime.config.type if runtime is not None else None,
                "descriptor": runtime.descriptor if runtime is not None else None,
                "borrowed": borrowed,
            },
        }

    @contextlib.asynccontextmanager
    async def provision(self, task: Task | None = None) -> AsyncIterator[Runtime]:
        """Provision a runtime from this agent's policy and tear it down on exit."""
        if task is None:
            config = self.runtime_config
        else:
            config = self.runtime_for(task)
            self._validate_pairing(task, config)
        runtime = make_runtime(config)
        try:
            await runtime.start()
            yield runtime
        finally:
            await runtime.stop()
