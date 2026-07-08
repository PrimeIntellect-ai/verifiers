"""Explicit executable agents.

An `Agent` is the reusable lower-level primitive under evals, topologies, and plain
agent programs: harness + model context + runtime policy, with one arrow,
`run(task) -> Trace`. Topologies add graph recording and deferred rewards around that
arrow; the agent itself only knows how to execute one task.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from verifiers.v1.clients import ModelContext
from verifiers.v1.env import (
    TimeoutConfig,
    resolve_runtime_config,
    resolve_stage_timeouts,
    validate_pairing,
)
from verifiers.v1.harness import Harness
from verifiers.v1.interception import InterceptionPool, RolloutLimits
from verifiers.v1.mcp import SharedServers
from verifiers.v1.retries import RolloutRetryConfig, run_with_retry
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import Runtime, RuntimeConfig, make_runtime
from verifiers.v1.services import RunServices
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


Parent = Trace | str


def _parent_ids(parents: Sequence[Parent]) -> list[str]:
    return [parent.id if isinstance(parent, Trace) else parent for parent in parents]


class _AgentAttempt:
    """One retryable attempt. Each `run()` builds a fresh `Rollout` when the agent owns
    runtime provisioning; borrowed-runtime attempts deliberately reuse the borrowed box."""

    def __init__(
        self,
        agent: "Agent",
        task: Task,
        *,
        ctx: ModelContext,
        runtime_config: RuntimeConfig,
        runtime: Runtime | None,
        shared: SharedServers | None,
        interception: InterceptionPool | None,
        parents: Sequence[Parent],
    ) -> None:
        self.agent = agent
        self.task = task
        self.ctx = ctx
        self.runtime_config = runtime_config
        self.runtime = runtime
        self.shared = shared
        self.interception = interception
        self.parents = parents

    async def run(self) -> Trace:
        rollout = self.agent.rollout(
            self.task,
            ctx=self.ctx,
            runtime_config=self.runtime_config,
            runtime=self.runtime,
            shared=self.shared,
            interception=self.interception,
        )
        trace = await rollout.run()
        self.agent.stamp(
            trace,
            parents=self.parents,
            runtime=rollout.runtime,
            ctx=self.ctx,
            borrowed=self.runtime is not None,
        )
        return trace


class Agent:
    """A harness + model context + runtime policy, runnable on any compatible task.

    `run(task)` provisions a fresh runtime from the policy and tears it down. To share a
    world explicitly, use `provision()` and pass the yielded runtime back to one or more
    `run(..., runtime=box)` calls; borrowed runtime rollouts never start or stop the box.
    """

    def __init__(
        self,
        harness: Harness,
        ctx: ModelContext,
        runtime: RuntimeConfig | None = None,
        *,
        name: str | None = None,
        trainable: bool = True,
        limits: RolloutLimits | None = None,
        timeout: TimeoutConfig | None = None,
        multiplex: int = 32,
        services: RunServices | None = None,
    ) -> None:
        self.harness = harness
        self.ctx = ctx
        self.runtime_config = runtime if runtime is not None else harness.config.runtime
        self.name = name
        self.trainable = trainable
        self.limits = limits or RolloutLimits()
        self.timeout = timeout or TimeoutConfig()
        self.multiplex = multiplex
        self._services = services
        self._owned_services: RunServices | None = None
        self._warned_resources: set[tuple[str, str]] = set()
        self._validated: set[tuple[type[Task], str, str]] = set()

    async def __aenter__(self) -> "Agent":
        if self._owned_services is not None:
            raise RuntimeError("Agent is already entered")
        if self._services is not None:
            raise RuntimeError("Agent already has external services; enter the owner")
        services = RunServices(self.multiplex)
        self._owned_services = await services.__aenter__()
        self._services = self._owned_services
        return self

    async def __aexit__(self, *exc) -> None:
        services = self._owned_services
        self._owned_services = None
        self._services = None
        if services is not None:
            await services.__aexit__(*exc)

    def runtime_for(self, task: Task, runtime: Runtime | None = None) -> RuntimeConfig:
        """The runtime placement this run will actually use."""
        if runtime is not None:
            return runtime.config
        return resolve_runtime_config(self.runtime_config, task, self._warned_resources)

    def _validate_pairing(self, task: Task, runtime_config: RuntimeConfig) -> None:
        key = (type(task), runtime_config.type, runtime_config.__class__.__name__)
        if key not in self._validated:
            validate_pairing(type(task), self.harness, runtime_config)
            self._validated.add(key)

    def rollout(
        self,
        task: Task,
        *,
        ctx: ModelContext | None = None,
        runtime_config: RuntimeConfig | None = None,
        runtime: Runtime | None = None,
        shared: SharedServers | None = None,
        interception: InterceptionPool | None = None,
    ) -> Rollout:
        """Build, but do not run, one rollout for `task`."""
        actual_runtime = runtime_config or self.runtime_for(task, runtime)
        self._validate_pairing(task, actual_runtime)
        timeouts = resolve_stage_timeouts(self.timeout, task, actual_runtime)
        return Rollout(
            task=task,
            harness=self.harness,
            ctx=ctx or self.ctx,
            runtime_config=actual_runtime,
            setup_timeout=timeouts.setup,
            harness_timeout=timeouts.rollout,
            finalize_timeout=timeouts.finalize,
            scoring_timeout=timeouts.scoring,
            limits=self.limits,
            shared=shared,
            interception=interception,
            runtime=runtime,
        )

    async def run(
        self,
        task: Task,
        *,
        parents: Sequence[Parent] = (),
        runtime: Runtime | None = None,
        ctx: ModelContext | None = None,
        services: RunServices | None = None,
        retry: RolloutRetryConfig | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return its trace."""
        ctx = ctx or self.ctx
        services = services if services is not None else self._services
        runtime_config = self.runtime_for(task, runtime)
        shared = services.shared if services is not None else None
        interception = (
            await services.pool_for(runtime_config) if services is not None else None
        )
        attempt = _AgentAttempt(
            self,
            task,
            ctx=ctx,
            runtime_config=runtime_config,
            runtime=runtime,
            shared=shared,
            interception=interception,
            parents=parents,
        )
        if retry is not None:
            return await run_with_retry(attempt, retry)
        return await attempt.run()

    def stamp(
        self,
        trace: Trace,
        *,
        parents: Sequence[Parent],
        runtime: Runtime | None,
        ctx: ModelContext,
        borrowed: bool,
    ) -> None:
        """Record agent provenance on a finished trace."""
        if self.name is not None:
            trace.agent = self.name
        trace.parents = _parent_ids(parents) or list(trace.task.sources)
        trace.trainable = self.trainable
        trace.info["agent"] = {
            "name": self.name,
            "harness": self.harness.config.id,
            "model": ctx.model,
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
