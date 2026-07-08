"""Explicit executable agents.

An `Agent` is the reusable execution primitive under evals, topologies, and plain agent
programs: harness + model context + runtime policy, with one arrow, `run(task) -> Trace`.
Topologies add graph recording and deferred rewards around that arrow; the agent itself
only knows how to execute one task.

A bare agent is fully standalone — each `run` provisions a runtime from the policy and
brings up its own per-rollout interception server, right for scripts and small programs:

    agent = vf.Agent(DirectHarness(DirectHarnessConfig()), ctx)
    trace = await agent.run(vf.Task(idx=0, prompt="..."))

Serving resources have exactly one owner: `RunServices`. For pooled operation (N
concurrent runs sharing interception servers and shared MCP tools, like an eval), enter
a scope and inject it — `async with RunServices() as services:
Agent(..., services=services)` — which is precisely what `TopologyRunner.serving` does
for every agent it binds. The agent never owns services; it borrows them.

The framework's config-driven packaging of agents — CLI/toml-addressable, persisted as
agent graphs, eval- and trainer-integrated — is the topology (`verifiers.v1.topology`);
reach for a bare `Agent` when you're scripting.
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
"""An upstream lineage link: the parent trace itself, or just its id."""


def _parent_ids(parents: Sequence[Parent]) -> list[str]:
    return [parent.id if isinstance(parent, Trace) else parent for parent in parents]


class _AgentAttempt:
    """One retryable attempt (see `retries.Runnable`): each `run()` builds and runs a
    fresh `Rollout` when the agent owns runtime provisioning; borrowed-runtime attempts
    deliberately reuse the borrowed box."""

    def __init__(
        self,
        agent: "Agent",
        task: Task,
        *,
        runtime_config: RuntimeConfig,
        runtime: Runtime | None,
        shared: SharedServers | None,
        interception: InterceptionPool | None,
        parents: Sequence[Parent],
    ) -> None:
        self.agent = agent
        self.task = task
        self.runtime_config = runtime_config
        self.runtime = runtime
        self.shared = shared
        self.interception = interception
        self.parents = parents

    async def run(self) -> Trace:
        agent = self.agent
        rollout = Rollout(
            task=self.task,
            harness=agent.harness,
            ctx=agent.ctx,
            runtime_config=self.runtime_config,
            timeouts=resolve_stage_timeouts(
                agent.timeout, self.task, self.runtime_config
            ),
            limits=agent.limits,
            shared=self.shared,
            interception=self.interception,
            runtime=self.runtime,
        )
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

    `run(task)` provisions a fresh runtime from the policy and tears it down. To share a
    world explicitly, use `provision()` and pass the yielded runtime back to one or more
    `run(..., runtime=box)` calls; borrowed-runtime rollouts never start or stop the box.

    `services` is the (optional) serving scope the agent borrows pooled interception and
    shared MCP servers from — see the module docstring; without one, every run brings up
    its own per-rollout interception server.
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
        services: RunServices | None = None,
    ) -> None:
        self.harness = harness
        self.ctx = ctx
        self.runtime_config = runtime if runtime is not None else harness.config.runtime
        self.name = name
        self.trainable = trainable
        self.limits = limits or RolloutLimits()
        self.timeout = timeout or TimeoutConfig()
        self._services = services
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
            validate_pairing(type(task), self.harness, runtime_config)
            self._validated.add(key)

    async def run(
        self,
        task: Task,
        *,
        parents: Sequence[Parent] = (),
        runtime: Runtime | None = None,
        retry: RolloutRetryConfig | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return its trace, stamped with lineage
        (`parents`) and this agent's provenance. `runtime` places the run into a live
        borrowed box instead of provisioning one; `retry` applies the whole-rollout
        retry policy."""
        runtime_config = self.runtime_for(task, runtime)
        self._validate_pairing(task, runtime_config)
        services = self._services
        attempt = _AgentAttempt(
            self,
            task,
            runtime_config=runtime_config,
            runtime=runtime,
            shared=services.shared if services is not None else None,
            interception=await services.pool_for(runtime_config)
            if services is not None
            else None,
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
        borrowed: bool,
    ) -> None:
        """Record agent provenance on a finished trace."""
        if self.name is not None:
            trace.agent = self.name
        trace.parents = _parent_ids(parents)
        trace.trainable = self.trainable
        trace.info["agent"] = {
            "name": self.name,
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
