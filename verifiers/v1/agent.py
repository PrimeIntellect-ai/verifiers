"""Explicit executable agents.

An `Agent` is the reusable execution primitive under evals, topologies, and plain agent
programs: harness + model context + runtime policy, with one arrow, `run(task) -> Trace`.
Topologies add graph recording and deferred rewards around that arrow; the agent itself
only knows how to execute one task.

A bare agent is fully standalone — each `run` provisions a runtime from the policy and
brings up its own per-rollout interception server, right for scripts and small programs:

    agent = vf.Agent(DirectHarness(DirectHarnessConfig()), ctx)
    trace = await agent.run(MyTask(vf.TaskData(idx=0, prompt="...")))

Serving resources have exactly one owner. For pooled operation (N concurrent runs sharing
interception servers, like an eval), bring up an `Interception` and inject it —
`async with make_interception(...) as interception: Agent(..., interception=interception)`
— which is precisely what `TopologyRunner.serving` does for every agent it binds. The
agent never owns the interception; it borrows it. Taskset-scoped shared tool servers
likewise arrive pre-served (`shared_tools=`, see `serve_shared`), from whoever owns the
taskset.

The framework's config-driven packaging of agents — CLI/toml-addressable, persisted as
agent graphs, eval- and trainer-integrated — is the topology (`verifiers.v1.topology`);
reach for a bare `Agent` when you're scripting.
"""

from __future__ import annotations

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

    `run(task)` provisions a fresh runtime from the policy and tears it down. To share a
    world explicitly, use `provision()` and pass the yielded runtime back to one or more
    `run(..., runtime=box)` calls; borrowed-runtime rollouts never start or stop the box.

    `interception` is the (optional) shared interception the agent's rollouts borrow —
    see the module docstring; without one, every run brings up its own per-rollout
    interception server. `shared_tools` are the taskset-scoped shared servers (already
    served, see `serve_shared`) this agent's rollouts reuse."""

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
        interception: Interception | None = None,
        shared_tools: dict[str, SharedToolServer] | None = None,
        on_rollout: Callable[[Rollout], None] | None = None,
    ) -> None:
        self.harness = harness
        self.ctx = ctx
        self.runtime_config = runtime if runtime is not None else harness.config.runtime
        self.name = name
        self.trainable = trainable
        self.limits = limits or RolloutLimits()
        self.timeout = timeout or TimeoutConfig()
        self.shared_tools = shared_tools or {}
        self._interception = interception
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
        retry: RolloutRetryConfig | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return its trace, stamped with lineage
        (`parents`) and this agent's provenance. `runtime` places the run into a live
        borrowed box instead of provisioning one; `retry` applies the whole-rollout
        retry policy."""
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
        trace.sampling = self.ctx.sampling
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
