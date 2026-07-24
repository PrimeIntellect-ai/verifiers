"""Run multiple agents sequentially in one runtime."""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

from verifiers.v1.agent import Agent
from verifiers.v1.interception import InterceptionServer
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

__all__ = ["SharedRuntime", "shared_runtime"]


class SharedRuntime:
    """A provisioned box shared by a fixed sequence of agents."""

    def __init__(self, runtime: Runtime, agents: tuple[Agent, ...]) -> None:
        self.runtime = runtime
        self.agents = agents

    async def run(
        self,
        agent: Agent,
        task: Task,
        *,
        tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        """Run one listed agent without reopening a restricted runtime's setup phase."""
        if agent not in self.agents:
            raise ValueError(
                "shared runtime can only run agents declared at construction"
            )
        prepared = self.runtime.network_restricted and self.runtime.execution_prepared
        needs_shared_setup = self.runtime.network_restricted and not prepared
        trace = await agent.run(
            task,
            runtime=self.runtime,
            tools=tools,
            on_trace=on_trace,
            _runtime_prepared=prepared,
            _runtime_setup=(
                (lambda: self._setup_remaining(agent)) if needs_shared_setup else None
            ),
        )
        if self.runtime.network_restricted and not self.runtime.execution_prepared:
            raise RuntimeError(
                "the first shared-runtime rollout failed before its execution policy "
                "activated; refusing to continue with unrestricted networking"
            )
        return trace

    async def _setup_remaining(self, active: Agent) -> None:
        for agent in self.agents:
            if agent is active:
                continue
            await asyncio.wait_for(
                agent.harness.setup(self.runtime),
                agent.timeout.setup,
            )


@asynccontextmanager
async def shared_runtime(
    owner: Agent,
    task: Task,
    *agents: Agent,
) -> AsyncIterator[SharedRuntime]:
    """Provision one box for sequential agents.

    Restricted runtimes get one trusted setup phase: every listed harness is
    installed before the first rollout activates the network policy. All runs
    are pinned to one interception route, which remains reachable afterward.
    """
    participants = tuple(dict.fromkeys((owner, *agents)))
    async with owner.provision(task) as runtime:
        async with AsyncExitStack() as stack:
            originals = tuple(agent.interception for agent in participants)
            if runtime.network_restricted:
                interception = owner.interception
                if interception is None:
                    interception = await stack.enter_async_context(
                        InterceptionServer(requires_tunnel=not runtime.is_local)
                    )
                else:
                    interception = await stack.enter_async_context(
                        interception.reserve()
                    )
                for agent in participants:
                    agent.interception = interception
            try:
                yield SharedRuntime(runtime, participants)
            finally:
                for agent, interception in zip(participants, originals, strict=True):
                    agent.interception = interception
