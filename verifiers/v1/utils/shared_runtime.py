"""Share one runtime across sequential agents."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from verifiers.v1.agent import Agent
from verifiers.v1.interception import InterceptionServer
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task

__all__ = ["shared_runtime"]


@asynccontextmanager
async def shared_runtime(
    owner: Agent,
    task: Task,
    *agents: Agent,
) -> AsyncIterator[Runtime]:
    """Provision one box and prepare peer harnesses before its network policy."""
    participants = (owner, *agents)
    async with owner.provision(task) as runtime:
        if not runtime.network_restricted:
            yield runtime
            return

        reservation = (
            InterceptionServer(requires_tunnel=not runtime.is_local)
            if owner.interception is None
            else owner.interception.reserve()
        )
        async with reservation as interception:
            originals = tuple(agent.interception for agent in participants)
            try:
                for agent in participants:
                    agent.interception = interception
                for agent in agents:
                    await asyncio.wait_for(
                        agent.harness.setup(runtime),
                        agent.timeout.setup,
                    )
                yield runtime
            finally:
                for agent, original in zip(participants, originals, strict=True):
                    agent.interception = original
