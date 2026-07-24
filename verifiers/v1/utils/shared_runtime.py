"""Share one runtime across sequential agents."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from verifiers.v1.agent import Agent
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
    async with owner.provision(task) as runtime:
        if runtime.network_restricted:
            for agent in agents:
                await asyncio.wait_for(
                    agent.harness.setup(runtime),
                    agent.timeout.setup,
                )
        yield runtime
