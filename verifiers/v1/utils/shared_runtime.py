"""Share one runtime across sequential agents."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from verifiers.v1.agent import Agent
from verifiers.v1.errors import HarnessError, boundary
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
                async with agent._gated(), boundary(HarnessError, "harness setup"):
                    await asyncio.wait_for(
                        agent.harness.setup(runtime),
                        agent.timeout.setup
                        if agent.timeout.setup is not None
                        else task.data.timeout.setup,
                    )
        yield runtime
