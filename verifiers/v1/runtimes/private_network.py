"""Machines of their own, joined to the workspace over the rollout's private network.

A machine's network policy covers the whole machine, so a node whose egress or
isolation differs from the workspace's, such as a tool server that calls public APIs
beside a workspace with none, or one that runs agent-submitted code with none, runs on
a machine of its own. Beside a workspace whose runtime declares
`SUPPORTS_PRIVATE_NETWORK`, as Prime's does, each rollout gets a network of its own on
which only the planned connections may be opened. A node is reached at its machine's
`network_host`.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

from verifiers.v1.runtimes import RuntimeBackend, RuntimeConfig, make_backend

logger = logging.getLogger(__name__)

WORKSPACE = "main"


@asynccontextmanager
async def private_network(
    workspace: RuntimeBackend,
    members: Mapping[str, RuntimeConfig],
    start_workspace: Callable[[], Awaitable[object]],
    rules: Collection[tuple[str, str]] | None = None,
) -> AsyncIterator[dict[str, RuntimeBackend]]:
    """Boot the workspace and one machine per member together, on a network of the
    rollout's own on which each `(source, target)` rule lets the source open
    connections to the target; by default, the workspace to every member."""
    backends = {
        name: make_backend(config, f"{workspace.name}-node-{index}")
        for index, (name, config) in enumerate(members.items())
    }
    if rules is None:
        rules = [(WORKSPACE, name) for name in backends]
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(
            type(workspace).private_network({WORKSPACE: workspace, **backends}, rules)
        )

        async def provision(backend: RuntimeBackend) -> None:
            # Pushed before `start()`: a failed start may already hold a paid machine.
            # Stopped before the network is removed.
            stack.push_async_callback(backend.stop)
            await backend.start()

        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(start_workspace())
                for backend in backends.values():
                    group.create_task(provision(backend))
        except ExceptionGroup as failed:
            raise failed.exceptions[0]  # one machine's failure, typed for retries
        logger.info(
            "private network joins %s to %s",
            workspace.network_host,
            {name: backend.network_host for name, backend in backends.items()},
        )
        yield backends
