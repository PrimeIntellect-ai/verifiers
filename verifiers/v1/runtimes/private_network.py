"""Tool servers on runtimes of their own, reached from the workspace over a private
network.

A machine's network policy covers the whole machine, so a tool server whose egress or
isolation differs from the workspace's, such as one that calls public APIs beside a
workspace with none, or one that runs agent-submitted code with none, runs on a runtime
of its own (`ToolsetConfig.runtime`). Beside a workspace whose runtime declares
`SUPPORTS_PRIVATE_NETWORK`, as Prime's does, each rollout gets a network of its own on
which the workspace may open connections to those servers' machines and nothing may
open one to the workspace. The harness reaches a server at its machine's
`network_host`.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

from verifiers.v1.runtimes import (
    RuntimeBackend,
    RuntimeConfig,
    backend_cls,
    make_backend,
)

logger = logging.getLogger(__name__)

WORKSPACE = "main"


def tool_vms(
    toolsets, workspace, *, composed: bool, borrowed: bool
) -> dict[str, RuntimeConfig]:
    """The task's tool servers that join the workspace over the rollout's private
    network, by server name: those on runtimes of their own, of the workspace's runtime,
    beside a workspace of the rollout's own that declares a private network and has no
    Compose services. Any other server keeps its runtime's published port, which takes
    no network policy."""
    if composed or borrowed or not backend_cls(workspace).SUPPORTS_PRIVATE_NETWORK:
        return {}
    vms = {
        toolset.server_name: toolset.config.runtime
        for toolset in toolsets
        if backend_cls(toolset.config.runtime) is backend_cls(workspace)
        and not toolset.config.colocated
        and not toolset.config.url
        and toolset.config.service is None
    }
    if WORKSPACE in vms:
        raise ValueError(
            f"a tool server on a machine of its own cannot be named {WORKSPACE!r}"
        )
    return vms


@asynccontextmanager
async def private_network(
    workspace: RuntimeBackend,
    members: Mapping[str, RuntimeConfig],
    start_workspace: Callable[[], Awaitable[object]],
) -> AsyncIterator[dict[str, RuntimeBackend]]:
    """Boot the workspace and one machine per member together, on a network of the
    rollout's own on which the workspace may open connections to every member."""
    backends = {
        name: make_backend(config, f"{workspace.name}-tool-{index}")
        for index, (name, config) in enumerate(members.items())
    }
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(
            type(workspace).private_network(
                {WORKSPACE: workspace, **backends},
                [(WORKSPACE, name) for name in backends],
            )
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
