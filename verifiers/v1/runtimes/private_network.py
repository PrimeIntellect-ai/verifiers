"""Tool servers on Prime runtimes of their own, reached from the workspace over a LAN.

A Prime VM's egress policy covers the whole VM, so a tool server whose egress or
isolation differs from the workspace's, such as one that calls public APIs beside a
workspace with none, or one that runs agent-submitted code with none, runs on a Prime
runtime of its own (`ToolsetConfig.runtime`). Each rollout gets a LAN network on which
the workspace's VM may open connections to those VMs and nothing may open one to the
workspace. The harness reaches a server at its VM's LAN name,
`<sandbox-id>.sandbox.internal`, which the LAN's DNS keeps current.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

from verifiers.v1.runtimes import (
    PrimeBackend,
    PrimeConfig,
    RuntimeBackend,
    make_backend,
)
from verifiers.v1.runtimes.backends.prime import leased_client
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

WORKSPACE_TAG = "vf-workspace"


def tool_vms(
    toolsets, workspace, *, composed: bool, borrowed: bool
) -> dict[str, PrimeConfig]:
    """The task's tool servers that join the workspace over the rollout's private
    network, by server name: those on Prime runtimes of their own beside a Prime
    workspace of the rollout's own, without Compose services. Any other server keeps its
    runtime's published port, which takes no network policy."""
    if composed or borrowed or not isinstance(workspace, PrimeConfig):
        return {}
    vms = {
        toolset.server_name: toolset.config.runtime
        for toolset in toolsets
        if isinstance(toolset.config.runtime, PrimeConfig)
        and not toolset.config.colocated
        and not toolset.config.url
        and toolset.config.service is None
    }
    if "main" in vms:
        raise ValueError("a tool server on a VM of its own cannot be named 'main'")
    return vms


def lan_host(backend: RuntimeBackend) -> str:
    """The name a LAN member resolves a VM by."""
    return f"{backend.info.id}.sandbox.internal"


def _tag(index: int) -> str:
    # Server names need not be valid LAN tags; a member's position names it instead.
    return f"vf-tool-{index}"


async def _keepalive(runtime, every: float) -> None:
    while True:
        await asyncio.sleep(every)
        with contextlib.suppress(Exception):
            await runtime.run(["true"], {})


@asynccontextmanager
async def private_network(
    workspace: RuntimeBackend,
    members: Mapping[str, PrimeConfig],
    start_workspace: Callable[[], Awaitable[object]],
) -> AsyncIterator[dict[str, RuntimeBackend]]:
    """Boot the workspace's VM and one VM per member together, each joining a LAN
    network of the rollout's own as it boots."""
    if not isinstance(workspace, PrimeBackend):
        raise TypeError(
            "tool servers on Prime runtimes of their own join the workspace over a "
            f"private network, which needs the workspace on Prime, not "
            f"{workspace.config.type}"
        )
    from prime_sandboxes import LanRule

    tags = {name: _tag(index) for index, name in enumerate(members)}
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(leased_client())
        # The network starts with its rules: only the workspace may open connections,
        # to the servers' VMs. A member's join completes only once the network exists
        # on the platform, so its first members already join under these rules.
        network = await client.create_lan_network(
            f"vf-{uuid.uuid4().hex[:16]}",
            rules=[
                LanRule(from_=f"tag:{WORKSPACE_TAG}", to=f"tag:{tag}")
                for tag in tags.values()
            ],
        )

        async def delete_network() -> None:
            try:
                await client.delete_lan_network(network.id)
            except Exception:
                logger.warning(
                    "LAN network %s deletion failed", network.id, exc_info=True
                )

        stack.push_async_callback(lambda: run_shielded(delete_network()))

        backends: dict[str, RuntimeBackend] = {}

        async def stop(task: asyncio.Task) -> None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        async def provision(name: str, config: PrimeConfig) -> None:
            backend = make_backend(config, f"{workspace.name}-{tags[name]}")
            assert isinstance(backend, PrimeBackend)
            backend.lan = (network.id, [tags[name]])
            # Pushed before `start()`: a failed start may already hold a paid sandbox.
            stack.push_async_callback(backend.stop)
            await backend.start()  # running, and joined to the network
            backends[name] = backend
            # LAN traffic is not activity to the platform, and a server's VM sees no
            # commands of its own: keep it inside its idle timeout until it is released.
            if config.idle_timeout is not None:
                every = max(10.0, config.idle_timeout / 3)
                stack.push_async_callback(
                    stop, asyncio.create_task(_keepalive(backend.bind(), every))
                )

        workspace.lan = (network.id, [WORKSPACE_TAG])
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(start_workspace())
                for name, config in members.items():
                    group.create_task(provision(name, config))
        except ExceptionGroup as failed:
            raise failed.exceptions[0]  # one VM's failure, typed for retries
        logger.info(
            "LAN network %s joins %s to %s",
            network.id,
            workspace.info.id,
            {name: backend.info.id for name, backend in backends.items()},
        )
        yield backends
