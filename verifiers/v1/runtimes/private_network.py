"""Prime task services and tool servers, reached over a rollout's private LAN.

A Prime VM's egress policy covers the whole VM, so a tool server whose egress or
isolation differs from the workspace's, such as one that calls public APIs beside a
workspace with none, or one that runs agent-submitted code with none, runs on a Prime
runtime of its own (`ToolsetConfig.runtime`). Independent tool VMs accept connections
only from the workspace; task-declared services can also call each other and the
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
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)

WORKSPACE_TAG = "vf-workspace"


def tool_vms(toolsets, workspace) -> dict[str, PrimeConfig]:
    """Prime tool VMs that join a Prime workspace over a rollout's private LAN."""
    if not isinstance(workspace, PrimeConfig):
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
async def shared_tool_network(
    config: PrimeConfig,
) -> AsyncIterator[tuple[RuntimeBackend, str]]:
    """Run one eval-wide tool VM on a LAN that Prime consumers may join."""
    from prime_sandboxes import LanRule

    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(leased_client())
        network = await client.create_lan_network(
            f"vf-shared-{uuid.uuid4().hex[:16]}",
            rules=[LanRule(from_="tag:consumer", to="tag:tool")],
        )

        async def delete_network() -> None:
            try:
                await client.delete_lan_network(network.id)
            except Exception:
                logger.warning(
                    "Failed to delete shared tool LAN network %s",
                    network.id,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )

        stack.push_async_callback(lambda: run_shielded(delete_network()))
        backend = make_backend(config)
        assert isinstance(backend, PrimeBackend)
        backend.lan = (network.id, ["tool"])
        stack.push_async_callback(backend.stop)
        await backend.start()
        if config.idle_timeout is not None:
            every = max(10.0, config.idle_timeout / 3)

            async def stop_keepalive(task: asyncio.Task) -> None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

            stack.push_async_callback(
                stop_keepalive, asyncio.create_task(_keepalive(backend.bind(), every))
            )
        yield backend, network.id


async def join_shared_tool_network(network_id: str, consumer: RuntimeBackend) -> None:
    """Give a Prime consumer private access to an eval-wide tool VM."""
    if not isinstance(consumer, PrimeBackend):
        raise TypeError("a shared Prime tool server requires a Prime consumer")
    async with leased_client() as client:
        await client._join_lan(network_id, consumer.info.id, ["consumer"])


@asynccontextmanager
async def private_network(
    workspace: RuntimeBackend,
    members: Mapping[str, PrimeConfig],
    start_workspace: Callable[[], Awaitable[object]] | None,
    *,
    interconnect: bool = False,
    networks: Mapping[str, list[str]] | None = None,
    tool_names: frozenset[str] = frozenset(),
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
    memberships = {
        "main": [WORKSPACE_TAG],
        **{name: [tag] for name, tag in tags.items()},
    }
    if interconnect and tool_names:
        for name, member_tags in memberships.items():
            if name not in tool_names:
                member_tags.append("vf-service")
    rules = (
        [LanRule(from_="tag:vf-service", to="tag:vf-service")]
        if interconnect and tool_names
        else [LanRule(from_="*", to="*")]
        if interconnect
        else [
            LanRule(from_=f"tag:{WORKSPACE_TAG}", to=f"tag:{tag}")
            for tag in tags.values()
        ]
    )
    if interconnect and tool_names:
        rules.extend(
            LanRule(from_=f"tag:{WORKSPACE_TAG}", to=f"tag:{tags[name]}")
            for name in tool_names
        )
    if networks is not None:
        rules = []
        for index, names in enumerate(networks.values()):
            tag = f"vf-network-{index}"
            rules.append(LanRule(from_=f"tag:{tag}", to=f"tag:{tag}"))
            for name in names:
                memberships[name].append(tag)
        rules.extend(
            LanRule(from_=f"tag:{WORKSPACE_TAG}", to=f"tag:{tags[name]}")
            for name in tool_names
        )
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(leased_client())
        # Install connectivity before boot: Compose peers must share a network;
        # independent tool servers accept only workspace connections.
        network = await client.create_lan_network(
            f"vf-{uuid.uuid4().hex[:16]}",
            rules=rules,
        )

        async def delete_network() -> None:
            try:
                await client.delete_lan_network(network.id)
            except Exception:
                logger.warning(
                    "LAN network %s deletion failed", network.id, exc_info=True
                )

        stack.push_async_callback(lambda: run_shielded(delete_network()))
        # Failed startup unwinds this stack before the deployment's workspace owner.
        if start_workspace is not None:
            stack.push_async_callback(workspace.stop)

        backends: dict[str, RuntimeBackend] = {}

        async def stop(task: asyncio.Task) -> None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        async def provision(name: str, config: PrimeConfig) -> None:
            backend = make_backend(config, f"{workspace.name}-{tags[name]}")
            assert isinstance(backend, PrimeBackend)
            backend.lan = (network.id, memberships[name])
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

        if start_workspace is None:
            # Borrowed Prime workspaces already run; joining is idempotent.
            await client._join_lan(network.id, workspace.info.id, memberships["main"])
        else:
            workspace.lan = (network.id, memberships["main"])
        try:
            async with asyncio.TaskGroup() as group:
                if start_workspace is not None:
                    group.create_task(start_workspace())
                for name, config in members.items():
                    group.create_task(provision(name, config))
        except ExceptionGroup as failed:
            raise failed.exceptions[0]  # one VM's failure, typed for retries
        logger.debug(
            "Joined LAN network %s from %s to %s",
            network.id,
            workspace.info.id,
            {name: backend.info.id for name, backend in backends.items()},
        )
        yield backends
