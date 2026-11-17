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
from verifiers.v1.utils.logging import clip, get_logger

logger = get_logger(__name__)

WORKSPACE_TAG = "vf-workspace"


def tool_vms(toolsets, workspace) -> dict[str, PrimeConfig]:
    """Prime tool VMs that join a Prime workspace over a rollout's private LAN."""
    from verifiers.v1.mcp.toolset import ToolsetConfig

    if not isinstance(workspace, PrimeConfig):
        return {}
    vms = {
        toolset.server_name: toolset.config.runtime
        for toolset in toolsets
        if isinstance(toolset.config, ToolsetConfig)
        and isinstance(toolset.config.runtime, PrimeConfig)
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
async def _lan_network(rules):
    """Own a LAN and its client until all of its VMs have been released."""
    async with leased_client() as client:
        network = await client.create_lan_network(
            f"vf-{uuid.uuid4().hex[:16]}", rules=rules
        )

        async def delete_network() -> None:
            try:
                await client.delete_lan_network(network.id)
            except Exception as exc:
                logger.warning(
                    "Failed to delete LAN network %s - %s: %s",
                    network.id,
                    type(exc).__name__,
                    clip(str(exc)),
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )

        try:
            yield client, network.id
        finally:
            await run_shielded(delete_network())


@asynccontextmanager
async def _tool_vm(
    config: PrimeConfig,
    network_id: str,
    tags: list[str],
    name: str | None = None,
) -> AsyncIterator[PrimeBackend]:
    """Provision a LAN member and keep it alive until its owner releases it."""
    async with AsyncExitStack() as stack:
        backend = make_backend(config, name)
        assert isinstance(backend, PrimeBackend)
        backend.lan = (network_id, tags)
        # A failed start may already hold a paid sandbox.
        stack.push_async_callback(backend.stop)
        await backend.start()
        # LAN traffic does not count as activity toward the platform's idle timeout.
        if config.idle_timeout is not None:
            every = max(10.0, config.idle_timeout / 3)

            async def stop_keepalive(task: asyncio.Task) -> None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

            stack.push_async_callback(
                stop_keepalive, asyncio.create_task(_keepalive(backend.bind(), every))
            )
        yield backend


@asynccontextmanager
async def shared_tool_network(
    config: PrimeConfig,
) -> AsyncIterator[tuple[RuntimeBackend, str]]:
    """Run one eval-wide tool VM on a LAN that Prime consumers may join."""
    from prime_sandboxes import LanRule

    rules = [LanRule(from_="tag:consumer", to="tag:tool")]
    async with (
        _lan_network(rules) as (_, network_id),
        _tool_vm(config, network_id, ["tool"]) as backend,
    ):
        yield backend, network_id


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
) -> AsyncIterator[tuple[dict[str, RuntimeBackend], Callable[[], Awaitable[None]]]]:
    """Join the workspace and service VMs to a rollout LAN, yielding services and
    a callback that ends the workspace's access before evidence collection.
    A missing `start_workspace` borrows an already running workspace."""
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
    rules = []
    if networks is not None:
        for index, names in enumerate(networks.values()):
            tag = f"vf-network-{index}"
            rules.append(LanRule(from_=f"tag:{tag}", to=f"tag:{tag}"))
            for name in names:
                memberships[name].append(tag)
    elif interconnect:
        for name, member_tags in memberships.items():
            if name not in tool_names:
                member_tags.append("vf-service")
        rules.append(LanRule(from_="tag:vf-service", to="tag:vf-service"))
    else:
        tool_names = frozenset(members)
    rules.extend(
        LanRule(from_=f"tag:{WORKSPACE_TAG}", to=f"tag:{tag}")
        for name, tag in tags.items()
        if name in tool_names
    )
    # Install connectivity before boot, and release the VMs before their LAN.
    async with (
        _lan_network(rules) as (client, network_id),
        AsyncExitStack() as stack,
    ):
        # Failed startup unwinds this stack before the deployment's workspace owner.
        if start_workspace is not None:
            stack.push_async_callback(workspace.stop)

        backends: dict[str, RuntimeBackend] = {}

        async def provision(name: str, config: PrimeConfig) -> None:
            backends[name] = await stack.enter_async_context(
                _tool_vm(
                    config,
                    network_id,
                    memberships[name],
                    name=f"{workspace.name}-{tags[name]}",
                )
            )

        async def stop_workspace() -> None:
            if start_workspace is not None:
                await workspace.stop()
            else:
                # Every rule on this LAN requires tags. Clear this membership's
                # tags and wait for the VM layer to apply them before collecting
                # service evidence; an API acknowledgement alone is not a barrier.
                await client._join_lan(network_id, workspace.info.id, [])

        if start_workspace is None:

            async def leave_workspace() -> None:
                await run_shielded(
                    client.remove_lan_member(network_id, workspace.info.id)
                )

            # Joining may attach the borrowed VM before failing or being cancelled.
            stack.push_async_callback(leave_workspace)
            await client._join_lan(network_id, workspace.info.id, memberships["main"])
        else:
            workspace.lan = (network_id, memberships["main"])
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
            network_id,
            workspace.info.id,
            {name: backend.info.id for name, backend in backends.items()},
        )
        yield backends, stop_workspace
