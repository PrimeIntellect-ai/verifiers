"""Prime task services and tool servers, reached over a rollout's private LAN.

A Prime VM's egress policy covers the whole VM, so a tool server whose egress or
isolation differs from the workspace's, such as one that calls public APIs beside a
workspace with none, or one that runs agent-submitted code with none, runs on a Prime
runtime of its own (`ToolsetConfig.runtime`). Independent tool VMs accept connections
from the workspace and harness. Task-declared services follow their network rules. The harness reaches a server at its VM's LAN name,
`<sandbox-id>.sandbox.internal`, which the LAN's DNS keeps current.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Collection, Mapping
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
HARNESS_TAG = "vf-harness"
TASK_TAG = "vf-task"


def tool_vms(
    toolsets, workspace, *, composed: bool, borrowed: bool
) -> dict[str, PrimeConfig]:
    """The task's tool servers that join the workspace over the rollout's private
    network, by server name: those on Prime runtimes of their own beside a Prime
    workspace of the rollout's own, without a task service spec. Any other server keeps its
    runtime's published port, which takes no network policy."""
    from verifiers.v1.mcp.toolset import ToolsetConfig

    if composed or borrowed or not isinstance(workspace, PrimeConfig):
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


@asynccontextmanager
async def keepalive(backend: PrimeBackend):
    """Keep a VM active while its tools or servers see only relay traffic."""
    idle = backend.config.idle_timeout
    if idle is None:
        yield
        return

    async def pulse() -> None:
        while not backend.stopped:
            await asyncio.sleep(max(10.0, idle / 3))
            if not backend.stopped:
                with contextlib.suppress(Exception):
                    await backend.bind().run(["true"], {})

    task = asyncio.create_task(pulse())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@asynccontextmanager
async def private_network(
    workspace: RuntimeBackend,
    members: Mapping[str, PrimeConfig],
    *,
    interconnect: bool = False,
    networks: Mapping[str, list[str]] | None = None,
    harness: PrimeBackend | None = None,
    harness_peers: Collection[str] = (),
) -> AsyncIterator[dict[str, RuntimeBackend]]:
    """Own one rollout's workspace, services and optional harness on a private LAN."""
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
    if networks is not None:
        rules = []
        for index, names in enumerate(networks.values()):
            tag = f"vf-network-{index}"
            rules.append(LanRule(from_=f"tag:{tag}", to=f"tag:{tag}"))
            for name in names:
                memberships[name].append(tag)
    elif interconnect:
        # Task services and the workspace, not the harness, reach each other.
        rules = [LanRule(from_=f"tag:{TASK_TAG}", to=f"tag:{TASK_TAG}")]
        for member_tags in memberships.values():
            member_tags.append(TASK_TAG)
    else:
        rules = [
            LanRule(from_=f"tag:{WORKSPACE_TAG}", to=f"tag:{tag}")
            for tag in tags.values()
        ]
    if harness is not None:
        # MCP servers are the harness's only direct LAN consumers. Model tools reach
        # the workspace through the host relay.
        destinations = (
            tags[name] for name in tags if not interconnect or name in harness_peers
        )
        rules.extend(
            LanRule(from_=f"tag:{HARNESS_TAG}", to=f"tag:{tag}") for tag in destinations
        )
        if "main" in harness_peers:
            rules.append(LanRule(from_=f"tag:{HARNESS_TAG}", to=f"tag:{WORKSPACE_TAG}"))
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(leased_client())
        name = f"vf-{uuid.uuid4().hex[:16]}"

        async def delete_network(network_id: str) -> None:
            try:
                await client.delete_lan_network(network_id)
            except Exception:
                logger.warning(
                    "Failed to delete private LAN %s", network_id, exc_info=True
                )

        async def create_network() -> str:
            network = await client.create_lan_network(name, rules=rules)
            stack.push_async_callback(delete_network, network.id)
            return network.id

        network_id = await run_shielded(create_network())
        backends = {
            name: make_backend(cfg, f"{workspace.name}-{tags[name]}")
            for name, cfg in members.items()
        }
        targets = [
            (workspace, memberships["main"]),
            *((vm, memberships[name]) for name, vm in backends.items()),
        ]
        if harness is not None:
            targets.append((harness, [HARNESS_TAG]))

        async def stop_vms() -> None:
            await asyncio.gather(
                *(vm.stop() for vm, _ in targets), return_exceptions=True
            )

        # Own every VM before startup so a cancelled or failed create still unwinds.
        stack.push_async_callback(stop_vms)
        try:
            async with asyncio.TaskGroup() as group:
                for backend, member_tags in targets:
                    assert isinstance(backend, PrimeBackend)
                    backend.lan = (network_id, member_tags)
                    group.create_task(backend.start())
        except ExceptionGroup as failed:
            raise failed.exceptions[0]
        for backend in backends.values():
            await stack.enter_async_context(keepalive(backend))
        if harness is not None:
            await stack.enter_async_context(keepalive(workspace))
            await stack.enter_async_context(keepalive(harness))
        logger.debug("Joined private LAN %s with %d VMs", network_id, len(targets))
        yield backends
