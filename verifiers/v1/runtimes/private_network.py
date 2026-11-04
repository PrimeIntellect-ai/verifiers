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

logger = logging.getLogger(__name__)

WORKSPACE_TAG = "vf-workspace"


def tool_vms(toolsets, workspace) -> dict[str, PrimeConfig]:
    """The task's tool servers that join the workspace over the rollout's private
    network, by server name. Shared servers are owned by the serving context."""
    if not isinstance(workspace, PrimeConfig):
        return {}
    vms = {
        toolset.server_name: toolset.config.runtime
        for toolset in toolsets
        if isinstance(toolset.config.runtime, PrimeConfig)
        and not toolset.config.shared
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
async def keepalive(backend: PrimeBackend):
    """LAN requests do not refresh a sandbox's platform idle timeout."""
    task = None
    if backend.config.idle_timeout is not None:
        task = asyncio.create_task(
            _keepalive(backend.bind(), max(10.0, backend.config.idle_timeout / 3))
        )
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@asynccontextmanager
async def lan_network(rules):
    """Own a LAN with explicit rules installed before any member joins."""
    async with leased_client() as client:
        network = None

        async def create():
            nonlocal network
            network = await client.create_lan_network(
                f"vf-{uuid.uuid4().hex[:16]}", rules=rules
            )

        try:
            await run_shielded(create())
            yield network.id
        finally:
            if network is not None:
                try:
                    await run_shielded(client.delete_lan_network(network.id))
                except Exception:
                    logger.warning(
                        "LAN network %s deletion failed", network.id, exc_info=True
                    )


@asynccontextmanager
async def private_network(
    workspace: RuntimeBackend,
    members: Mapping[str, PrimeConfig],
    start_workspace: Callable[[], Awaitable[object]] | None = None,
    *,
    interconnect: bool = False,
    networks: Mapping[str, list[str]] | None = None,
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
    rules = (
        [LanRule(from_="*", to="*")]
        if interconnect
        else [
            LanRule(from_=f"tag:{WORKSPACE_TAG}", to=f"tag:{tag}")
            for tag in tags.values()
        ]
    )
    if networks is not None:
        rules = []
        for index, names in enumerate(networks.values()):
            tag = f"vf-network-{index}"
            rules.append(LanRule(from_=f"tag:{tag}", to=f"tag:{tag}"))
            for name in names:
                memberships[name].append(tag)
    async with AsyncExitStack() as stack:
        network_id = await stack.enter_async_context(lan_network(rules))
        if start_workspace is not None:
            # A failed boot must release the workspace before its network.
            stack.push_async_callback(workspace.stop)
            workspace.lan = (network_id, memberships["main"])
        else:
            await stack.enter_async_context(
                workspace.join_network(network_id, memberships["main"])
            )

        backends: dict[str, RuntimeBackend] = {}

        async def provision(name: str, config: PrimeConfig) -> None:
            backend = make_backend(config, f"{workspace.name}-{tags[name]}")
            assert isinstance(backend, PrimeBackend)
            backend.lan = (network_id, memberships[name])
            # Pushed before `start()`: a failed start may already hold a paid sandbox.
            stack.push_async_callback(backend.stop)
            await backend.start()  # running, and joined to the network
            backends[name] = backend
            await stack.enter_async_context(keepalive(backend))

        try:
            async with asyncio.TaskGroup() as group:
                if start_workspace is not None:
                    group.create_task(start_workspace())
                for name, config in members.items():
                    group.create_task(provision(name, config))
        except ExceptionGroup as failed:
            raise failed.exceptions[0]  # one VM's failure, typed for retries
        logger.info(
            "LAN network %s joins %s to %s",
            network_id,
            workspace.info.id,
            {name: backend.info.id for name, backend in backends.items()},
        )
        yield backends
