"""Prime task services and tool servers, reached over a rollout's private LAN.

A Prime VM's egress policy covers the whole VM, so a tool server whose egress or
isolation differs from the workspace's, such as one that calls public APIs beside a
workspace with none, or one that runs agent-submitted code with none, runs on a Prime
runtime of its own (`ToolsetConfig.runtime`). Independent tool VMs accept connections
only from the workspace; task-declared services can also call each other and the
workspace. The harness reaches a server at its VM's LAN name,
`<sandbox-id>.sandbox.internal`, which the LAN's DNS keeps current.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from verifiers.v1.runtimes import (
    PrimeBackend,
    PrimeConfig,
    RuntimeBackend,
    make_backend,
)
from verifiers.v1.runtimes.backends.prime import leased_client
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import get_logger

if TYPE_CHECKING:
    from verifiers.v1.runtimes.harness_pool import HarnessLease

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


async def _keepalive(runtime, every: float) -> None:
    while True:
        await asyncio.sleep(every)
        with contextlib.suppress(Exception):
            await runtime.run(["true"], {})


@dataclass
class PrivateNetworkGroup:
    """Live private LAN and member VMs owned by one rollout deployment."""

    backends: dict[str, RuntimeBackend] = field(default_factory=dict)
    network_id: str | None = None
    _delete_fn: Callable[[], Awaitable[None]] | None = None

    async def delete_network(self) -> None:
        if self._delete_fn is not None:
            await self._delete_fn()


@asynccontextmanager
async def private_network(
    workspace: RuntimeBackend,
    members: Mapping[str, PrimeConfig],
    start_workspace: Callable[[], Awaitable[object]],
    *,
    interconnect: bool = False,
    networks: Mapping[str, list[str]] | None = None,
    harness_lease: HarnessLease | None = None,
    harness_peers: Collection[str] = (),
    borrowed_workspace: bool = False,
    out: PrivateNetworkGroup | None = None,
) -> AsyncIterator[dict[str, RuntimeBackend]]:
    """Boot the workspace's VM and one VM per member together, each joining a LAN
    network of the rollout's own as it boots. A pooled `harness_lease` joins too: it
    may open connections to the workspace, to tool servers' VMs and to the task services
    in `harness_peers`, and nothing may open one to it."""
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
    has_harness = harness_lease is not None
    if has_harness:
        rules.append(LanRule(from_=f"tag:{HARNESS_TAG}", to=f"tag:{WORKSPACE_TAG}"))
        consumed = (
            tags.values()
            if networks is None and not interconnect
            else [tags[name] for name in harness_peers if name in tags]
        )
        rules += [
            LanRule(from_=f"tag:{HARNESS_TAG}", to=f"tag:{tag}") for tag in consumed
        ]
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(leased_client())
        name = f"vf-{uuid.uuid4().hex[:16]}"

        deleted = False
        delete_lock = asyncio.Lock()

        async def delete_network(network_id: str) -> None:
            nonlocal deleted
            async with delete_lock:
                if deleted:
                    return
                for attempt in range(2):
                    try:
                        await client.delete_lan_network(network_id)
                        deleted = True
                        return
                    except Exception:
                        if attempt == 0:
                            await asyncio.sleep(0.3)
                            continue
                        deleted = True
                        logger.warning(
                            "LAN network %s deletion failed", network_id, exc_info=True
                        )

        async def create_and_own() -> str:
            network_id = (await client.create_lan_network(name, rules=rules)).id
            delete_cb = lambda: run_shielded(delete_network(network_id))
            stack.push_async_callback(delete_cb)
            if out is not None:
                out.network_id = network_id
                out._delete_fn = delete_cb
            return network_id

        network_task = asyncio.create_task(create_and_own())
        await asyncio.sleep(0)
        if network_task.done() and network_task.exception() is not None:
            await network_task

        def assign_lan(backend: PrimeBackend, member_tags: list[str]) -> None:
            if network_task.done():
                backend.lan = (network_task.result(), member_tags)
            else:
                backend._lan_pending = (network_task, member_tags)

        started_vms: list[RuntimeBackend] = [] if borrowed_workspace else [workspace]

        async def stop_all_vms() -> None:
            await asyncio.gather(
                *(vm.stop() for vm in started_vms), return_exceptions=True
            )

        # Failed startup unwinds VMs concurrently before deleting the network.
        stack.push_async_callback(stop_all_vms)

        backends: dict[str, RuntimeBackend] = {}

        async def stop_keepalive(task: asyncio.Task) -> None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        async def provision(name: str, config: PrimeConfig) -> None:
            backend = make_backend(config, f"{workspace.name}-{tags[name]}")
            assert isinstance(backend, PrimeBackend)
            assign_lan(backend, memberships[name])
            started_vms.append(backend)
            await backend.start()
            backends[name] = backend
            if config.idle_timeout is not None:
                every = max(10.0, config.idle_timeout / 3)
                stack.push_async_callback(
                    stop_keepalive,
                    asyncio.create_task(_keepalive(backend.bind(), every)),
                )

        async def detach_borrowed_workspace() -> None:
            net_id = out.network_id if out is not None else None
            if (
                not net_id
                or workspace.stopped
                or workspace.info.id is None
                or workspace._client is None
            ):
                return
            try:
                await workspace._client.remove_lan_member(net_id, workspace.info.id)
            except Exception:
                logger.warning(
                    "Failed to detach borrowed prime workspace %s from LAN %s",
                    workspace.info.id,
                    net_id,
                    exc_info=True,
                )

        async def provision_workspace() -> None:
            if borrowed_workspace:
                from verifiers.v1.runtimes.backends.prime import join_lan

                net_id = await asyncio.shield(network_task)
                stack.push_async_callback(
                    lambda: run_shielded(detach_borrowed_workspace())
                )
                await join_lan(workspace, net_id, memberships["main"])
            else:
                await start_workspace()
            idle_timeout = workspace.config.idle_timeout
            if has_harness and idle_timeout is not None:
                stack.push_async_callback(
                    stop_keepalive,
                    asyncio.create_task(
                        _keepalive(workspace.bind(), max(10.0, idle_timeout / 3))
                    ),
                )

        async def provision_harness() -> None:
            assert harness_lease is not None
            try:
                await harness_lease.start(network_task, [HARNESS_TAG])
            except asyncio.CancelledError:
                raise
            except BaseException:
                net_id = out.network_id if out is not None else None
                await harness_lease.release(net_id, healthy=False)
                raise

        if not borrowed_workspace:
            assign_lan(workspace, memberships["main"])
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(run_shielded(network_task))
                group.create_task(provision_workspace())
                if harness_lease is not None:
                    group.create_task(provision_harness())
                for name, config in members.items():
                    group.create_task(provision(name, config))
        except ExceptionGroup as failed:
            raise failed.exceptions[0]
        if out is not None:
            out.backends = backends
        logger.debug(
            "Joined LAN network %s from %s to %s",
            network_task.result(),
            workspace.info.id,
            {name: backend.info.id for name, backend in backends.items()},
        )
        yield backends
