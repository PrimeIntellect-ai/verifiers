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
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping
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
    *,
    interconnect: bool = False,
    networks: Mapping[str, list[str]] | None = None,
    harness: RuntimeBackend | None = None,
    start_harness: Callable[[], Awaitable[object]] | None = None,
    harness_peers: Collection[str] = (),
    own_harness: bool = True,
    assign_harness_lan: bool = True,
    network_state: dict | None = None,
) -> AsyncIterator[dict[str, RuntimeBackend]]:
    """Boot the workspace's VM and one VM per member together, each joining a LAN
    network of the rollout's own as it boots. A split deployment's `harness` VM joins
    too: it may open connections to the workspace, to tool servers' VMs and to the
    task services in `harness_peers`, and nothing may open one to it."""
    if not isinstance(workspace, PrimeBackend):
        raise TypeError(
            "tool servers on Prime runtimes of their own join the workspace over a "
            f"private network, which needs the workspace on Prime, not "
            f"{workspace.config.type}"
        )
    from prime_sandboxes import LanRule

    from verifiers.v1.runtimes.controller import current_job

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
        rules.append(LanRule(from_=f"tag:{HARNESS_TAG}", to=f"tag:{WORKSPACE_TAG}"))
        # The harness consumes the tool servers, on VMs of their own or in services.
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
        # Install connectivity before boot: Compose peers must share a network;
        # independent tool servers accept only workspace connections.
        name = f"vf-{uuid.uuid4().hex[:16]}"
        job = current_job.get()
        if job and harness is not None:
            # B and task services have no route into C. Only A can reach the session port.
            rules += [
                LanRule(
                    from_=f"tag:{HARNESS_TAG}",
                    to="tag:vf-controller",
                    ports=[job.server.port],
                ),
                LanRule(from_="tag:vf-controller", to=f"tag:{HARNESS_TAG}"),
            ]

        deleted = False
        delete_lock = asyncio.Lock()

        async def delete_network(network_id: str) -> None:
            nonlocal deleted
            async with delete_lock:
                if deleted:
                    return
                for attempt in range(2):
                    try:
                        if job:
                            await job.launcher.call(
                                "delete_network", network_id=network_id
                            )
                        else:
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
            if job:
                network_id = await job.launcher.call(
                    "network",
                    join_controller=harness is not None,
                    name=name,
                    rules=[rule.model_dump(by_alias=True) for rule in rules],
                )
            else:
                network_id = (await client.create_lan_network(name, rules=rules)).id
            stack.push_async_callback(lambda: run_shielded(delete_network(network_id)))
            if network_state is not None:
                network_state["id"] = network_id
                network_state["delete"] = lambda: run_shielded(
                    delete_network(network_id)
                )
            return network_id

        network_task = asyncio.create_task(create_and_own())
        if network_state is not None:
            network_state["task"] = network_task
        await asyncio.sleep(0)
        if network_task.done() and network_task.exception() is not None:
            await network_task

        def assign_lan(backend: PrimeBackend, member_tags: list[str]) -> None:
            if network_task.done():
                backend.lan = (network_task.result(), member_tags)
            else:
                backend._lan_pending = (network_task, member_tags)

        started_vms: list[RuntimeBackend] = [workspace]

        async def stop_all_vms() -> None:
            await asyncio.gather(
                *(vm.stop() for vm in started_vms), return_exceptions=True
            )

        # Failed startup unwinds VMs concurrently before deleting the network.
        stack.push_async_callback(stop_all_vms)

        backends: dict[str, RuntimeBackend] = {}

        async def stop(task: asyncio.Task) -> None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        async def provision(name: str, config: PrimeConfig) -> None:
            backend = make_backend(config, f"{workspace.name}-{tags[name]}")
            assert isinstance(backend, PrimeBackend)
            assign_lan(backend, memberships[name])
            # Registered before `start()`: a failed start may already hold a paid sandbox.
            started_vms.append(backend)
            await backend.start()
            backends[name] = backend
            # LAN traffic is not activity to the platform, and a server's VM sees no
            # commands of its own: keep it inside its idle timeout until it is released.
            if config.idle_timeout is not None:
                every = max(10.0, config.idle_timeout / 3)
                stack.push_async_callback(
                    stop, asyncio.create_task(_keepalive(backend.bind(), every))
                )

        async def provision_workspace() -> None:
            await start_workspace()
            idle_timeout = workspace.config.idle_timeout
            if harness is not None and idle_timeout is not None:
                # The harness's tools reach the workspace over the LAN alone.
                stack.push_async_callback(
                    stop,
                    asyncio.create_task(
                        _keepalive(workspace.bind(), max(10.0, idle_timeout / 3))
                    ),
                )

        assign_lan(workspace, memberships["main"])
        if harness is not None:
            assert isinstance(harness, PrimeBackend)
            if assign_harness_lan:
                assign_lan(harness, [HARNESS_TAG])
            if own_harness:
                started_vms.append(harness)
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(run_shielded(network_task))
                group.create_task(provision_workspace())
                if start_harness is not None:
                    group.create_task(start_harness())
                for name, config in members.items():
                    group.create_task(provision(name, config))
        except ExceptionGroup as failed:
            raise failed.exceptions[0]  # one VM's failure, typed for retries
        network_id = network_task.result()
        logger.debug(
            "Joined LAN network %s from %s to %s",
            network_id,
            workspace.info.id,
            {name: backend.info.id for name, backend in backends.items()},
        )
        yield backends
