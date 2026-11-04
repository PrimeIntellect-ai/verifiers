"""Prime task services and tool servers, reached over private LANs.

A Prime VM's egress policy covers the whole VM, so a tool server whose egress or
isolation differs from the workspace's, such as one that calls public APIs beside a
workspace with none, or one that runs agent-submitted code with none, runs on a Prime
runtime of its own (`ToolsetConfig.runtime`). A rollout's own tool VMs join its LAN,
beside any task services; a `shared` tool VM joins a LAN of the eval's own, which every
rollout's workspace joins. Tool VMs accept connections only from workspaces, and
workspaces never accept any. A member is reached at its VM's LAN name,
`<sandbox-id>.sandbox.internal`, which the LAN's DNS keeps current.

A sandbox joins at most 32 LAN networks, so one shared VM cannot join every rollout's
LAN; the eval LAN has no member cap instead.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Collection, Coroutine, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    PrimeBackend,
    PrimeConfig,
    Runtime,
    RuntimeBackend,
    make_backend,
)
from verifiers.v1.runtimes.backends.prime import leased_client
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

WORKSPACE_TAG = "vf-workspace"
LAN_DOMAIN = ".sandbox.internal"
_JOIN_TIMEOUT_SECONDS = 120
_JOIN_POLL_SECONDS = 2
# Waits in the workspace until its DNS answers for every LAN name given; images without
# getent cannot check, and connect as they would anyway.
_RESOLVE = (
    "command -v getent >/dev/null || exit 0; "
    f"for _ in $(seq {_JOIN_TIMEOUT_SECONDS * 2}); do "
    'getent hosts "$@" >/dev/null && exit 0; sleep 0.5; done; exit 1'
)


def tool_vms(toolsets, workspace, *, borrowed: bool) -> dict[str, PrimeConfig]:
    """The task's tool servers that join the rollout's LAN, by server name: those that
    serve this rollout alone from a Prime runtime of their own, beside a Prime
    workspace of the rollout's own."""
    if borrowed or not isinstance(workspace, PrimeConfig):
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
    return f"{backend.info.id}{LAN_DOMAIN}"


def _tag(index: int) -> str:
    # Server names need not be valid LAN tags; a member's position names it instead.
    return f"vf-tool-{index}"


async def _keepalive(runtime, every: float) -> None:
    while True:
        await asyncio.sleep(every)
        with contextlib.suppress(Exception):
            await runtime.run(["true"], {})


async def _lan(stack: AsyncExitStack, rules) -> str:
    """Create a LAN network with its rules in force before anyone joins, deleted with
    `stack`; return its id."""
    client = await stack.enter_async_context(leased_client())
    network = await client.create_lan_network(
        f"vf-{uuid.uuid4().hex[:16]}", rules=rules
    )

    async def delete() -> None:
        try:
            await client.delete_lan_network(network.id)
        except Exception:
            logger.warning("LAN network %s deletion failed", network.id, exc_info=True)

    stack.push_async_callback(lambda: run_shielded(delete()))
    return network.id


async def _boot(
    stack: AsyncExitStack, config: PrimeConfig, name: str, lan: tuple[str, list[str]]
) -> PrimeBackend:
    """Boot a VM that joins `lan` as it boots, released with `stack`."""
    backend = make_backend(config, name)
    assert isinstance(backend, PrimeBackend)
    backend.lan = lan
    # Pushed before `start()`: a failed start may already hold a paid sandbox.
    stack.push_async_callback(backend.stop)
    await backend.start()

    async def stop(task: asyncio.Task) -> None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    # LAN traffic is not activity to the platform, and a server's VM sees no commands
    # of its own: keep it inside its idle timeout until it is released.
    if config.idle_timeout is not None:
        every = max(10.0, config.idle_timeout / 3)
        stack.push_async_callback(
            stop, asyncio.create_task(_keepalive(backend.bind(), every))
        )
    return backend


async def _boot_all(
    boots: Mapping[str, Coroutine[None, None, PrimeBackend]],
    *others: Coroutine,
) -> dict[str, PrimeBackend]:
    """Boot VMs together, beside `others` such as the workspace's own start."""
    backends: dict[str, PrimeBackend] = {}

    async def boot(name: str, pending: Coroutine[None, None, PrimeBackend]) -> None:
        backends[name] = await pending

    try:
        async with asyncio.TaskGroup() as group:
            for other in others:
                group.create_task(other)
            for name, pending in boots.items():
                group.create_task(boot(name, pending))
    except ExceptionGroup as failed:
        raise failed.exceptions[0]  # one VM's failure, typed for retries
    return backends


@asynccontextmanager
async def private_network(
    workspace: RuntimeBackend,
    members: Mapping[str, PrimeConfig],
    start_workspace: Callable[[], Coroutine],
    *,
    tools: frozenset[str] = frozenset(),
    interconnect: bool = False,
    networks: Mapping[str, list[str]] | None = None,
) -> AsyncIterator[dict[str, RuntimeBackend]]:
    """Boot the workspace's VM and one VM per member together, each joining a LAN
    network of the rollout's own as it boots. The workspace may connect to the `tools`
    members; task services follow `networks`, or all reach each other with
    `interconnect`."""
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
    rules = [
        LanRule(from_=f"tag:{WORKSPACE_TAG}", to=f"tag:{tag}")
        for name, tag in tags.items()
        if name in tools
    ]
    if networks is not None:
        for index, names in enumerate(networks.values()):
            tag = f"vf-network-{index}"
            rules.append(LanRule(from_=f"tag:{tag}", to=f"tag:{tag}"))
            for name in names:
                memberships[name].append(tag)
    elif interconnect:
        rules.append(LanRule(from_="*", to="*"))
    async with AsyncExitStack() as stack:
        # Install connectivity before boot: Compose peers must share a network;
        # independent tool servers accept only workspace connections.
        network = await _lan(stack, rules)
        # Failed startup unwinds this stack before the deployment's workspace owner.
        stack.push_async_callback(workspace.stop)
        workspace.lan = (network, memberships["main"])
        backends = await _boot_all(
            {
                name: _boot(
                    stack,
                    config,
                    f"{workspace.name}-{tags[name]}",
                    (network, memberships[name]),
                )
                for name, config in members.items()
            },
            start_workspace(),
        )
        logger.info(
            "LAN network %s joins %s to %s",
            network,
            workspace.info.id,
            {name: backend.info.id for name, backend in backends.items()},
        )
        yield backends


@asynccontextmanager
async def eval_lan(
    members: Mapping[str, PrimeConfig],
) -> AsyncIterator[tuple[str, dict[str, PrimeBackend]]]:
    """Boot one VM per shared tool server on a LAN of the eval's own; yield the LAN's
    id, which each rollout's workspace joins (`join_lan`), and the VMs by name."""
    from prime_sandboxes import LanRule

    tags = {name: _tag(index) for index, name in enumerate(members)}
    async with AsyncExitStack() as stack:
        network = await _lan(
            stack,
            [
                LanRule(from_=f"tag:{WORKSPACE_TAG}", to=f"tag:{tag}")
                for tag in tags.values()
            ],
        )
        prefix = f"vf-eval-{uuid.uuid4().hex[:8]}"
        backends = await _boot_all(
            {
                name: _boot(stack, config, f"{prefix}-{tags[name]}", (network, [tag]))
                for (name, config), tag in zip(members.items(), tags.values())
            }
        )
        logger.info(
            "eval LAN network %s serves %s",
            network,
            {name: backend.info.id for name, backend in backends.items()},
        )
        yield network, backends


async def join_lan(workspace: Runtime, network: str, hosts: Collection[str]) -> None:
    """Join a running Prime workspace to an eval LAN, and wait until it resolves the
    LAN names `hosts`. The join's own response reports when the platform has applied
    it: a read of the network lists every member, so polling one costs as much as the
    eval is wide. The workspace's DNS can lag that report."""
    sandbox = workspace.backend.info.id
    async with leased_client() as client:
        try:
            async with asyncio.timeout(_JOIN_TIMEOUT_SECONDS):
                while not (
                    await client.add_lan_member(network, sandbox, [WORKSPACE_TAG])
                ).synced:
                    await asyncio.sleep(_JOIN_POLL_SECONDS)
        except TimeoutError as e:
            raise SandboxError(
                f"workspace {sandbox} did not join LAN network {network} "
                f"within {_JOIN_TIMEOUT_SECONDS}s"
            ) from e
    resolved = await workspace.run(["sh", "-c", _RESOLVE, "resolve", *hosts], {})
    if resolved.exit_code:
        raise SandboxError(
            f"workspace {sandbox} joined LAN network {network} but does not resolve "
            f"{sorted(hosts)} within {_JOIN_TIMEOUT_SECONDS}s"
        )
