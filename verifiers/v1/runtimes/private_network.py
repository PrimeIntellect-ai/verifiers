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
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
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
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)

WORKSPACE_TAG = "vf-workspace"


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


_ALIAS = r"""
for _ in $(seq 30); do
  address=$(getent hosts "$2" | awk 'NR == 1 {print $1}')
  [ -n "$address" ] && break
  sleep 1
done
[ -n "$address" ] || { echo "$2 does not resolve" >&2; exit 1; }
grep -v " $1\$" /etc/hosts > /tmp/vf-hosts || true
printf '%s %s\n' "$address" "$1" >> /tmp/vf-hosts
cat /tmp/vf-hosts > /etc/hosts
"""


async def alias_hosts(runtime: Runtime, aliases: Mapping[str, str]) -> None:
    """Make the box resolve each old LAN name in `aliases` to the address of its new
    one. A box restored from a memory snapshot still reaches its peers at the names of
    the VMs it first ran beside; restored peers are new VMs with new names."""
    for old, new in aliases.items():
        if old == new:
            continue
        result = await runtime.run(["sh", "-c", _ALIAS, "sh", old, new], {})
        if result.exit_code != 0:
            raise SandboxError(
                f"could not resolve {old} to {new} in the restored box: "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )


async def rejoin(backends: list[RuntimeBackend]) -> None:
    """Join each VM to its LAN network again after a memory snapshot. A VM that resumes in
    place reaches no LAN peer, by name or address, though the network still lists it as
    a synced member; leaving and joining again restores its routes and names."""
    joined = [
        backend
        for backend in backends
        if isinstance(backend, PrimeBackend) and backend.lan is not None
    ]
    async with leased_client() as client:
        for backend in joined:
            await client.remove_lan_member(backend.lan[0], backend.info.id)
        for backend in joined:
            await client._join_lan(backend.lan[0], backend.info.id, backend.lan[1])


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

        workspace.lan = (network.id, memberships["main"])
        try:
            async with asyncio.TaskGroup() as group:
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
