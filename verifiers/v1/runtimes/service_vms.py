"""Task services on Prime VMs of their own, reached from the workspace over a LAN.

A Prime VM's egress policy covers the whole VM, so a service whose egress or isolation
differs from the workspace's, such as a tool server that calls public APIs beside a
workspace with none, or one that runs agent-submitted code with none, runs on a VM of its
own (`ServiceVMSpec`). Each rollout gets a LAN network on which the workspace's VM may open
connections to the services and nothing may open one to the workspace. The workspace reaches a service at its LAN name,
`<sandbox-id>.sandbox.internal` (`Deployment.hosts`), which the LAN's DNS keeps current.
"""

import asyncio
import contextlib
import logging
import re
import shlex
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeBackend, RuntimeBackend, make_backend
from verifiers.v1.runtimes.prime import leased_client
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

WORKSPACE = "main"
WORKSPACE_TAG = "vf-workspace"
# Service names become LAN tags (`vf-service-<name>`), which allow these characters.
_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,50}")
# Without a readiness probe, a service need only still be running this long after it starts.
STARTED_SECONDS = 1.0
SYNC_SECONDS = 120
SYNC_POLL_SECONDS = 1.0


@dataclass(frozen=True)
class Probe:
    """A readiness check run in the service's VM: the service is ready once `argv`
    exits 0."""

    argv: list[str]
    interval: float = 5.0
    timeout: float = 10.0
    attempts: int = 60

    def __post_init__(self):
        if self.attempts < 1:
            raise ValueError("a readiness probe needs at least one attempt")


@dataclass(frozen=True)
class ServiceVM:
    """A long-running task service on a Prime VM of its own."""

    image: str
    """Docker image of the service's VM."""
    command: list[str]
    """The service's process; a VM does not run the image's entrypoint. Like a
    container's main process, it must keep running: setup fails if it exits."""
    environment: Mapping[str, str] = field(default_factory=dict)
    """The process's environment. It reaches the service's VM only, never the
    workspace."""
    workdir: str | None = None
    """Working directory of the process and its probe, created if missing."""
    readiness: Probe | None = None
    """Setup waits until the probe passes. Without one, the process need only still
    be running a moment after it starts."""
    cpu: float | None = None
    memory: float | None = None
    """CPU cores and memory (GB) of the service's VM; None uses the workspace's."""
    allow: list[str] = field(default_factory=lambda: ["*"])
    """Destinations the service's VM may reach once setup is done, as in a runtime's
    `allow`: `*` is unrestricted, and `[]` keeps only the framework's routes, such as
    a tool server's state channel. Setup itself, readiness included, is unrestricted."""


@dataclass(frozen=True)
class ServiceVMSpec:
    """Task services on Prime VMs of their own. The workspace (`main`) is the agent's
    own VM; `ToolsetConfig.service` can place a task's tool server in a service."""

    services: Mapping[str, ServiceVM]

    def __post_init__(self):
        for name in self.services:
            if name == WORKSPACE or not _NAME.fullmatch(name):
                raise ValueError(
                    f"service VM name {name!r} must not be {WORKSPACE!r} and must be "
                    "lowercase letters, digits, '.', '_' or '-'"
                )


def lan_host(backend: RuntimeBackend) -> str:
    """The name a LAN member resolves a service's VM by."""
    return f"{backend.info.id}.sandbox.internal"


def _tag(name: str) -> str:
    return f"vf-service-{name}"


def in_workdir(argv: list[str], workdir: str | None) -> list[str]:
    """Run `argv` in `workdir`, created if missing."""
    if workdir is None:
        return argv
    directory = shlex.quote(workdir)
    return [
        "sh",
        "-c",
        f"mkdir -p {directory} && cd {directory} && exec {shlex.join(argv)}",
    ]


async def _start(runtime, name: str, service: ServiceVM) -> None:
    log = f"/tmp/vf-service-{name}.log"
    status = f"/tmp/vf-service-{name}.exit"
    environment = dict(service.environment)
    # Record the exit status, so a crashed service fails setup instead of its readiness wait.
    script = f"{shlex.join(in_workdir(service.command, service.workdir))}; echo $? > {shlex.quote(status)}"
    await runtime.run_background(["sh", "-c", script], environment, log)

    async def exited() -> str | None:
        result = await runtime.run(["cat", status], {})
        return None if result.exit_code else f"exited with {result.stdout.strip()}"

    async def failure(state: str) -> SandboxError:
        tail = await runtime.run(["tail", "-c", "2000", log], {})
        return SandboxError(f"service {name!r} {state}: {tail.stdout}")

    probe = service.readiness
    if probe is None:
        await asyncio.sleep(STARTED_SECONDS)
        if state := await exited():
            raise await failure(state)
        return
    # The probe sees the service's environment and working directory.
    check = in_workdir(probe.argv, service.workdir)
    for attempt in range(probe.attempts):
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(probe.timeout + 10):
                if (await runtime.run(check, environment)).exit_code == 0:
                    if state := await exited():
                        raise await failure(state)
                    return
        if state := await exited():
            raise await failure(state)
        if attempt == probe.attempts - 1:
            raise await failure("never became ready")
        await asyncio.sleep(probe.interval)


async def _keepalive(runtime, every: float) -> None:
    while True:
        await asyncio.sleep(every)
        with contextlib.suppress(Exception):
            await runtime.run(["true"], {})


@asynccontextmanager
async def service_vms(
    workspace: RuntimeBackend,
    services: Mapping[str, ServiceVM],
    start_workspace: Callable[[], Awaitable[object]],
) -> AsyncIterator[dict[str, RuntimeBackend]]:
    """Boot the workspace's VM and one VM per service together, each joining a LAN
    network of their own as it boots, and start the services."""
    if not isinstance(workspace, PrimeBackend):
        raise TypeError(f"service VMs run on Prime VMs, not {workspace.config.type}")
    from prime_sandboxes import LanRule

    config = workspace.config
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(leased_client())
        network = await client.create_lan_network(f"vf-{uuid.uuid4().hex[:16]}")

        async def delete_network() -> None:
            try:
                await client.delete_lan_network(network.id)
            except Exception:
                logger.warning(
                    "LAN network %s deletion failed", network.id, exc_info=True
                )

        stack.push_async_callback(lambda: run_shielded(delete_network()))
        # A new network lets every member reach every other: narrow it before anyone
        # joins, and wait until the platform enforces that, since a member's join can
        # sync before the network's rules do.
        state = await client.set_lan_rules(
            network.id,
            [
                LanRule(from_=f"tag:{WORKSPACE_TAG}", to=f"tag:{_tag(name)}")
                for name in services
            ],
        )
        try:
            async with asyncio.timeout(SYNC_SECONDS):
                while not state.synced:
                    await asyncio.sleep(SYNC_POLL_SECONDS)
                    state = await client.get_lan_network(network.id)
        except TimeoutError as e:
            raise SandboxError(
                f"LAN network {network.id}'s rules were not applied within "
                f"{SYNC_SECONDS}s"
            ) from e

        backends: dict[str, RuntimeBackend] = {}

        async def stop(task: asyncio.Task) -> None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        async def provision(name: str, service: ServiceVM) -> None:
            vm = config.model_copy(
                update={
                    "image": service.image,
                    "workdir": "/",
                    "checkpoint": None,
                    "allow": list(service.allow),
                    "block": [],
                    "gpu": None,
                    "cpu": service.cpu or config.cpu,
                    "memory": service.memory or config.memory,
                }
            )
            backend = make_backend(vm, f"{workspace.name}-{name}")
            assert isinstance(backend, PrimeBackend)
            backend.lan = (network.id, [_tag(name)])
            # Pushed before `start()`: a failed start may already hold a paid sandbox.
            stack.push_async_callback(backend.stop)
            await backend.start()  # running, and joined to the network
            backends[name] = backend
            # LAN traffic is not activity to the platform, and a ready service sees no
            # commands: keep it inside its idle timeout until the rollout releases it.
            if config.idle_timeout is not None:
                every = max(10.0, config.idle_timeout / 3)
                stack.push_async_callback(
                    stop, asyncio.create_task(_keepalive(backend.bind(), every))
                )
            await _start(backend.bind(), name, service)

        workspace.lan = (network.id, [WORKSPACE_TAG])
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(start_workspace())
                for name, service in services.items():
                    group.create_task(provision(name, service))
        except ExceptionGroup as failed:
            raise failed.exceptions[0]  # one VM's failure, typed for retries
        logger.info(
            "LAN network %s joins %s to services %s",
            network.id,
            workspace.info.id,
            {name: backend.info.id for name, backend in backends.items()},
        )
        yield backends
