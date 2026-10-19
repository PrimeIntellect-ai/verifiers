"""How a deployment realizes its plan: the workspace alone, the workspace's machine
packed with Compose services, or machines of their own on a private network."""

import asyncio
import json
import logging
import tempfile
from collections.abc import Collection, Mapping
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path
from typing import Any

from verifiers.v1.runtimes import Runtime, RuntimeConfig
from verifiers.v1.runtimes.compose import ComposeSpec, compose_services
from verifiers.v1.runtimes.deployment import _DeploymentStrategy
from verifiers.v1.runtimes.nodes import (
    WORKSPACE,
    PlacementError,
    ProcessConfig,
    compose_project,
)
from verifiers.v1.runtimes.private_network import private_network

logger = logging.getLogger(__name__)


class _ColocatedDeployment(_DeploymentStrategy):
    async def start(self, d):
        if d.borrowed_runtime is not None:
            d.execution_backend = d.harness_backend = d.borrowed_runtime.backend
        else:
            d.backend.env = d.env
            await d.own(d.backend)
            d.execution_backend = d.harness_backend = d.backend
        d.target_backends = {"main": d.execution_backend}

    async def prepare_verifier(self, deployment):
        # Borrowed solvers belong to their caller; owned solvers can release their
        # resources now that metrics and submission artifacts have been collected.
        if deployment.borrowed_runtime is None:
            try:
                await deployment.execution_backend.stop()
            except Exception:
                logger.warning("solver teardown failed before grading", exc_info=True)


class _ContainerDeployment(_DeploymentStrategy):
    def __init__(self, project: dict[str, Any] | None = None) -> None:
        self._project = project
        """A Compose project generated for declared nodes, in place of the task's."""

    async def start(self, d):
        if d.borrowed_runtime is not None:
            raise ValueError("a composed deployment cannot borrow a single target")
        self._services = await d.enter(AsyncExitStack())
        spec = d.spec
        if self._project is not None:
            directory = self._services.enter_context(
                tempfile.TemporaryDirectory(prefix="vf-nodes-")
            )
            (Path(directory) / "compose.json").write_text(json.dumps(self._project))
            spec = ComposeSpec(directory=directory, file="compose.json")
        services, self._stop_main, _ = await self._services.enter_async_context(
            compose_services(
                d.backend.config, spec, d.env, trust_compose=d.config.trust_compose
            )
        )
        d.target_backends = dict(services)
        d.execution_backend = d.harness_backend = services[spec.workspace]

    async def stop_execution(self, d):
        await self._stop_main()

    async def prepare_verifier(self, deployment):
        # Evidence is collected; release the solver before the verifier clock starts.
        try:
            await self._services.aclose()
        except Exception:
            logger.warning(
                "solver services teardown failed before grading", exc_info=True
            )


class _PrivateNetworkDeployment(_DeploymentStrategy):
    def __init__(
        self,
        machines: Mapping[str, RuntimeConfig],
        rules: Collection[tuple[str, str]],
        processes: Mapping[str, ProcessConfig] | None = None,
        served: Collection[str] = (),
    ) -> None:
        self._machines = dict(machines)
        """Each node on a machine of its own, by name, with that machine's runtime."""
        self._rules = list(rules)
        self._processes = dict(processes or {})
        self._served = set(served)
        """The nodes that run tool servers, which keep their state channel."""

    async def start(self, d):
        if d.borrowed_runtime is not None:
            raise ValueError(
                "a deployment with machines of its own cannot borrow a single target"
            )
        d.backend.env = d.env
        self._services = await d.enter(AsyncExitStack())
        machines = await self._services.enter_async_context(
            private_network(
                d.backend, self._machines, partial(d.own, d.backend), self._rules
            )
        )
        d.target_backends = {WORKSPACE: d.backend, **machines}
        d.hosts = {name: backend.network_host for name, backend in machines.items()}
        d.execution_backend = d.harness_backend = d.backend
        self._service_backends = machines
        await asyncio.gather(
            *(
                start_process(machines[name].bind(), process)
                for name, process in self._processes.items()
            )
        )

    async def prepare_execution(self, d, routes, state_route=None):
        # Each machine enforces its own policy. A tool server's machine keeps only its
        # state channel, reached as the server reaches it; private network traffic is
        # unaffected by either.
        def own(name, backend):
            if state_route and name in self._served:
                return [backend.host_url(state_route)]
            return []

        await asyncio.gather(
            d.harness_backend.prepare_execution(routes),
            *(
                backend.prepare_execution(own(name, backend))
                for name, backend in self._service_backends.items()
            ),
        )

    async def stop_execution(self, d):
        # As Compose stops its workspace: nothing the agent left running may reach the
        # tool servers' VMs while their evidence is collected.
        await d.execution_backend.stop()

    async def prepare_verifier(self, deployment):
        # Evidence is collected; release the solver's VMs before the verifier clock
        # starts, the workspace's even if the tool server VMs' or the network's release
        # fails.
        for release, what in (
            (deployment.execution_backend.stop, "workspace VM"),
            (self._services.aclose, "tool server VMs"),
        ):
            try:
                await release()
            except Exception:
                logger.warning("%s teardown failed before grading", what, exc_info=True)


async def start_process(runtime: Runtime, process: ProcessConfig) -> None:
    """Start a node's process on its machine and wait until its readiness check
    passes."""
    assert process.command is not None
    argv = process.command
    if process.workdir is not None:
        argv = [
            "sh",
            "-c",
            'cd "$1" && shift && exec "$@"',
            "sh",
            process.workdir,
            *argv,
        ]
    await runtime.run_background(argv, dict(process.environment), "vf-process.log")
    if process.readiness is not None:
        while (await runtime.run(process.readiness, {})).exit_code:
            await asyncio.sleep(1)


def _deployment_strategy(config, backend, spec, plan=None, topology=None):
    """The realization of `plan` (from `topology`), refused before anything is
    provisioned when this deployment cannot realize it. Without a plan, the task's
    Compose services or the workspace alone."""
    if plan is None or topology is None:
        if spec is None:
            return _ColocatedDeployment()
        if not backend.SUPPORTS_COMPOSE:
            raise ValueError(
                f"the task deploys Compose services, but the {backend.type} runtime "
                "cannot run Compose services; use a supported runtime"
            )
        return _ContainerDeployment()
    workspace = plan.machine(WORKSPACE)
    assert workspace is not None
    own = {m.nodes[0]: m.runtime for m in plan.machines if m is not workspace}
    packed = [name for name in workspace.nodes if name != WORKSPACE]
    if spec is not None or packed:
        if own:
            raise PlacementError(
                f"nodes {sorted(own)} need machines of their own, but the workspace's "
                "machine runs Compose services, which cannot join it on a private "
                "network yet; give those nodes the workspace's policy so they join its "
                "Compose project"
            )
        if declared := [name for name in packed if name not in topology.compose]:
            if spec is not None:
                raise PlacementError(
                    f"nodes {declared} would share the workspace's machine with the "
                    "task's Compose project; declare them as services of that project"
                )
            return _ContainerDeployment(compose_project(topology, plan))
        return _ContainerDeployment()
    if not own:
        return _ColocatedDeployment()
    processes = {
        name: process
        for name in own
        if (process := topology.node(name).process) is not None
    }
    if imaged := sorted(n for n, p in processes.items() if p.command is None):
        raise PlacementError(
            f"nodes {imaged} run on machines of their own, which do not run their "
            "image's own command; set `process.command`"
        )
    return _PrivateNetworkDeployment(
        own,
        [(c.source, c.target) for c in plan.connections if c.via == "private network"],
        processes,
        served=set(topology.mcp_servers.values()),
    )
