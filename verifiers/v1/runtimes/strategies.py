"""Colocated execution, composed task services, and tool servers on VMs of their own."""

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial

from verifiers.v1.runtimes.compose import compose_services
from verifiers.v1.runtimes.deployment import _DeploymentStrategy
from verifiers.v1.runtimes.private_network import lan_host, private_network

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
    async def start(self, d):
        if d.borrowed_runtime is not None:
            raise ValueError("a composed deployment cannot borrow a single target")
        self._services = await d.enter(AsyncExitStack())
        (
            services,
            self._stop_main,
            host,
            built_image,
        ) = await self._services.enter_async_context(
            compose_services(
                d.backend.config, d.spec, d.env, trust_compose=d.config.trust_compose
            )
        )
        d.target_backends = dict(services)
        d.execution_backend = d.harness_backend = services[d.spec.workspace]
        self._remote_built_image = built_image if host is not None else None

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

    @asynccontextmanager
    async def verifier(self, deployment, config, env):
        if self._remote_built_image is None or config.image != self._remote_built_image:
            async with super().verifier(deployment, config, env) as backend:
                yield backend
            return
        # Rebuild from trusted task inputs on a fresh host: the solver's Docker tag
        # is local to its released VM, and its modified container must never travel.
        async with compose_services(
            config,
            deployment.spec,
            env,
            trust_compose=deployment.config.trust_compose,
        ) as (services, _, _, _):
            yield services[deployment.spec.workspace]


class _PrivateNetworkDeployment(_DeploymentStrategy):
    async def start(self, d):
        if d.borrowed_runtime is not None:
            raise ValueError(
                "a deployment with tool servers on VMs of their own cannot borrow a "
                "single target"
            )
        d.backend.env = d.env
        self._services = await d.enter(AsyncExitStack())
        vms = await self._services.enter_async_context(
            private_network(d.backend, d.tool_vms, partial(d.own, d.backend))
        )
        d.target_backends = {"main": d.backend, **vms}
        d.hosts = {name: lan_host(backend) for name, backend in vms.items()}
        d.execution_backend = d.harness_backend = d.backend
        self._service_backends = list(vms.values())

    async def prepare_execution(self, d, routes, state_route=None):
        # Each VM enforces its own policy. A tool server's VM keeps only its state
        # channel, reached as the server reaches it; LAN traffic is unaffected by either.
        def own(backend):
            return [backend.host_url(state_route)] if state_route else []

        await asyncio.gather(
            d.harness_backend.prepare_execution(routes),
            *(
                backend.prepare_execution(own(backend))
                for backend in self._service_backends
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


def _deployment_strategy(config, backend, spec, tool_vms=None):
    if spec is not None:
        if not backend.SUPPORTS_COMPOSE:
            raise ValueError(
                f"the task deploys Compose services, but the {backend.type} runtime "
                "cannot run Compose services; use a supported runtime"
            )
        return _ContainerDeployment()
    if tool_vms:
        return _PrivateNetworkDeployment()
    return _ColocatedDeployment()
