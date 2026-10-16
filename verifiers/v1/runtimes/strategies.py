"""Colocated execution, composed task services, and task services on VMs of their own."""

import logging
from contextlib import AsyncExitStack
from functools import partial

from verifiers.v1.runtimes import DockerConfig, ModalConfig, PrimeConfig
from verifiers.v1.runtimes.compose import compose_services
from verifiers.v1.runtimes.deployment import _DeploymentStrategy
from verifiers.v1.runtimes.service_vms import (
    WORKSPACE,
    ServiceVMSpec,
    lan_host,
    service_vms,
)

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
        services, self._stop_main, _ = await self._services.enter_async_context(
            compose_services(
                d.backend.config, d.spec, d.env, trust_compose=d.config.trust_compose
            )
        )
        d.target_backends = dict(services)
        d.execution_backend = d.harness_backend = services[d.spec.workspace]

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


class _ServiceVMDeployment(_DeploymentStrategy):
    async def start(self, d):
        if d.borrowed_runtime is not None:
            raise ValueError(
                "a deployment with service VMs cannot borrow a single target"
            )
        d.backend.env = d.env
        self._services = await d.enter(AsyncExitStack())
        services = await self._services.enter_async_context(
            service_vms(d.backend, d.spec.services, partial(d.own, d.backend))
        )
        d.target_backends = {WORKSPACE: d.backend, **services}
        d.hosts = {name: lan_host(backend) for name, backend in services.items()}
        d.execution_backend = d.harness_backend = d.backend

    async def prepare_verifier(self, deployment):
        # Evidence is collected; release the solver's VMs before the verifier clock starts.
        try:
            await self._services.aclose()
            await deployment.execution_backend.stop()
        except Exception:
            logger.warning("service VMs teardown failed before grading", exc_info=True)


def _deployment_strategy(config, runtime_config, spec):
    if isinstance(spec, ServiceVMSpec):
        if isinstance(runtime_config, PrimeConfig):
            return _ServiceVMDeployment()
        raise ValueError("service VMs require the Prime runtime")
    if spec is not None:
        if not isinstance(runtime_config, (DockerConfig, PrimeConfig, ModalConfig)):
            raise ValueError("Compose requires Docker, Prime, or Modal VM runtime")
        return _ContainerDeployment()
    return _ColocatedDeployment()
