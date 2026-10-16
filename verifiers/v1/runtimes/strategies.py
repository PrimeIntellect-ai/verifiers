"""Colocated execution and composed task services."""

import logging
from contextlib import AsyncExitStack

from verifiers.v1.runtimes import DockerConfig, ModalConfig, PrimeConfig
from verifiers.v1.runtimes.compose import compose_services
from verifiers.v1.runtimes.deployment import _DeploymentStrategy
from verifiers.v1.runtimes.lan import own_host_programs, own_host_services

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
        self._stop_main = None
        if d.spec.own_host:
            # Every service runs on a VM of its own; the workspace is the agent's VM.
            programs = own_host_programs(d.spec, d.env)
            d.backend.env = d.env
            await d.own(d.backend)
            services = await self._services.enter_async_context(
                own_host_services(d.backend, programs)
            )
            d.target_backends = {d.spec.workspace: d.backend, **services}
            d.execution_backend = d.harness_backend = d.backend
            return
        services, self._stop_main, _ = await self._services.enter_async_context(
            compose_services(
                d.backend.config, d.spec, d.env, trust_compose=d.config.trust_compose
            )
        )
        d.target_backends = dict(services)
        d.execution_backend = d.harness_backend = services[d.spec.workspace]

    async def stop_execution(self, d):
        if self._stop_main is not None:
            await self._stop_main()

    async def prepare_verifier(self, deployment):
        # Evidence is collected; release the solver before the verifier clock starts.
        try:
            await self._services.aclose()
            if self._stop_main is None:
                await deployment.execution_backend.stop()
        except Exception:
            logger.warning(
                "solver services teardown failed before grading", exc_info=True
            )


def _deployment_strategy(config, runtime_config, spec):
    if spec is not None:
        if not isinstance(runtime_config, (DockerConfig, PrimeConfig, ModalConfig)):
            raise ValueError("Compose requires Docker, Prime, or Modal VM runtime")
        return _ContainerDeployment()
    return _ColocatedDeployment()
