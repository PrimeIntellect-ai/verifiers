"""Concrete placement strategies; provider and topology assumptions stay here."""

from __future__ import annotations

import contextlib
import json
import logging
from contextlib import AsyncExitStack

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    DockerConfig,
    PrimeConfig,
)
from verifiers.v1.runtimes.compose import compose_services
from verifiers.v1.runtimes.deployment import Deployment, _DeploymentStrategy

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
    """Containers share a provider VM or use the local Docker daemon."""

    def verifier_config(self, deployment):
        return deployment.backend.config

    def resolve_execution(self, harness, execution):
        if not isinstance(execution, DockerConfig) or not isinstance(
            harness, PrimeConfig
        ):
            raise TypeError(
                "split execution requires a Docker workspace inside a Prime VM"
            )

        if execution.cpu is None:
            execution = execution.model_copy(update={"cpu": (harness.cpu or 2) * 0.75})
        if execution.memory is None:
            execution = execution.model_copy(
                update={"memory": (harness.memory or 8) * 0.75}
            )
        assert execution.memory is not None
        if harness.memory is not None and execution.memory >= harness.memory:
            raise ValueError(
                "task memory must leave space for the harness in the outer VM"
            )
        if (
            execution.cpu is not None
            and harness.cpu is not None
            and execution.cpu > harness.cpu
        ):
            raise ValueError("task CPUs exceed the outer VM allocation")
        if execution.gpu:
            raise ValueError("split task runtimes do not yet support GPU limits")
        if execution.disk is not None and execution.disk >= harness.disk:
            raise ValueError("execution disk must leave storage for the harness")
        if execution.network_restricted and execution.allow:
            raise ValueError(
                "split execution supports unrestricted task networking or an empty allowlist"
            )
        return execution

    async def start(self, deployment: Deployment) -> None:
        d = deployment
        self._stop_main = None
        self._grading_networks = []
        self._isolated_networks = []
        if d.spec is not None:
            if d.borrowed_runtime is not None:
                raise ValueError("a composed deployment cannot borrow a single target")
            self._services = await d.enter(AsyncExitStack())
            (
                services,
                self._stop_main,
                host_backend,
            ) = await self._services.enter_async_context(
                compose_services(
                    d.backend.config,
                    d.spec,
                    d.env,
                    trust_compose=d.config.trust_compose,
                    execution=d.execution_config,
                )
            )
            d.target_backends = dict(services)
            d.execution_backend = services[d.spec.workspace]
            d.harness_backend = d.execution_backend
            if d.execution_config is not None:
                assert host_backend is not None
                d.harness_backend = host_backend
            return
        if d.borrowed_runtime is not None:
            d.execution_backend = d.harness_backend = d.borrowed_runtime.backend
        elif d.execution_config is None:
            d.backend.env = d.env
            await d.own(d.backend)
            d.execution_backend = d.harness_backend = d.backend
        else:
            from verifiers.v1.runtimes.container_target import ContainerTarget
            from verifiers.v1.runtimes.docker_host import prepare_docker_host

            d.harness_backend = d.backend
            await d.own(d.harness_backend)
            await prepare_docker_host(
                d.harness_backend,
                d.execution_config.disk or d.backend.config.disk * 0.6,
            )
            d.execution_backend = ContainerTarget(
                d.execution_config,
                host_backend=d.harness_backend,
                name=f"{d.backend.name}-task",
            )
            d.execution_backend.env = d.env
            await d.own(d.execution_backend)
        d.target_backends = {"main": d.execution_backend}

    async def prepare_execution(self, d: Deployment, routes: list[str]) -> None:
        assert d.execution_backend is not None and d.harness_backend is not None
        await d.harness_backend.prepare_execution(routes)
        if not d.split:
            return
        if d.spec is None or not d.execution_config.network_restricted:
            await d.execution_backend.prepare_execution([])
            return
        # Replace each task network with an internal network, retaining service
        # membership and DNS aliases. The harness is outside the task networks.
        import uuid

        networks = {}
        for name, target in d.target_backends.items():
            result = await target._run_host(
                "docker",
                "inspect",
                "--format",
                "{{json .NetworkSettings.Networks}}",
                target._container,
            )
            if result.exit_code:
                raise SandboxError("cannot inspect task network membership")
            for network, settings in json.loads(result.stdout).items():
                networks.setdefault(network, []).append(
                    (name, target, settings.get("Aliases") or [])
                )
        for original, members in networks.items():
            isolated = f"vf-internal-{uuid.uuid4().hex}"
            result = await d.execution_backend._run_host(
                "docker", "network", "create", "--internal", isolated
            )
            if result.exit_code:
                raise SandboxError(f"cannot isolate task network: {result.stderr}")
            containers = []
            self._isolated_networks.append((isolated, containers))
            for name, target, aliases in members:
                args = ["docker", "network", "connect", "--alias", name]
                for alias in aliases:
                    args += ["--alias", alias]
                result = await target._run_host(*args, isolated, target._container)
                if result.exit_code:
                    raise SandboxError(
                        f"cannot connect isolated service: {result.stderr}"
                    )
                containers.append(target._container)
                self._grading_networks.append(
                    (target, original, isolated, name, aliases)
                )
                result = await target._run_host(
                    "docker", "network", "disconnect", original, target._container
                )
                if result.exit_code:
                    raise SandboxError(f"cannot remove task egress: {result.stderr}")

    async def prepare_grading(self, d: Deployment) -> None:
        """Restore task networking for a trusted shared-workspace verifier."""
        for target, original, isolated, name, aliases in self._grading_networks:
            args = ["docker", "network", "connect", "--alias", name]
            for alias in aliases:
                args += ["--alias", alias]
            result = await target._run_host(*args, original, target._container)
            if result.exit_code:
                raise SandboxError(f"cannot restore verifier network: {result.stderr}")
            result = await target._run_host(
                "docker", "network", "disconnect", isolated, target._container
            )
            if result.exit_code:
                raise SandboxError(f"cannot release isolated network: {result.stderr}")
        self._grading_networks.clear()
        if d.split:
            await d.harness_backend.prepare_execution(None)
        await d.execution_backend.prepare_execution(None)

    async def stop_execution(self, d):
        if self._stop_main is not None:
            await self._stop_main()

    async def close(self, d):
        if d.execution_backend is None:
            return
        for network, containers in self._isolated_networks:
            for container in containers:
                with contextlib.suppress(Exception):
                    await d.execution_backend._run_host(
                        "docker", "network", "disconnect", "--force", network, container
                    )
            with contextlib.suppress(Exception):
                await d.execution_backend._run_host("docker", "network", "rm", network)
        self._isolated_networks.clear()

    async def prepare_verifier(self, deployment):
        # Evidence is collected; release the solver before the verifier clock starts.
        if not deployment.split:
            try:
                await self._services.aclose()
            except Exception:
                logger.warning(
                    "solver services teardown failed before grading", exc_info=True
                )


def _deployment_strategy(config, runtime_config, spec):
    if config.mode == "split":
        if not isinstance(runtime_config, PrimeConfig):
            raise ValueError("split deployment is only supported by Prime VMs")
        return _ContainerDeployment()
    if spec is not None:
        from verifiers.v1.runtimes import ModalConfig

        if not isinstance(runtime_config, (DockerConfig, PrimeConfig, ModalConfig)):
            raise ValueError("Compose requires Docker, Prime, or Modal VM runtime")
        return _ContainerDeployment()
    return _ColocatedDeployment()
