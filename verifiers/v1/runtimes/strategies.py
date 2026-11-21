"""Split execution, local Compose, and native Prime service VMs."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager

from verifiers.v1.configs.deployment import DOCKER_HARNESS_IMAGE, HARNESS_IMAGE
from verifiers.v1.configs.runtime import DockerConfig, NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import DockerBackend, PrimeBackend, PrimeConfig, make_backend
from verifiers.v1.runtimes.compose import compose_services
from verifiers.v1.runtimes.deployment import ServicesSpec, _DeploymentStrategy
from verifiers.v1.runtimes.private_network import (
    keepalive,
    lan_host,
    private_network,
)
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)


def _make_harness(d):
    """Place the harness beside the workspace using the harness image and its own resources."""
    workspace = d.execution_backend.config
    policy = d.execution_network
    if isinstance(workspace, PrimeConfig):
        config = PrimeConfig(
            image=HARNESS_IMAGE,
            cpu=2,
            memory=4,
            workdir=workspace.workdir,
            region=workspace.region,
            labels=workspace.labels,
            creates_per_min=workspace.creates_per_min,
            allow=policy.allow,
            block=policy.block,
        )
    elif isinstance(workspace, DockerConfig):
        config = DockerConfig(
            image=DOCKER_HARNESS_IMAGE,
            cpu=2,
            memory=4,
            workdir=workspace.workdir,
            allow=policy.allow,
            block=policy.block,
        )
    else:
        config = workspace
    return make_backend(config, name=f"{d.execution_backend.name}-harness")


class _WorkspaceOnlyDeployment(_DeploymentStrategy):
    """Model-free workspace deployment for `vf-validate` and `grade_submission`."""

    async def start(self, d):
        if d.borrowed_runtime is not None:
            d.execution_backend = d.borrowed_runtime.backend
        else:
            d.backend.env = d.env
            await d.own(d.backend)
            d.execution_backend = d.backend
        d.target_backends = {"main": d.execution_backend}

    async def prepare_verifier(self, deployment):
        if deployment.borrowed_runtime is None:
            try:
                await deployment.execution_backend.stop()
            except Exception:
                logger.warning(
                    "Failed to tear down the solver before grading", exc_info=True
                )


class _SplitDeployment(_DeploymentStrategy):
    """One workspace and one dedicated harness, connected through the evaluator."""

    async def start(self, d):
        d.execution_backend = (
            d.borrowed_runtime.backend if d.borrowed_runtime is not None else d.backend
        )
        if (
            d.borrowed_runtime is not None
            and isinstance(d.execution_backend, DockerBackend)
            and not d.execution_backend._restricted
        ):
            raise SandboxError(
                "borrowed Docker workspace cannot enforce the agent network policy; provision it with agent.provision() or provision_runtime()"
            )
        d.harness_backend = _make_harness(d)
        if d.borrowed_runtime is not None:
            await d.own(d.harness_backend)
        else:
            d.backend.env = d.env
            try:
                async with asyncio.TaskGroup() as group:
                    group.create_task(d.own(d.backend))
                    group.create_task(d.own(d.harness_backend))
            except ExceptionGroup as failed:
                raise failed.exceptions[0]
        if isinstance(d.execution_backend, PrimeBackend):
            await d.enter(keepalive(d.execution_backend))
        d.target_backends = {"main": d.execution_backend}

    async def prepare_execution(self, d, routes, state_route=None):
        await d.harness_backend.prepare_execution(routes)
        if isinstance(d.execution_network, NetworkPolicyConfig):
            await _prepare_split_workspace(d)

    async def stop_execution(self, d):
        owned = [d.harness_backend]
        if d.borrowed_runtime is None:
            owned.append(d.execution_backend)
        await asyncio.gather(*(backend.stop() for backend in owned))

    async def prepare_verifier(self, d):
        try:
            await self.stop_execution(d)
        except Exception:
            logger.warning(
                "Failed to tear down solver runtimes before grading", exc_info=True
            )


async def _prepare_split_workspace(d):
    await d.enter(
        d.execution_backend.workspace_network(
            d.execution_network,
            d.control_routes,
            borrowed=d.borrowed_runtime is not None,
        )
    )


class _ContainerDeployment(_DeploymentStrategy):
    async def start(self, d):
        if d.borrowed_runtime is not None:
            raise ValueError("a composed deployment cannot borrow a single target")
        self._services = await d.enter(AsyncExitStack())
        services, self._stop_main = await self._services.enter_async_context(
            compose_services(
                d.backend.config, d.spec, d.env, trust_compose=d.config.trust_compose
            )
        )
        d.target_backends = dict(services)
        d.execution_backend = services[d.spec.workspace]
        if d.for_agent:
            d.harness_backend = await d.own(_make_harness(d))

    async def prepare_execution(self, d, routes, state_route=None):
        if d.harness_backend is not None:
            await d.harness_backend.prepare_execution(routes)
        if d.execution_backend.network_restricted:
            await _prepare_split_workspace(d)

    async def stop_execution(self, d):
        if d.harness_backend is not None:
            await d.harness_backend.stop()
        await self._stop_main()

    async def prepare_verifier(self, deployment):
        try:
            await self._services.aclose()
        except Exception:
            logger.warning(
                "Failed to tear down solver services before grading", exc_info=True
            )
        if deployment.harness_backend is not None:
            try:
                await deployment.harness_backend.stop()
            except Exception:
                logger.warning(
                    "Failed to tear down the harness before grading", exc_info=True
                )


class _PrivateNetworkDeployment(_DeploymentStrategy):
    _networks: dict[str, list[str]] | None = None
    _internal: frozenset[str] = frozenset()

    async def start(self, d):
        if d.borrowed_runtime is not None:
            raise ValueError(
                "a deployment with service VMs cannot borrow a single target"
            )
        native = isinstance(d.spec, ServicesSpec)
        members = d.spec.services if native else d.tool_vms
        d.execution_backend = d.backend
        d.backend.env = d.env
        if d.for_agent:
            d.harness_backend = _make_harness(d)
        self._services = await d.enter(AsyncExitStack())
        vms = await self._services.enter_async_context(
            private_network(
                d.backend,
                members,
                interconnect=native,
                networks=self._networks,
                harness=d.harness_backend,
                harness_peers=d.tool_services,
            )
        )
        d.target_backends = {"main": d.backend, **vms}
        d.hosts = {name: lan_host(backend) for name, backend in vms.items()}
        if native:
            d.hosts["main"] = lan_host(d.backend)

    async def prepare_execution(self, d, routes, state_route=None):
        calls = []
        for name, backend in d.target_backends.items():
            if name == "main":
                if d.harness_backend is not None:
                    calls.append(_prepare_split_workspace(d))
                else:
                    calls.append(backend.prepare_execution(routes))
            else:
                own = (
                    [backend.host_url(state_route)]
                    if state_route and name not in self._internal
                    else []
                )
                calls.append(backend.prepare_execution(own))
        if d.harness_backend is not None:
            calls.append(d.harness_backend.prepare_execution(routes))
        await asyncio.gather(*calls)

    async def stop_execution(self, d):
        vms = [d.execution_backend]
        if d.harness_backend is not None:
            vms.append(d.harness_backend)
        await asyncio.gather(*(vm.stop() for vm in vms))

    async def prepare_verifier(self, deployment):
        try:
            await self._services.aclose()
        except Exception:
            logger.warning(
                "Failed to tear down service VMs before grading", exc_info=True
            )


class _NativeComposeDeployment(_PrivateNetworkDeployment):
    _processes = None

    @asynccontextmanager
    async def verifier(self, deployment, config, env):
        main = self._processes.services["main"]
        if isinstance(config, PrimeConfig) and config.image == main.config.image:
            # Dockerfile ENV cannot preserve multiline values; pass image defaults
            # directly, excluding the solver's Compose and rollout overrides.
            env = main.image_env | env
        async with super().verifier(deployment, config, env) as backend:
            yield backend

    async def start(self, d):
        from verifiers.v1.runtimes.native_compose import NativeProcesses, prepare_native

        if d.borrowed_runtime is not None:
            raise ValueError("a composed deployment cannot borrow a single target")
        services = await prepare_native(
            d.backend.config.model_copy(
                update={"workdir": d.backend._workdir_override}
            ),
            d.spec,
            d.env,
        )
        main = services["main"]
        d.backend.config = main.config
        d.execution_network = main.config
        for field, value in main.config.model_dump().items():
            setattr(d.backend.info, field, value)
        d.env = main.env
        self._networks = {}
        for name, service in services.items():
            # Default-bridge peers share IP connectivity without Compose aliases.
            networks = [
                f"project:{network}" for network in service.definition["networks"]
            ]
            for network in networks or ["bridge"]:
                self._networks.setdefault(network, []).append(name)
        self._internal = frozenset(
            name for name, service in services.items() if service.internal
        )
        d.spec = ServicesSpec(
            {
                name: service.config
                for name, service in services.items()
                if name != "main"
            }
        )
        await super().start(d)
        self._processes = NativeProcesses(services)
        await self._processes.start(d)

    async def stop_execution(self, d):
        await self._processes.stop(d, ["main"])
        await super().stop_execution(d)

    async def prepare_verifier(self, deployment):
        if self._processes is not None:
            await self._processes.stop(
                deployment, reversed(list(self._processes.processes))
            )
        await super().prepare_verifier(deployment)

    async def close(self, deployment):
        if self._processes is not None:
            await self._processes.stop(
                deployment, reversed(list(self._processes.processes))
            )
        await super().close(deployment)


def _deployment_strategy(
    config, backend, spec, tool_vms=None, *, has_harness: bool = False
):
    if isinstance(spec, ServicesSpec):
        if not isinstance(backend, PrimeBackend):
            raise TypeError("Native service VMs require a Prime workspace")
        if "main" in spec.services:
            raise ValueError(
                "'main' is the task workspace; name its services separately"
            )
        if any(
            not isinstance(runtime, PrimeConfig) for runtime in spec.services.values()
        ):
            raise ValueError("Native services require PrimeConfig runtimes")
        return _PrivateNetworkDeployment()
    if spec is not None:
        if isinstance(backend, PrimeBackend):
            return _NativeComposeDeployment()
        if not backend.SUPPORTS_COMPOSE:
            raise ValueError(
                f"the task deploys Compose services, but the {backend.type} runtime "
                "cannot run Compose services; use ServicesSpec on Prime or local Docker"
            )
        return _ContainerDeployment()
    if isinstance(backend, PrimeBackend) and tool_vms:
        return _PrivateNetworkDeployment()
    if has_harness:
        return _SplitDeployment()
    return _WorkspaceOnlyDeployment()
