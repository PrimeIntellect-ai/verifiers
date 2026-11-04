"""Attempt-scoped target ownership, independent of placement."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from functools import partial

from verifiers.v1.configs.deployment import DeploymentConfig
from verifiers.v1.errors import ToolsetError
from verifiers.v1.runtimes import (
    PrimeBackend,
    PrimeConfig,
    Runtime,
    RuntimeBackend,
    RuntimeConfig,
    provision_backend,
)
from verifiers.v1.runtimes.base import (
    register,
)
from verifiers.v1.runtimes.compose import ComposeSpec
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServicesSpec:
    """Named Prime VMs beside the task's `main` workspace, on one private LAN.

    Each config selects a service's image and resources. Task setup starts its
    processes through `deployment.target(name)` and waits for them to be ready.
    `deployment.hosts` maps every target, including `main`, to its LAN hostname.
    """

    services: dict[str, PrimeConfig]


class _DeploymentStrategy:
    """Placement, connectivity and lifecycle hooks for one deployment attempt."""

    async def start(self, deployment: Deployment) -> None:
        raise NotImplementedError

    async def prepare_execution(
        self, deployment: Deployment, routes: list[str], state_route: str | None = None
    ) -> None:
        await deployment.harness_backend.prepare_execution(routes)

    async def stop_execution(self, deployment: Deployment) -> None:
        return

    async def close(self, deployment: Deployment) -> None:
        return

    async def prepare_verifier(self, deployment: Deployment) -> None:
        return

    @asynccontextmanager
    async def verifier(
        self, deployment: Deployment, config: RuntimeConfig, env: dict[str, str]
    ) -> AsyncIterator[RuntimeBackend]:
        async with provision_backend(config, env=env) as backend:
            yield backend


class Deployment:
    """Coordinate the execution resources for one rollout attempt.

    Provisions and connects the harness, task workspace, and supporting services,
    and cleans up resources it acquires, including fresh verification environments.
    Runtimes implement resource operations; the rollout controls agent execution.
    """

    def __init__(
        self,
        backend: RuntimeBackend,
        *,
        spec: ComposeSpec | ServicesSpec | None = None,
        tool_vms: Mapping[str, PrimeConfig] | None = None,
        shared_networks: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
        config: DeploymentConfig | None = None,
        borrowed_runtime: Runtime | None = None,
    ):
        self.backend = backend
        self.spec = spec
        self.tool_vms = dict(tool_vms or {})
        """Tool servers on Prime runtimes of their own, by server name, which join the
        workspace over the rollout's private network."""
        self.shared_networks = shared_networks
        self.env = dict(env or {})
        self.config = config or DeploymentConfig()
        self.borrowed_runtime = borrowed_runtime
        from verifiers.v1.runtimes.strategies import _deployment_strategy

        self._strategy = _deployment_strategy(self.config, backend, spec)
        self.target_backends: dict[str, RuntimeBackend] = {}
        self.hosts: dict[str, str] = {}
        """LAN hostnames of native services or tool servers on their own VMs."""
        self.harness_backend: RuntimeBackend | None = None
        self.execution_backend: RuntimeBackend | None = None
        self._stack = AsyncExitStack()
        self._tools = AsyncExitStack()
        self._owned: set[int] = set()
        self._close_task: asyncio.Task[None] | None = None

    async def own(self, backend: RuntimeBackend) -> RuntimeBackend:
        """Acquire a target; register cleanup before startup can partially fail."""
        if id(backend) in self._owned:
            raise ValueError("runtime already owned by this deployment")
        self._owned.add(id(backend))
        register(backend)
        self._stack.push_async_callback(backend.stop)
        await backend.start()
        return backend

    async def enter(self, resource):
        """Own a provider resource such as a network or composed service group."""
        return await self._stack.enter_async_context(resource)

    async def start(self) -> None:
        from verifiers.v1.runtimes.private_network import (
            WORKSPACE_TAG,
            lan_host,
            private_network,
        )

        try:
            await self.enter(self._tools)
            tools = {}
            if self.tool_vms and self.spec is None and self.borrowed_runtime is None:
                # Fresh VMs can join their LAN at creation and boot together.
                tools = await self._tools.enter_async_context(
                    private_network(
                        self.backend,
                        self.tool_vms,
                        partial(self._strategy.start, self),
                    )
                )
            else:
                await self._strategy.start(self)
            if self.harness_backend is None or self.execution_backend is None:
                raise ValueError(
                    "deployment strategy must bind harness and execution roles"
                )
            if isinstance(self.harness_backend, PrimeBackend):
                for network_id in self.shared_networks:
                    await self._tools.enter_async_context(
                        self.harness_backend.join_network(network_id, [WORKSPACE_TAG])
                    )
            if self.tool_vms:
                if overlap := self.tool_vms.keys() & self.target_backends.keys():
                    raise ValueError(
                        f"tool servers collide with task services: {sorted(overlap)}"
                    )
                tools = tools or await self._tools.enter_async_context(
                    private_network(self.execution_backend, self.tool_vms)
                )
                self.target_backends.update(tools)
                self.hosts.update(
                    {name: lan_host(backend) for name, backend in tools.items()}
                )
        except BaseException:
            try:
                await self.close()
            except Exception:
                logger.warning(
                    "deployment teardown failed after startup error", exc_info=True
                )
            raise

    @property
    def execution_name(self) -> str:
        return next(
            (
                name
                for name, target_backend in self.target_backends.items()
                if target_backend is self.execution_backend
            ),
            "main",
        )

    @property
    def workspace(self) -> Runtime:
        """Runtime view where task commands execute and task files live.

        This is the single runtime for ordinary tasks, or the selected Compose
        service; it names an execution role rather than a separate resource type.
        """
        if self.borrowed_runtime is not None:
            return self.borrowed_runtime.with_env(self.env)
        return self.execution_backend.bind(env=self.env)

    def target(self, name: str) -> Runtime:
        """Access a named service or workspace through an execution view."""
        return (
            self.workspace
            if self.target_backends[name] is self.execution_backend
            else self.target_backends[name].bind()
        )

    def tool_target(self, toolset) -> tuple[Runtime | None, str | None]:
        """Resolve an explicit service or an independently provisioned tool VM."""
        name = toolset.config.service
        if name is not None:
            if (
                self.spec is None
                or name not in self.target_backends
                or name in self.tool_vms
            ):
                raise ToolsetError(
                    f"server {toolset.server_name!r} needs a task whose services define {name!r}"
                )
        elif toolset.server_name in self.tool_vms:
            name = toolset.server_name
        else:
            return None, None
        runtime = self.target(name)
        if runtime.backend is self.execution_backend:
            raise ToolsetError(
                f"server {toolset.server_name!r} needs a service separate from the workspace"
            )
        if runtime.is_local:
            raise ToolsetError(
                f"server {toolset.server_name!r} needs a native Prime service VM"
            )
        return runtime, self.hosts.get(name, name)

    @asynccontextmanager
    async def verifier(
        self, config: RuntimeConfig, env: dict[str, str]
    ) -> AsyncIterator[Runtime]:
        # Parent ownership is registered before acquisition, including cancellation.
        stack = AsyncExitStack()
        self._stack.push_async_callback(stack.aclose)
        async with stack:
            backend = await stack.enter_async_context(
                self._strategy.verifier(self, config, env)
            )
            yield backend.bind()

    async def prepare_verifier(self) -> None:
        """Release solver resources after collection, outside verifier budgets.

        Finish cleanup even if the caller is cancelled, then propagate cancellation.
        """
        await run_shielded(self._strategy.prepare_verifier(self))
        await run_shielded(self._tools.aclose())

    async def stop_execution(self) -> None:
        await self._strategy.stop_execution(self)
        if (
            self.tool_vms
            and self.borrowed_runtime is None
            and not self.execution_backend.stopped
        ):
            # Stop agent descendants before collecting evidence from dedicated tools.
            await self.execution_backend.stop()

    async def prepare_execution(
        self, routes: list[str], *, state_route: str | None = None
    ) -> None:
        """Apply each target's execution policy: the harness's keeps `routes`, and a tool
        server's own VM keeps only `state_route`, its state channel."""
        await self._strategy.prepare_execution(self, routes, state_route)
        await asyncio.gather(
            *(
                self.target_backends[name].prepare_execution(
                    [state_route] if state_route else []
                )
                for name in self.tool_vms
            )
        )

    async def close(self) -> None:
        async def release():
            try:
                await self._strategy.close(self)
            finally:
                await self._stack.aclose()

        if self._close_task is None:
            self._close_task = asyncio.create_task(release())
        await run_shielded(self._close_task)
