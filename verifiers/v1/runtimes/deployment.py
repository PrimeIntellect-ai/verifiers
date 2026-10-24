"""Attempt-scoped target ownership, independent of placement."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

from verifiers.v1.configs.deployment import DeploymentConfig
from verifiers.v1.runtimes import (
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
        spec: ComposeSpec | None = None,
        tool_vms: Mapping[str, PrimeConfig] | None = None,
        env: dict[str, str] | None = None,
        config: DeploymentConfig | None = None,
        borrowed_runtime: Runtime | None = None,
    ):
        self.backend = backend
        self.spec = spec
        self.tool_vms = dict(tool_vms or {})
        """Tool servers on Prime runtimes of their own, by server name, which join the
        workspace over the rollout's private network."""
        self.env = dict(env or {})
        self.config = config or DeploymentConfig()
        self.borrowed_runtime = borrowed_runtime
        from verifiers.v1.runtimes.strategies import _deployment_strategy

        self._strategy = _deployment_strategy(self.config, backend, spec, self.tool_vms)
        self.target_backends: dict[str, RuntimeBackend] = {}
        self.hosts: dict[str, str] = {}
        """The hostname the workspace reaches a named target by, where it is not the
        target's name (a tool server VM's LAN name)."""
        self.harness_backend: RuntimeBackend | None = None
        self.execution_backend: RuntimeBackend | None = None
        self._stack = AsyncExitStack()
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
        try:
            await self._strategy.start(self)
            if self.harness_backend is None or self.execution_backend is None:
                raise ValueError(
                    "deployment strategy must bind harness and execution roles"
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
        """Access a named runtime, such as a Compose service, through an execution view."""
        return (
            self.workspace
            if self.target_backends[name] is self.execution_backend
            else self.target_backends[name].bind()
        )

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

    async def stop_execution(self) -> None:
        await self._strategy.stop_execution(self)

    async def prepare_execution(
        self, routes: list[str], *, state_route: str | None = None
    ) -> None:
        """Apply each target's execution policy: the harness's keeps `routes`, and a tool
        server's own VM keeps only `state_route`, its state channel."""
        await self._strategy.prepare_execution(self, routes, state_route)

    async def close(self) -> None:
        async def release():
            try:
                await self._strategy.close(self)
            finally:
                await self._stack.aclose()

        if self._close_task is None:
            self._close_task = asyncio.create_task(release())
        await run_shielded(self._close_task)
