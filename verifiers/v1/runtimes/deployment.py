"""Attempt-scoped target ownership and supervision, independent of placement."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field
from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    make_runtime,
    provision_runtime,
)
from verifiers.v1.runtimes.base import (
    ProgramResult,
    RuntimeProcess,
    TargetStatus,
    register,
)
from verifiers.v1.runtimes.compose import ComposeSpec
from verifiers.v1.utils.aio import run_shielded


class ExecutionTarget(Protocol):
    """Process and file access, independent of provisioning ownership."""

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult: ...
    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess: ...
    async def read(self, path: str, max_bytes: int | None = None) -> bytes: ...
    async def write(self, path: str, data: bytes) -> None: ...
    async def execution_status(self) -> TargetStatus: ...


@dataclass(frozen=True)
class ExecutionConnection:
    """Launch transport bound to a source and destination, not a provider topology.

    A strategy may supply a local exec command or an authenticated remote worker
    command. Credential installation, readiness and teardown belong to the strategy.
    """

    source: Runtime
    target: Runtime
    launch: Callable[[list[str]], list[str]]

    def command(self, argv: list[str]) -> list[str]:
        return self.launch(argv)


class DeploymentConfig(BaseConfig):
    strategy: str = "auto"
    """Placement strategy: auto, shared, nested, independent, or a registered strategy."""
    trust_compose: bool = False
    """Permit task-authored Compose privileges and bind mounts on local Docker."""


class ExecutionFailurePolicy(BaseConfig):
    workspace_loss: Literal["error", "zero"] = "error"
    """How to score confirmed workspace loss during execution, with unknown cause."""
    agent_timeout: Literal["error", "zero", "grade"] = "grade"
    """Whole-agent deadline policy; includes model latency, unlike command timeouts."""


class Termination(BaseModel):
    kind: Literal["target_lost", "timeout"]
    target: str = "main"
    attribution: Literal["agent", "infra", "unknown"] = "unknown"
    cause: str = "unknown"
    evidence: dict = Field(default_factory=dict)
    policy: str
    valid_sample: bool
    reward: float | None = None
    rule: str = "execution-v1"


class TargetLost(SandboxError):
    def __init__(self, status: TargetStatus):
        super().__init__(f"workspace terminated during execution ({status.cause})")
        self.status = status


class DeploymentStrategy:
    """Placement, connectivity and lifecycle hooks for one deployment attempt."""

    def resolve_execution(
        self, harness: RuntimeConfig, execution: RuntimeConfig
    ) -> RuntimeConfig:
        return execution

    async def start(self, deployment: Deployment) -> None:
        raise NotImplementedError

    async def connect(
        self, deployment: Deployment, source: Runtime, target: Runtime
    ) -> ExecutionConnection:
        try:
            target.execution_command(source, [])
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"{type(self).__name__} has no transport from {source.type} to "
                f"{target.type}; use a deployment strategy that provides this connection"
            ) from exc
        return ExecutionConnection(
            source, target, lambda argv: target.execution_command(source, argv)
        )

    async def prepare_execution(
        self, deployment: Deployment, routes: list[str]
    ) -> None:
        await deployment.harness.prepare_execution(routes)
        if deployment.split:
            await deployment.execution.prepare_execution([])

    async def prepare_grading(self, deployment: Deployment) -> None:
        await deployment.execution.prepare_execution(None)

    async def quiesce(self, deployment: Deployment) -> None:
        if deployment.split:
            await deployment.execution.quiesce()

    async def stop_execution(self, deployment: Deployment) -> None:
        return

    async def close(self, deployment: Deployment) -> None:
        return

    def verifier_config(self, deployment: Deployment) -> RuntimeConfig:
        return deployment.execution_config or deployment.owner.config

    @asynccontextmanager
    async def verifier(
        self, deployment: Deployment, config: RuntimeConfig, env: dict[str, str]
    ) -> AsyncIterator[Runtime]:
        async with provision_runtime(config, env=env) as runtime:
            yield runtime


class Deployment:
    """Live targets for one attempt, including its fresh verification targets."""

    def __init__(
        self,
        owner: Runtime,
        *,
        spec: ComposeSpec | None = None,
        execution: RuntimeConfig | None = None,
        env: dict[str, str] | None = None,
        config: DeploymentConfig | None = None,
        borrowed: Runtime | None = None,
        strategy: DeploymentStrategy | None = None,
    ):
        self.owner = owner
        self.spec = spec
        self.execution_config = execution
        self.env = dict(env or {})
        self.config = config or DeploymentConfig()
        self.borrowed = borrowed
        self.strategy = strategy or owner.deployment_strategy(
            self.config, execution, composed=spec is not None
        )
        if execution is not None:
            self.execution_config = self.strategy.resolve_execution(
                owner.config, execution
            )
        self.targets: dict[str, Runtime] = {}
        self.harness: Runtime | None = None
        self.execution: Runtime | None = None
        self._stack = AsyncExitStack()
        self._owned: set[int] = set()
        self._quiesced = False
        self._close_task: asyncio.Task[None] | None = None

    async def own(self, runtime: Runtime) -> Runtime:
        """Acquire a target; register cleanup before startup can partially fail."""
        if id(runtime) in self._owned:
            raise ValueError("runtime already owned by this deployment")
        self._owned.add(id(runtime))
        register(runtime)
        self._stack.push_async_callback(runtime.stop)
        await runtime.start()
        return runtime

    async def allocate(
        self, name: str, config: RuntimeConfig, *, env: dict[str, str] | None = None
    ) -> Runtime:
        """Allocate an independent named target. Strategies may add any number."""
        if name in self.targets:
            raise ValueError(f"target {name!r} already exists")
        runtime = make_runtime(config)
        runtime.env = dict(env or {})
        self.targets[name] = await self.own(runtime)
        return runtime

    async def enter(self, resource):
        """Own a provider resource such as a network or composed service group."""
        return await self._stack.enter_async_context(resource)

    async def start(self) -> None:
        try:
            await self.strategy.start(self)
            if self.harness is None or self.execution is None:
                raise ValueError(
                    "deployment strategy must bind harness and execution roles"
                )
        except BaseException:
            await self.close()
            raise

    @property
    def execution_name(self) -> str:
        return next(
            (name for name, target in self.targets.items() if target is self.execution),
            "main",
        )

    def target(self, name: str) -> Runtime:
        return self.targets[name]

    @property
    def split(self) -> bool:
        return self.execution is not self.harness

    async def connection(self, source: Runtime, target: Runtime) -> ExecutionConnection:
        return await self.strategy.connect(self, source, target)

    async def status(self, target: Runtime | None = None) -> TargetStatus:
        runtime = target or self.execution
        return (
            await runtime.execution_status()
            if runtime is not None
            else TargetStatus("unknown")
        )

    async def watch(self, target: Runtime | None = None) -> None:
        while True:
            status = await self.status(target)
            if status.state == "lost":
                raise TargetLost(status)
            await asyncio.sleep(1)

    async def run(self, operation, *, target: Runtime | None = None):
        """Supervise an operation on any target, including a fresh verifier."""
        work = asyncio.create_task(operation)
        watcher = asyncio.create_task(self.watch(target))
        try:
            done, _ = await asyncio.wait(
                (work, watcher), return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done:
                await watcher
            try:
                result = await work
            except Exception:
                status = await self.status(target)
                if status.state == "lost":
                    raise TargetLost(status) from None
                raise
            status = await self.status(target)
            if status.state == "lost":
                raise TargetLost(status)
            return result
        finally:
            work.cancel()
            watcher.cancel()
            await asyncio.gather(work, watcher, return_exceptions=True)

    @asynccontextmanager
    async def verifier(
        self, config: RuntimeConfig, env: dict[str, str]
    ) -> AsyncIterator[Runtime]:
        if "verifier" in self.targets:
            raise ValueError("verifier target name is already in use")
        # Parent ownership is registered before acquisition, including cancellation.
        stack = AsyncExitStack()
        self._stack.push_async_callback(stack.aclose)
        async with stack:
            runtime = await stack.enter_async_context(
                self.strategy.verifier(self, config, env)
            )
            self.targets["verifier"] = runtime
            try:
                yield runtime
            finally:
                self.targets.pop("verifier", None)

    async def quiesce(self) -> None:
        if not self._quiesced:
            await self.strategy.quiesce(self)
            self._quiesced = True

    async def stop_execution(self) -> None:
        await self.strategy.stop_execution(self)

    async def prepare_execution(self, routes: list[str]) -> None:
        await self.strategy.prepare_execution(self, routes)

    async def prepare_grading(self) -> None:
        await self.strategy.prepare_grading(self)

    async def close(self) -> None:
        async def release():
            try:
                await self.strategy.close(self)
            finally:
                await self._stack.aclose()

        if self._close_task is None:
            self._close_task = asyncio.create_task(release())
        await run_shielded(self._close_task)
